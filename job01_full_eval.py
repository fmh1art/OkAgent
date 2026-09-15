"""Read-only job01 DuckDB adapter and candidate-level out-of-fold evaluation.

No API requests, candidate text exports, or LLM-generated metric calculations.
The five operators operate on the same immutable split and saved model settings.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits

ORDER = ["Partition", "Sample", "Label", "Proxy", "Deploy"]
DEFAULT_MODEL = {"name": "LogisticRegression", "feature": "unstructured_embedding",
                 "C": 1.0, "class_weight": "balanced", "max_iter": 2000}


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def source_paths(data_dir: Path) -> tuple[Path, Path]:
    folder = data_dir / "job-candidate-embedding-20260722"
    return (folder / "hiring_job01_full_segvec.db", folder / "hiring_job01_llm_pass.db")


def source_identity(data_dir: Path) -> list[dict]:
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns} for p in source_paths(data_dir)]


def source_profile(data_dir: Path, expected_count=34761, dimension=2048) -> dict:
    """Validate candidate grain and source IDs before any training or API calls."""
    vector_db, label_db = source_paths(data_dir)
    with duckdb.connect(str(vector_db), read_only=True, config={"threads": 2}) as con:
        con.execute("ATTACH '" + str(label_db).replace("'", "''") + "' AS labels (READ_ONLY)")
        label_counts = con.execute("SELECT llm_pass,count(*) FROM labels.llm_pass GROUP BY 1 ORDER BY 1").fetchall()
        if {r[0] for r in label_counts} != {0, 1}:
            raise ValueError("Labels must contain exactly 0 and 1, with no NULLs")
        count = sum(r[1] for r in label_counts)
        unique = con.execute("SELECT count(DISTINCT candidate_id) FROM labels.llm_pass").fetchone()[0]
        if count != expected_count or unique != count:
            raise ValueError(f"Expected {expected_count} unique labeled candidates; rows={count}, unique={unique}")
        rows, ids = con.execute("SELECT count(*),count(DISTINCT candidate_id) FROM candidate_segments WHERE segment='unstructured'").fetchone()
        if rows != count or ids != count:
            raise ValueError("Each labeled candidate needs exactly one unstructured vector")
        unmatched = con.execute("""SELECT count(*) FROM
            (SELECT candidate_id FROM candidate_segments WHERE segment='unstructured') v
            FULL OUTER JOIN labels.llm_pass l USING(candidate_id)
            WHERE v.candidate_id IS NULL OR l.candidate_id IS NULL""").fetchone()[0]
        if unmatched:
            raise ValueError("Vector and label candidate IDs differ")
        bad = con.execute("""SELECT count(*) FROM candidate_segments WHERE segment='unstructured'
            AND (vec IS NULL OR array_length(vec) != ?)""", [dimension]).fetchone()[0]
        if bad:
            raise ValueError("Missing vectors or incorrect embedding dimension")
        segments = con.execute("SELECT segment,count(*) FROM candidate_segments GROUP BY 1 ORDER BY 1").fetchall()
    return {"job": "job01", "row_count": count, "label_counts": {str(k): v for k, v in label_counts},
            "embedding_dimension": dimension, "segment_counts": dict(segments),
            "feature": "unstructured_embedding", "label_source": "precomputed_llm_is_match",
            "source_identity": source_identity(data_dir)}


def prepare(data_dir: Path, run_dir: Path, folds: int, seed: int,
            expected_count=34761, dimension=2048) -> dict:
    profile = source_profile(data_dir, expected_count, dimension)
    n = profile["row_count"]
    if not 2 <= folds <= min(profile["label_counts"].values()):
        raise ValueError("Number of folds must be between 2 and the minority-class count")
    x = np.lib.format.open_memmap(run_dir / "features.npy", mode="w+", dtype=np.float32, shape=(n, dimension))
    ids, labels = [], []
    vector_db, label_db = source_paths(data_dir)
    with duckdb.connect(str(vector_db), read_only=True, config={"threads": 2}) as con:
        con.execute("ATTACH '" + str(label_db).replace("'", "''") + "' AS labels (READ_ONLY)")
        cursor = con.execute("""SELECT v.candidate_id,l.llm_pass,v.vec FROM candidate_segments v
            JOIN labels.llm_pass l USING(candidate_id) WHERE v.segment='unstructured'
            ORDER BY v.candidate_id""")
        start = 0
        while batch := cursor.fetchmany(512):
            values = np.asarray([r[2] for r in batch], dtype=np.float32)
            if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
                raise ValueError("Non-finite or zero embedding detected")
            x[start:start + len(batch)] = values
            ids.extend(r[0] for r in batch)
            labels.extend(r[1] for r in batch)
            start += len(batch)
    if start != n or source_identity(data_dir) != profile["source_identity"]:
        raise ValueError("Source changed while preparing features")
    x.flush()
    del x
    y = np.asarray(labels, dtype=np.int8)
    assignment = np.full(n, -1, dtype=np.int16)
    summaries = []
    for fold, (train, test) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), y)):
        if np.intersect1d(train, test).size:
            raise ValueError("Train/test overlap")
        assignment[test] = fold
        summaries.append({"fold": fold, "train_rows": len(train), "test_rows": len(test),
                          "train_positive": int(y[train].sum()), "test_positive": int(y[test].sum())})
    np.save(run_dir / "labels.npy", y)
    np.save(run_dir / "folds.npy", assignment)
    write_json(run_dir / "candidate_ids.json", ids)
    fingerprint = hashlib.sha256(json.dumps([ids, labels, assignment.tolist()], separators=(",", ":")).encode()).hexdigest()
    result = {**profile, "folds": summaries, "n_folds": folds, "seed": seed,
              "split_fingerprint": fingerprint, "status": "ok"}
    write_json(run_dir / "dataset.json", result)
    return result


def model_settings(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("proxy_model must be an object")
    raw = dict(raw)
    nested = raw.pop("hyperparams", {})
    if not isinstance(nested, dict) or set(raw).intersection(nested):
        raise ValueError("Invalid or duplicate hyperparameters")
    raw.update(nested)
    name = raw.get("name", "LogisticRegression")
    if raw.get("feature", "unstructured_embedding") != "unstructured_embedding":
        raise ValueError("Full evaluation currently uses unstructured_embedding")
    if name == "LogisticRegression":
        result = {**DEFAULT_MODEL, **raw}
        if set(result) - {"name", "feature", "C", "class_weight", "max_iter", "random_state", "solver"}:
            raise ValueError("Unsupported LogisticRegression parameter")
        if not 0.0001 <= float(result["C"]) <= 100 or not 100 <= int(result["max_iter"]) <= 5000:
            raise ValueError("C or max_iter outside supported range")
        if result.get("solver", "lbfgs") != "lbfgs":
            raise ValueError("Use solver=lbfgs for this evaluation")
    elif name == "RandomForestClassifier":
        result = {"name": name, "feature": "unstructured_embedding", "n_estimators": 200,
                  "max_depth": 16, "min_samples_leaf": 2, "class_weight": "balanced", **raw}
        if set(result) - {"name", "feature", "n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state"}:
            raise ValueError("Unsupported RandomForest parameter")
        if not 10 <= int(result["n_estimators"]) <= 300 or not 1 <= int(result["max_depth"]) <= 30 or not 1 <= int(result["min_samples_leaf"]) <= 100:
            raise ValueError("RandomForest parameters outside supported range")
    else:
        raise ValueError("Supported models: LogisticRegression, RandomForestClassifier")
    if result["class_weight"] not in (None, "balanced"):
        raise ValueError("class_weight must be null or balanced")
    return result


def build_model(settings: dict, seed: int):
    params = {k: v for k, v in settings.items() if k not in {"name", "feature"}}
    params["random_state"] = seed
    if settings["name"] == "LogisticRegression":
        estimator = LogisticRegression(**params)
    else:
        estimator = RandomForestClassifier(**params, n_jobs=2)
    return make_pipeline(Normalizer(norm="l2"), estimator)


def metrics(y, scores) -> dict:
    pred = (scores >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"rows": len(y), "positive": int(y.sum()), "predicted_positive": int(pred.sum()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)), "accuracy": float((pred == y).mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def train_folds(run_dir: Path, raw_settings: dict) -> dict:
    data = json.loads((run_dir / "dataset.json").read_text())
    settings = model_settings(raw_settings)
    settings["random_state"] = data["seed"]
    config = {"model": settings, "split_fingerprint": data["split_fingerprint"], "seed": data["seed"]}
    config_path = run_dir / "model_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Model settings are locked for this run; create a new run directory to change them")
    write_json(config_path, config)
    x = np.load(run_dir / "features.npy", mmap_mode="r")
    y, assignment = np.load(run_dir / "labels.npy"), np.load(run_dir / "folds.npy")
    for fold in range(data["n_folds"]):
        path = run_dir / f"model_fold_{fold}.joblib"
        if path.exists():
            continue
        write_json(run_dir / "training_progress.json", {"fold": fold, "total_folds": data["n_folds"], "status": "training"})
        train = np.flatnonzero(assignment != fold)
        model = build_model(settings, data["seed"])
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x[train], y[train])
        temporary = path.with_suffix(".tmp")
        joblib.dump({"model": model, "train_indices": train,
                     "split_fingerprint": data["split_fingerprint"], "config": config}, temporary)
        temporary.replace(path)
        write_json(run_dir / "training_progress.json", {"fold": fold, "total_folds": data["n_folds"], "status": "fold_complete"})
    return {"status": "ok", "model": settings, "trained_folds": data["n_folds"],
            "training_only": True, "test_metrics_not_used_for_selection": True}


def evaluate(run_dir: Path) -> dict:
    import csv
    data = json.loads((run_dir / "dataset.json").read_text())
    config = json.loads((run_dir / "model_config.json").read_text())
    x = np.load(run_dir / "features.npy", mmap_mode="r")
    y, assignment = np.load(run_dir / "labels.npy"), np.load(run_dir / "folds.npy")
    ids = json.loads((run_dir / "candidate_ids.json").read_text())
    scores = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=np.int8)
    folds = []
    for fold in range(data["n_folds"]):
        artifact = joblib.load(run_dir / f"model_fold_{fold}.joblib")
        train, test = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        if (artifact["split_fingerprint"] != data["split_fingerprint"] or artifact["config"] != config
                or not np.array_equal(artifact["train_indices"], train)
                or np.intersect1d(artifact["train_indices"], test).size):
            raise ValueError("Model/split mismatch or training leakage")
        with threadpool_limits(limits=2):
            scores[test] = artifact["model"].predict_proba(x[test])[:, 1]
        coverage[test] += 1
        folds.append({"fold": fold, **metrics(y[test], scores[test])})
    if not np.all(coverage == 1) or not np.isfinite(scores).all() or len(set(ids)) != len(y):
        raise ValueError("Every candidate must receive exactly one held-out prediction")
    pred = (scores >= 0.5).astype(np.int8)
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "fold", "llm_label", "predicted_label", "score"])
        writer.writerows((cid, int(fold), int(label), int(p), float(s)) for cid, fold, label, p, s in zip(ids, assignment, y, pred, scores))
    result = {"status": "complete", "job": "job01", "evaluation": "stratified_candidate_out_of_fold",
              "n_folds": data["n_folds"], "seed": data["seed"], "threshold": 0.5,
              "model": config["model"], "normalization": "per-candidate L2",
              "coverage": {"unique_candidates": len(ids), "predictions_per_candidate": 1, "train_test_overlap": 0},
              "overall": metrics(y, scores), "per_fold": folds,
              "majority_baseline": {"accuracy": float((y == 0).mean()), "recall": 0.0, "f1": 0.0},
              "reference_label": "LLM matching result, not human or hiring ground truth",
              "split_fingerprint": data["split_fingerprint"], "source_identity": data["source_identity"],
              "versions": {k: importlib.metadata.version(k) for k in ["duckdb", "numpy", "scikit-learn", "joblib"]}}
    write_json(run_dir / "metrics.json", result)
    return result


def execute_operator(payload: dict) -> dict:
    run_dir = Path(payload["run_dir"])
    config = payload["config"]
    op = payload["operator"]
    state_path = run_dir / "operator_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": [], "observations": {}}
    name = op["operator"]
    if name in state["completed"]:
        return state["observations"][name]
    if len(state["completed"]) == len(ORDER) or name != ORDER[len(state["completed"])]:
        raise ValueError("Required order: Partition -> Sample -> Label -> Proxy -> Deploy")
    if name == "Partition":
        if op["params"]["partition_method"] != "stratified_kfold":
            raise ValueError("Use partition_method=stratified_kfold")
        obs = prepare(Path(config["data_dir"]), run_dir, config["folds"], config["seed"],
                      config["expected_count"], config["dimension"])
    else:
        data = json.loads((run_dir / "dataset.json").read_text())
        if name == "Sample":
            if op["params"] != {"sample_method": "all_train", "sample_number": data["row_count"]}:
                raise ValueError("Full evaluation must keep all candidates across folds; no subsampling")
            obs = {"status": "ok", "sample_method": "all_train", "candidate_universe": data["row_count"],
                   "folds": data["folds"], "note": "Each fold uses all its training candidates"}
        elif name == "Label":
            obs = {"status": "ok", "label_source": "precomputed_llm_label", "label_counts": data["label_counts"],
                   "external_label_calls": 0}
        elif name == "Proxy":
            obs = train_folds(run_dir, op["params"]["proxy_model"])
        else:
            obs = evaluate(run_dir)
    state["completed"].append(name)
    state["observations"][name] = obs
    write_json(state_path, state)
    return obs
