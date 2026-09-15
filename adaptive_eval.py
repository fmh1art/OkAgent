"""Data preparation and independent validation/final test for the adaptive agent."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import joblib
import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from job01_full_eval import source_profile, source_paths, source_identity
from react_runtime import save


def prepare_search(config, run_dir):
    folder = run_dir / "data"
    profile = source_profile(Path(config["data_dir"]), config["expected_count"], config["dimension"])
    marker = folder / "prepared.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["source_identity"] != profile["source_identity"]:
            raise ValueError("Source changed; use a new run directory")
        return saved
    folder.mkdir(exist_ok=True)
    n, dim = profile["row_count"], profile["embedding_dimension"]
    x, y, ids = np.empty((n, dim), np.float32), np.empty(n, np.int8), []
    vector_db, labels_db = source_paths(Path(config["data_dir"]))
    with duckdb.connect(str(vector_db), read_only=True, config={"threads": 2}) as con:
        con.execute("ATTACH '" + str(labels_db).replace("'", "''") + "' AS labels (READ_ONLY)")
        cursor = con.execute("""SELECT v.candidate_id,l.llm_pass,v.vec FROM candidate_segments v
            JOIN labels.llm_pass l USING(candidate_id) WHERE v.segment='unstructured' ORDER BY v.candidate_id""")
        start = 0
        while rows := cursor.fetchmany(512):
            block = np.asarray([r[2] for r in rows], np.float32)
            if block.shape != (len(rows), dim) or not np.isfinite(block).all():
                raise ValueError("Invalid embedding")
            x[start:start + len(rows)] = block
            y[start:start + len(rows)] = [r[1] for r in rows]
            ids.extend(r[0] for r in rows)
            start += len(rows)
    if start != n or source_identity(Path(config["data_dir"])) != profile["source_identity"]:
        raise ValueError("Data source changed during reading")
    ids = np.asarray(ids)
    develop, test = train_test_split(np.arange(n), test_size=config["test_size"], stratify=y, random_state=config["seed"])
    train, validation = train_test_split(develop, test_size=config["validation_size"], stratify=y[develop], random_state=config["seed"] + 1)
    splits = {"train": train, "validation": validation, "test": test}
    summaries = {}
    for name, indices in splits.items():
        if len(np.unique(y[indices])) != 2:
            raise ValueError("Every split needs both classes")
        np.savez(folder / f"{name}.npz", x=x[indices], y=y[indices], ids=ids[indices])
        summaries[name] = {"rows": len(indices), "positive": int(y[indices].sum())}
    if len(set(train) | set(validation) | set(test)) != n or sum(map(len, splits.values())) != n:
        raise ValueError("Overlapping or incomplete splits")
    with (folder / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "split", "label"])
        for name, indices in splits.items():
            writer.writerows((ids[i], name, int(y[i])) for i in indices)
    result = {**profile, "splits": summaries, "seed": config["seed"]}
    save(marker, result)
    return result


def scores_for(model, x):
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return np.asarray(model.predict_proba(x)[:, classes.index(1)], dtype=float)
    if hasattr(model, "decision_function"):
        return expit(np.asarray(model.decision_function(x), dtype=float))
    return np.asarray(model.predict(x), dtype=float)


def score_metrics(y, scores, threshold):
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid prediction scores")
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"rows": len(y), "positive": int(y.sum()), "threshold": threshold,
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def validate_candidate(request):
    run_dir, directory = Path(request["run_dir"]).resolve(), Path(request["round_dir"]).resolve()
    observation = request["observation"]
    original = Path(observation["model_file"])
    if not original.is_absolute():
        original = directory / original
    original = original.resolve()
    if not original.is_relative_to(run_dir) or not original.is_file():
        raise ValueError("model_file must be an existing model artifact inside this run")
    snapshot = directory / "validated_model.joblib"
    if original != snapshot:
        shutil.copyfile(original, snapshot)
    model = joblib.load(snapshot)
    clone(model)  # Final selection must support refitting without using holdout labels.
    threshold = float(observation.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    with np.load(run_dir / "data/validation.npz", allow_pickle=False) as data, threadpool_limits(limits=2):
        scores = scores_for(model, data["x"])
        metrics = score_metrics(data["y"], scores, threshold)
    result = {"status": "ok", "model_file": str(snapshot), "model_class": type(model).__name__,
              "threshold": threshold, "validation": metrics,
              "model_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "message": "Validation metrics independently recomputed; final test not accessed."}
    save(directory / "validation.json", result)
    return result


def finalize_model(request):
    run_dir = Path(request["run_dir"])
    complete = run_dir / "final_metrics.json"
    if complete.exists():
        return json.loads(complete.read_text())
    chosen = request["chosen"]
    path = Path(chosen["observation"]["model_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != chosen["observation"]["model_sha256"]:
        raise ValueError("Selected model changed after validation")
    model = clone(joblib.load(path))
    threshold = chosen["observation"]["threshold"]
    with np.load(run_dir / "data/train.npz") as a, np.load(run_dir / "data/validation.npz") as b:
        x, y = np.concatenate([a["x"], b["x"]]), np.concatenate([a["y"], b["y"]])
    with threadpool_limits(limits=2):
        model.fit(x, y)
    joblib.dump(model, run_dir / "final_model.joblib")
    with np.load(run_dir / "data/test.npz") as test, threadpool_limits(limits=2):
        scores = scores_for(model, test["x"])
        result = {"evaluation": "adaptive_validation_then_single_holdout_test",
                  "selected_round": chosen["round"], "validation": chosen["observation"]["validation"],
                  "train_rows_final": len(y), "test": score_metrics(test["y"], scores, threshold),
                  "reference_label": "Historical LLM label, not human hiring ground truth"}
        with (run_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "label", "prediction", "score"])
            writer.writerows((i, int(label), int(s >= threshold), float(s)) for i, label, s in zip(test["ids"], test["y"], scores))
    save(complete, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "finalize"])
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = validate_candidate(request) if args.action == "validate" else finalize_model(request)
    print(json.dumps(result, ensure_ascii=False))
