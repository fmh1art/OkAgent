"""Train and evaluate three section proxies plus two learned routers on job01.

The script accepts a frozen training-ID set or an existing teacher ledger.  It
never uses unsampled historical labels for fitting; those labels are opened only
for final strict-unsampled diagnostics and repeated recall calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score

from okagent.job01_io import resolve_job01
from okagent.recall_calibration import repeated_calibration, select_recall_threshold
from okagent.section_router import DEFAULT_EXPERT_GROUPS, SectionRouterEnsemble


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def load_train_ids(args) -> list[str]:
    if args.train_ids:
        rows = read_json(args.train_ids)
        ids = [row["candidate_id"] if isinstance(row, dict) else row for row in rows]
    else:
        with sqlite3.connect(args.train_ledger) as db:
            ids = [row[0] for row in db.execute(
                "SELECT candidate_id FROM labels WHERE status='complete' ORDER BY candidate_id")]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(cid, str) for cid in ids):
        raise ValueError("training IDs must be non-empty unique strings")
    return ids


def load_embedding_groups(data_path: Path, candidate_ids: list[str], groups: dict):
    """Read selected sections once and return candidate-level mean embeddings."""
    index = {cid: pos for pos, cid in enumerate(candidate_ids)}
    segment_to_group = {}
    for name, segments in groups.items():
        for segment in segments:
            segment_to_group.setdefault(segment, []).append(name)
    wanted_segments = sorted(segment_to_group)
    placeholders = ",".join("?" for _ in wanted_segments)
    with duckdb.connect(str(data_path), read_only=True) as db:
        cursor = db.execute(
            f"SELECT candidate_id,segment,vec FROM candidate_segments "
            f"WHERE segment IN ({placeholders}) ORDER BY candidate_id", wanted_segments)
        first = cursor.fetchone()
        if first is None:
            raise ValueError("no section embeddings found")
        dimension = len(first[2])
        matrices = {name: np.zeros((len(candidate_ids), dimension), dtype=np.float32)
                    for name in groups}
        counts = {name: np.zeros(len(candidate_ids), dtype=np.int16) for name in groups}

        def add(row):
            cid, segment, vector = row
            pos = index.get(cid)
            if pos is None or vector is None:
                return
            value = np.asarray(vector, dtype=np.float32)
            if len(value) != dimension:
                raise ValueError("embedding dimensions do not match")
            for name in segment_to_group[segment]:
                matrices[name][pos] += value
                counts[name][pos] += 1

        add(first)
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            for row in rows:
                add(row)
    presence = {}
    for name in groups:
        presence[name] = counts[name] > 0
        matrices[name][presence[name]] /= counts[name][presence[name], None]
    return matrices, presence


def load_multitask_labels(path: Path, train_ids: list[str], threshold: float):
    by_id = {row["candidate_id"]: row for row in read_json(path)}
    if not set(train_ids).issubset(by_id):
        raise ValueError("batch labels do not cover every training candidate")
    overall = np.asarray([by_id[cid]["overall"] >= threshold for cid in train_ids],
                         dtype=np.int8)
    expert = {
        "global": np.asarray([by_id[cid]["overall"] for cid in train_ids], dtype=np.float64),
        "experience": np.asarray([
            (by_id[cid]["experience"] + by_id[cid]["projects"]) / 2
            for cid in train_ids], dtype=np.float64),
        "credentials": np.asarray([
            (by_id[cid]["education"] + by_id[cid]["technical_skills"] +
             by_id[cid]["research"]) / 3
            for cid in train_ids], dtype=np.float64),
    }
    return overall, expert


def method_metrics(labels: np.ndarray, scores: np.ndarray, *, seeds: list[int]) -> dict:
    result = {"average_precision": float(average_precision_score(labels, scores))}
    for target in (0.8, 0.9):
        threshold, oracle = select_recall_threshold(labels, scores, target)
        result[f"p_at_r{round(target * 100)}"] = {
            **oracle, "threshold": threshold,
        }
        result[f"calibration_r{round(target * 100)}"] = repeated_calibration(
            labels, scores, target_recall=target, seeds=seeds,
            calibration_fraction=0.2, require_lower_bound=False,
        )
        result[f"lower_bound_calibration_r{round(target * 100)}"] = repeated_calibration(
            labels, scores, target_recall=target, seeds=seeds,
            calibration_fraction=0.2, require_lower_bound=True,
        )
    return result


def run(args) -> dict:
    data_path, truth_path, _, _ = resolve_job01(args.config)
    with duckdb.connect(str(data_path), read_only=True) as db:
        candidate_ids = [row[0] for row in db.execute(
            "SELECT DISTINCT candidate_id FROM candidate_segments ORDER BY candidate_id").fetchall()]
    with duckdb.connect(str(truth_path), read_only=True) as db:
        truth = dict(db.execute("SELECT candidate_id,llm_pass FROM llm_pass").fetchall())
    if set(candidate_ids) != set(truth):
        raise ValueError("historical labels do not exactly cover the candidate population")
    train_ids = load_train_ids(args)
    if not set(train_ids).issubset(truth):
        raise ValueError("training IDs are outside job01")
    if args.expected_train_count and len(train_ids) != args.expected_train_count:
        raise ValueError(f"expected {args.expected_train_count} training IDs, got {len(train_ids)}")

    groups = {key: tuple(value) for key, value in DEFAULT_EXPERT_GROUPS.items()}
    matrices, presence = load_embedding_groups(data_path, candidate_ids, groups)
    positions = {cid: index for index, cid in enumerate(candidate_ids)}
    train_index = np.asarray([positions[cid] for cid in train_ids])
    if args.batch_labels:
        labels, expert_labels = load_multitask_labels(
            args.batch_labels, train_ids, args.teacher_threshold)
        label_source = "batch_teacher_multitask"
    else:
        labels = np.asarray([truth[cid] for cid in train_ids], dtype=np.int8)
        expert_labels = None
        label_source = "historical_binary"
    train_features = {name: matrix[train_index] for name, matrix in matrices.items()}
    train_presence = {name: mask[train_index] for name, mask in presence.items()}
    ensemble = SectionRouterEnsemble(folds=args.folds, seed=args.seed).fit(
        train_features, train_presence, labels, expert_labels=expert_labels,
    )
    all_scores = ensemble.predict_all(matrices, presence)

    train_set = set(train_ids)
    unsampled_index = np.asarray([index for index, cid in enumerate(candidate_ids)
                                  if cid not in train_set])
    unsampled_labels = np.asarray([truth[candidate_ids[index]] for index in unsampled_index],
                                  dtype=np.int8)
    seeds = list(range(args.calibration_seed_start,
                       args.calibration_seed_start + args.calibration_seeds))
    metrics = {name: method_metrics(unsampled_labels, scores[unsampled_index], seeds=seeds)
               for name, scores in all_scores.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "candidate_ids.json", candidate_ids)
    np.savez_compressed(args.output / "scores.npz", **all_scores)
    with (args.output / "model.pkl").open("wb") as handle:
        pickle.dump(ensemble, handle)
    report = {
        "protocol": {
            "job": "job01", "model_seed": args.seed, "folds": args.folds,
            "label_source": label_source, "train_count": len(train_ids),
            "train_id_source": str((args.train_ids or args.train_ledger).resolve()),
            "train_id_sha256": hashlib.sha256(
                "\n".join(sorted(train_ids)).encode()).hexdigest(),
            "train_positive": int(labels.sum()), "unsampled_count": len(unsampled_index),
            "unsampled_positive": int(unsampled_labels.sum()),
            "expert_groups": {key: list(value) for key, value in groups.items()},
            "training_ids_excluded_from_evaluation": True,
            "calibration_seeds": seeds,
        },
        "training": ensemble.training_summary_,
        "metrics": metrics,
    }
    write_json(args.output / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-ids", type=Path)
    source.add_argument("--train-ledger", type=Path)
    p.add_argument("--batch-labels", type=Path,
                   help="Optional labels.json from batch_label_job01 for expert-specific supervision")
    p.add_argument("--teacher-threshold", type=float, default=0.5)
    p.add_argument("--expected-train-count", type=int, default=2_000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("_config/hiring.json"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--calibration-seeds", type=int, default=20)
    p.add_argument("--calibration-seed-start", type=int, default=10_000)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.folds < 2 or args.calibration_seeds <= 0:
        raise ValueError("folds and calibration seed count must be positive")
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
