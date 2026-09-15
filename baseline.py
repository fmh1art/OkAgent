#!/usr/bin/env python3
"""Job01 baseline: unstructured embedding -> L2 -> LR -> one held-out test set."""

import argparse
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits


def load_data(data_dir):
    """Read one 2048-D vector per candidate; preserve all 34,761 source IDs."""
    folder = data_dir / "job-candidate-embedding-20260722"
    vector_db = folder / "hiring_job01_full_segvec.db"
    label_db = folder / "hiring_job01_llm_pass.db"
    with duckdb.connect(str(vector_db), read_only=True, config={"threads": 2}) as con:
        con.execute("ATTACH '" + str(label_db).replace("'", "''") + "' AS labels (READ_ONLY)")
        total, unique = con.execute(
            "SELECT count(*),count(DISTINCT candidate_id) FROM labels.llm_pass"
        ).fetchone()
        if total != 34761 or unique != total:
            raise ValueError("Expected exactly 34,761 unique labeled candidates")
        counts = dict(con.execute("SELECT llm_pass,count(*) FROM labels.llm_pass GROUP BY 1").fetchall())
        if counts != {0: 34379, 1: 382}:
            raise ValueError(f"Unexpected job01 label distribution: {counts}")
        rows, unique = con.execute("""SELECT count(*),count(DISTINCT candidate_id)
            FROM candidate_segments WHERE segment='unstructured'""").fetchone()
        if rows != total or unique != total:
            raise ValueError("Expected exactly one unstructured vector per candidate")
        cursor = con.execute("""SELECT v.candidate_id,l.llm_pass,v.vec
            FROM candidate_segments v JOIN labels.llm_pass l USING(candidate_id)
            WHERE v.segment='unstructured' ORDER BY v.candidate_id""")
        x = np.empty((total, 2048), dtype=np.float32)
        y = np.empty(total, dtype=np.int8)
        ids = []
        offset = 0
        while batch := cursor.fetchmany(512):
            vectors = np.asarray([r[2] for r in batch], dtype=np.float32)
            if vectors.shape != (len(batch), 2048) or not np.isfinite(vectors).all():
                raise ValueError("Invalid embedding shape or non-finite values")
            end = offset + len(batch)
            x[offset:end] = vectors
            y[offset:end] = [r[1] for r in batch]
            ids.extend(r[0] for r in batch)
            offset = end
    if offset != total or len(set(ids)) != total:
        raise ValueError("Vector/label IDs do not match completely")
    return np.asarray(ids), x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "datasets/20260722")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    output = args.output_dir or Path(__file__).resolve().parent / "runs" / (
        "baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    print("Loading all 34,761 job01 candidates...", flush=True)
    ids, x, y = load_data(args.data_dir)
    train, test = train_test_split(
        np.arange(len(y)), test_size=args.test_size, random_state=args.seed, stratify=y
    )
    if np.intersect1d(train, test).size or len(train) + len(test) != len(y):
        raise ValueError("Invalid train/test partition")
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Both splits must contain positive and negative candidates")
    print(f"Training on {len(train)} candidates; testing on {len(test)}...", flush=True)

    # Same feature normalization and LR settings as smoke_test.py.
    # Normalizer is per row; it does not estimate any statistics from test data.
    model = make_pipeline(
        Normalizer(norm="l2"),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x[train], y[train])
        prediction = model.predict(x[test])
        score = model.predict_proba(x[test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y[test], prediction, labels=[0, 1]).ravel()
    result = {
        "status": "ok", "evaluation": "single_stratified_train_test_split", "job": "job01",
        "full_rows": len(y), "train_rows": len(train), "test_rows": len(test),
        "train_positive": int(y[train].sum()), "test_positive": int(y[test].sum()),
        "train_test_overlap": 0, "test_size": args.test_size, "seed": args.seed,
        "feature": "unstructured_embedding", "embedding_dimension": x.shape[1],
        "model": "L2 + LogisticRegression(C=1, class_weight=balanced, max_iter=2000)",
        "threshold": 0.5,
        "precision": float(precision_score(y[test], prediction, zero_division=0)),
        "recall": float(recall_score(y[test], prediction, zero_division=0)),
        "f1": float(f1_score(y[test], prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y[test], score)),
        "roc_auc": float(roc_auc_score(y[test], score)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "note": "Test metrics compare against historical LLM labels, not human hiring ground truth.",
    }
    with (output / "split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "split", "label"])
        for name, indices in [("train", train), ("test", test)]:
            writer.writerows((ids[i], name, int(y[i])) for i in indices)
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "label", "prediction", "score"])
        writer.writerows((ids[i], int(y[i]), int(p), float(s)) for i, p, s in zip(test, prediction, score))
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
