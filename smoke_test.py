#!/usr/bin/env python3
"""Validate the demo package and run an unstructured-embedding LR smoke test."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parent


def load_jsonl(split):
    rows = []
    for file_path in sorted((ROOT / split).glob("*.jsonl")):
        with file_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def main():
    full = load_jsonl("full")
    train = load_jsonl("train")
    test = load_jsonl("test")
    train_ids = {row["candidate_id"] for row in train}
    test_ids = {row["candidate_id"] for row in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {row["candidate_id"] for row in full}
    assert set(row["label"] for row in train) == {0, 1}
    assert set(row["label"] for row in test) == {0, 1}

    parquet = str(ROOT / "embeddings" / "embeddings_00001.parquet")
    con = duckdb.connect()
    embedding_rows = con.execute(
        "select embedding_id, text_sha256, embedding from read_parquet(?)",
        [parquet],
    ).fetchall()
    vectors = {row[0]: np.asarray(row[2], dtype=np.float32) for row in embedding_rows}
    parquet_hashes = {row[0]: row[1] for row in embedding_rows}

    referenced = set()
    for row in full:
        for attribute in row["attributes"].values():
            emb_id = attribute["embedding_id"]
            referenced.add(emb_id)
            assert emb_id in vectors
            assert parquet_hashes[emb_id] == attribute["text_sha256"]
            assert hashlib.sha256(attribute["text"].encode()).hexdigest() == attribute["text_sha256"]
            assert vectors[emb_id].shape == (2048,)
    assert referenced == set(vectors)

    def matrix(rows):
        x = np.vstack([
            vectors[row["attributes"]["unstructured"]["embedding_id"]]
            for row in rows
        ])
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        y = np.asarray([row["label"] for row in rows], dtype=np.int8)
        return x / norms, y

    x_train, y_train = matrix(train)
    x_test, y_test = matrix(test)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    result = {
        "status": "ok",
        "full_rows": len(full),
        "train_rows": len(train),
        "test_rows": len(test),
        "embedding_rows": len(embedding_rows),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "note": "Metrics only prove the pipeline runs; this sampled demo is not a benchmark.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
