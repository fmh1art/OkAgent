import json
from types import SimpleNamespace

import duckdb
import numpy as np

from benchmarks.multi_proxy_router_job01 import load_multitask_labels, run


def test_multi_proxy_job01_end_to_end_writes_strict_unsampled_report(tmp_path):
    rng = np.random.default_rng(11)
    raw = tmp_path / "raw"
    descriptions = raw / "job-description-20260722"
    embeddings = raw / "job-candidate-embedding-20260722"
    descriptions.mkdir(parents=True)
    embeddings.mkdir()
    (descriptions / "01_synthetic.json").write_text(json.dumps({
        "job_title": "synthetic", "must_have_qualifications": ["signal"],
    }), encoding="utf-8")
    count, dimension = 240, 12
    ids = [f"c{index:04d}" for index in range(count)]
    latent = rng.normal(size=count)
    labels = (latent + rng.normal(scale=.7, size=count) > .7).astype(np.int8)
    data_path = embeddings / "hiring_job01_full_segvec.db"
    with duckdb.connect(str(data_path)) as db:
        db.execute(f"CREATE TABLE candidate_segments("
                   f"candidate_id VARCHAR,segment VARCHAR,text VARCHAR,vec FLOAT[{dimension}])")
        rows = []
        for index, cid in enumerate(ids):
            for segment, signal in (
                ("unstructured", 1.0), ("education", .5), ("applications", .35),
                ("projects", .8), ("work", .7),
            ):
                if segment in ("projects", "work") and index % 4 == 0:
                    continue
                vector = rng.normal(size=dimension).astype(np.float32)
                vector[0] += latent[index] * signal
                rows.append((cid, segment, segment, vector.tolist()))
        db.executemany("INSERT INTO candidate_segments VALUES (?,?,?,?)", rows)
    truth_path = embeddings / "hiring_job01_llm_pass.db"
    with duckdb.connect(str(truth_path)) as db:
        db.execute("CREATE TABLE llm_pass(candidate_id VARCHAR,llm_pass TINYINT)")
        db.executemany("INSERT INTO llm_pass VALUES (?,?)", zip(ids, labels.tolist()))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "raw_root": "raw", "job": "job01", "as_of": "2026-07-22",
    }), encoding="utf-8")
    positive = [cid for cid, label in zip(ids, labels) if label][:30]
    negative = [cid for cid, label in zip(ids, labels) if not label][:30]
    train_ids = tmp_path / "train_ids.json"
    train_ids.write_text(json.dumps(positive + negative), encoding="utf-8")
    output = tmp_path / "output"
    report = run(SimpleNamespace(
        config=config, train_ids=[train_ids], train_ledger=None, batch_labels=None,
        teacher_threshold=.5, expected_train_count=60, output=output,
        folds=3, seed=11, calibration_seed_start=100, calibration_seeds=3,
    ))
    assert report["protocol"]["train_count"] == 60
    assert len(report["protocol"]["train_id_sha256"]) == 64
    assert report["protocol"]["model_seed"] == 11
    assert report["protocol"]["unsampled_count"] == 180
    assert report["protocol"]["training_ids_excluded_from_evaluation"] is True
    assert {"global", "experience", "credentials", "expert_mean",
            "stacking_router", "embedding_router"}.issubset(report["metrics"])
    assert (output / "report.json").is_file()
    assert (output / "scores.npz").is_file()
    assert (output / "model.pkl").is_file()


def test_multiround_batch_labels_merge_without_rounding(tmp_path):
    first = tmp_path / "round1.json"
    second = tmp_path / "round2.json"
    dimensions = {
        "overall": .73, "education": .4, "experience": .8,
        "technical_skills": .7, "projects": .6, "research": .2,
        "evidence_quality": .9, "missing_requirements": [],
    }
    first.write_text(json.dumps([{"candidate_id": "a", **dimensions}]), encoding="utf-8")
    second.write_text(json.dumps([{"candidate_id": "b", **dimensions}]), encoding="utf-8")
    hard, experts = load_multitask_labels([first, second], ["a", "b"], .5)
    assert hard.tolist() == [1, 1]
    assert experts["global"].tolist() == [.73, .73]
    assert experts["experience"].tolist() == [.7, .7]
