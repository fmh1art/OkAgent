import json

import duckdb
import numpy as np

from okagent.qwen_proxy import QwenFeatureConfig, load_qwen_features, load_resume_texts


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([[len(text), text.count("project"), 1, 2, 3, 4] for text in texts], dtype=np.float32)


def test_qwen_features_are_section_balanced_and_cached(tmp_path):
    database = tmp_path / "data.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR, segment VARCHAR, text VARCHAR)")
        con.executemany("INSERT INTO candidate_segments VALUES (?,?,?)", [
            ("b", "work", "w" * 200), ("a", "basic", "a" * 200),
            ("a", "projects", "project " * 40), ("b", "projects", "project b")])
    ids, texts = load_resume_texts(database, max_chars=120)
    assert ids == ["a", "b"] and all("[projects]" in text for text in texts)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"job_title": "engineer"}))
    config = QwenFeatureConfig(dimensions=4, max_chars=120, batch_size=2)
    first = FakeEncoder()
    ids, vectors, query, metadata = load_qwen_features(
        database, job, config=config, cache_dir=tmp_path / "cache", encoder=first)
    assert ids == ["a", "b"] and vectors.shape == (2, 4) and query.shape == (4,)
    assert not metadata["cached"] and "Instruct:" in first.calls[-1][0]
    second = FakeEncoder()
    cached = load_qwen_features(database, job, config=config, cache_dir=tmp_path / "cache", encoder=second)
    assert cached[3]["cached"] and not second.calls
    assert np.allclose(cached[1], vectors)
