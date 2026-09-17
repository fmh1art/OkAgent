import json

import duckdb

from okagent.qwen_causal_proxy import QwenCausalConfig, QwenCausalProxy, _select_demonstrations
from okagent.prompts import PROXY_SKILL


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


def test_qwen_proxy_uses_training_demonstrations_and_serializes(tmp_path):
    database = tmp_path / "data.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR, segment VARCHAR, text VARCHAR)")
        con.executemany("INSERT INTO candidate_segments VALUES (?, ?, ?)", [
            ("a", "unstructured", "good resume"),
            ("b", "work", "bad resume"),
            ("b", "projects", "project text"),
        ])
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"job_title": "engineer", "must_have_qualifications": ["good"]}))
    proxy = QwenCausalProxy(database, job, config=QwenCausalConfig(max_demonstrations=2),
                            cache_dir=tmp_path / "cache")
    proxy.fit({"a": 1, "b": 0})
    assert proxy.demonstrations == [("a", 1), ("b", 0)]
    prompt = proxy._prompt("a", FakeTokenizer())
    assert "A=不匹配，B=匹配" in prompt
    artifact = tmp_path / "proxy.pkl"
    proxy.save(artifact, threshold=0.42)
    loaded, threshold = QwenCausalProxy.load(artifact, tmp_path, cache_dir=tmp_path / "cache")
    assert loaded.demonstrations == proxy.demonstrations and threshold == 0.42


def test_prompts_compare_qwen_as_model_not_embedding():
    assert "QwenCausalProxy" in PROXY_SKILL
    assert "不是 Qwen embedding + LR" in PROXY_SKILL
    assert "相同的独立验证 ID" in PROXY_SKILL
    assert _select_demonstrations({"n": 0, "p": 1}, 2) == [("p", 1), ("n", 0)]
