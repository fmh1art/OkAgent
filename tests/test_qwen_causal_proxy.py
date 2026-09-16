import json
from types import SimpleNamespace

import duckdb
import numpy as np
import torch

from okagent.qwen_causal_proxy import QwenCausalConfig, QwenCausalProxy
from okagent.prompts import PROXY_SKILL


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, value, **kwargs):
        return {"A": [1], "B": [2]}[value]

    def __call__(self, prompts, **kwargs):
        markers = [9 if "当前候选人简历：\n[resume] good" in prompt else 8 for prompt in prompts]
        return {"input_ids": torch.tensor([[3, marker] for marker in markers]),
                "attention_mask": torch.ones((len(markers), 2), dtype=torch.long)}


class FakeModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, logits_to_keep=None, **kwargs):
        assert logits_to_keep == 1
        self.calls += 1
        logits = torch.zeros((len(input_ids), 1, 4))
        for index, marker in enumerate(input_ids[:, -1]):
            logits[index, 0, 2 if marker == 9 else 1] = 4
        return SimpleNamespace(logits=logits)


def test_qwen_causal_model_is_the_classifier_and_scores_are_cached(tmp_path):
    database = tmp_path / "data.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE candidate_segments(candidate_id VARCHAR, segment VARCHAR, text VARCHAR)")
        con.executemany("INSERT INTO candidate_segments VALUES (?,'resume',?)",
                        [("a", "good"), ("b", "bad"), ("c", "good")])
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"job_title": "engineer", "must_have_qualifications": ["good"]}))
    tokenizer, model = FakeTokenizer(), FakeModel()
    proxy = QwenCausalProxy(database, job,
                            config=QwenCausalConfig(batch_size=2, max_demonstrations=2),
                            cache_dir=tmp_path / "cache", tokenizer=tokenizer, model=model)
    proxy.fit({"a": 1, "b": 0})
    prompt = proxy._prompt("a", tokenizer)
    assert prompt.rfind("岗位：") > prompt.rfind("已由昂贵标注模型判断的示例")
    assert prompt.rfind("当前候选人简历") > prompt.rfind("岗位：")
    scores = proxy.score_ids(["a", "b", "c"])
    assert scores[0] > 0.9 and scores[1] < 0.1 and scores[2] > 0.9
    assert model.calls == 2
    assert np.allclose(proxy.score_ids(["c", "b"]), scores[[2, 1]])
    assert model.calls == 2

    artifact = tmp_path / "proxy.json"
    proxy.save(artifact, threshold=0.4)
    loaded, threshold = QwenCausalProxy.load(
        artifact, tmp_path, tokenizer=tokenizer, model=model, cache_dir=tmp_path / "cache")
    assert loaded.demonstrations == [("a", 1), ("b", 0)] and threshold == 0.4


def test_agent_prompts_require_qwen_model_not_qwen_embeddings():
    assert "QwenCausalProxy" in PROXY_SKILL and "Qwen3-0.6B" in PROXY_SKILL
    assert "不得另训 LR" in PROXY_SKILL
