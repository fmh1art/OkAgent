import json

import duckdb
import pytest

from okagent.data import prepare, write_json
from okagent.evaluation import evaluate


def output(run, ids, calls=2):
    write_json(run / "workspace/output/candidate_ids.json", ids)
    write_json(run / "workspace/output/usage.json", {"llm_calls": calls})


def test_prepare_and_evaluate(run):
    workspace = run / "workspace"
    assert (workspace / "data.duckdb").is_symlink()
    assert not list(workspace.glob("*llm_pass*"))
    prompt = (workspace / "prompt.md").read_text()
    assert all(name in prompt for name in ("Partition", "Sample", "Label", "Proxy", "Deploy"))
    assert "最多 3 次" in prompt
    assert "禁止传 `device_map`" in prompt
    assert "唯一的 proxy model" in prompt
    assert "backend 必须是 `qwen_direct`" in prompt
    output(run, ["a", "b"])
    result = evaluate(run)
    assert [result[key] for key in ("tp", "fp", "fn", "tn")] == [1, 1, 2, 2]
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["precision"] == 0.5
    assert result["within_budget"] is True
    assert result["usage_source"] == "agent_reported"
    assert json.loads((run / "evaluation.json").read_text()) == result


@pytest.mark.parametrize("variant, marker", [
    ("cascade", "backend 必须是 `lr_qwen_cascade`"),
    ("paper_skill", "output/sampling_trace.json"),
    ("paper_skill_lr", "backend 必须是 `embedding_lr_paper_skill`"),
    ("combined", "同时执行 paper_skill sampling 与 LR→Qwen cascade"),
])
def test_prepare_proxy_variants(source, tmp_path, variant, marker):
    run = tmp_path / variant
    prepare(source, run, proxy_variant=variant)
    assert marker in (run / "workspace/prompt.md").read_text(encoding="utf-8")
    assert json.loads((run / "task.json").read_text())["proxy_variant"] == variant


@pytest.mark.parametrize("variant", ["paper_skill", "paper_skill_lr"])
def test_paper_skill_uses_minority_stratified_active_learning(source, tmp_path, variant):
    run = tmp_path / variant
    prepare(source, run, proxy_variant=variant)
    prompt = (run / "workspace/prompt.md").read_text(encoding="utf-8")
    assert "AL（主动学习）" in prompt
    assert "少数类 stratum" in prompt
    assert "cumulative_train.json" in prompt
    assert "禁止把多个路径拼成一个" in prompt
    assert "proxy_valid=false" in prompt


def test_paper_skill_lr_requires_paper_proxy_guards(source, tmp_path):
    run = tmp_path / "paper_skill_lr"
    prepare(source, run, proxy_variant="paper_skill_lr")
    prompt = (run / "workspace/prompt.md").read_text(encoding="utf-8")
    assert "WHERE segment='unstructured'" in prompt
    assert "1:1、1:3、1:5" in prompt
    assert "纯 proxy" in prompt
    assert "Adaptive Proxy Selection" in prompt


def test_unknown_proxy_variant_fails_before_creating_run(source, tmp_path):
    run = tmp_path / "unknown"
    with pytest.raises(ValueError, match="Unknown proxy_variant"):
        prepare(source, run, proxy_variant="unknown")


@pytest.mark.parametrize("ids", [["a", "a"], ["unknown"], [123], {}])
def test_invalid_ids(run, ids):
    output(run, ids)
    with pytest.raises(ValueError):
        evaluate(run)


def test_missing_labels_and_over_budget(run):
    task = json.loads((run / "task.json").read_text())
    with duckdb.connect(task["labels"]) as con:
        con.execute("DELETE FROM llm_pass WHERE candidate_id='a'")
    output(run, ["a"], calls=4)
    result = evaluate(run)
    assert result["evaluated_count"] == 5
    assert result["unknown_gold_count"] == 1
    assert result["recall"] == 0
    assert result["precision"] is None
    assert result["within_budget"] is False


def test_missing_job_data_and_existing_run(source, run, tmp_path):
    with pytest.raises(FileNotFoundError, match="job06"):
        prepare(source, tmp_path / "missing", job="job06")
    with pytest.raises(FileExistsError):
        prepare(source, run)
