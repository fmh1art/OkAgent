import json
import re
import shlex

import pytest

pytest.importorskip("minisweagent")
from minisweagent.models.test_models import DeterministicModel, make_output

from okagent.agent import make_model
from okagent.data import llm_config, write_json
from okagent.lo_ph_agent import LogicalOperators, run_lo_ph_agent
from test_agent import judgment, llm


def test_online_qwen_proxy_requires_labeled_validation(run):
    workspace = run / "workspace"
    write_json(workspace / "train.json", [{"candidate_id": "a", "label": 1}])
    operators = LogicalOperators(workspace, proxy_variant="paper_skill_qwen_online")
    with pytest.raises(ValueError, match="separately labeled validation"):
        operators.proxy({"train": "train.json"}, "Train online Qwen")


def test_online_qwen_proxy_uses_fixed_trainer_command(run, monkeypatch):
    import okagent.lo_ph_agent as module
    workspace = run / "workspace"
    operators = LogicalOperators(workspace, proxy_variant="paper_skill_qwen_online")
    (workspace / "operators/proxy-test").mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        target = workspace / "operators/proxy-test"
        write_json(target / "metadata.json", {"train_count": 5,
                   "validation_positive": 2, "proxy_valid": False})
        return None

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    operators._run_qwen_proxy({"train": "train.json", "train_round_1": "round.json",
                              "validation": "validation.json"},
                             "model_name=Qwen/Qwen3-0.6B", "operators/proxy-test",
                             workspace / "operators/proxy-test")
    script = (workspace / "operators/proxy-test/implementation.py").read_text()
    assert "Qwen/Qwen3-0.6B" in script
    assert "--train', 'train.json', '--train', 'round.json'" in script
    assert len(calls) == 1
    result = json.loads((workspace / "operators/proxy-test/result.json").read_text())
    assert result["artifacts"]["adapter"] == "operators/proxy-test/adapter"


def tool(name, arguments):
    return {"role": "assistant", "content": "planner-only-marker" if name != "bash" else "Implement this operator",
            "tool_calls": [{"id": "test-call", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def test_native_functions_delegate_and_share_workspace(run, llm, monkeypatch):
    monkeypatch.setenv("OKAGENT_CODE_MODEL", "openai/test")
    paths = {}

    def implement(body):
        task = body["messages"][1]["content"]
        request = json.loads(re.search(r"本次调用：(.*)\n", task)[1])
        directory = re.search(r"产物目录：(.*)\n", task)[1]
        name, inputs = request["operator"], request["inputs"]
        script = f'''import json, pickle, duckdb
from pathlib import Path
from okagent.semantic import SemanticOperator
directory = Path({directory!r})
inputs = {inputs!r}
with duckdb.connect('data.duckdb', read_only=True) as con:
    rows = con.execute("SELECT candidate_id,string_agg(text, ' ') FROM candidate_segments GROUP BY candidate_id ORDER BY candidate_id").fetchall()
texts = dict(rows)
'''
        if name == "sample":
            script += "value = list(texts)[:2]\nkey = 'table'\n"
        elif name == "label":
            script += "op = SemanticOperator('.')\nvalue = [{'candidate_id': cid, 'label': op.label(cid)} for cid in json.loads(Path(inputs['table']).read_text())]\nkey = 'table'\n"
        elif name == "proxy":
            script += '''from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train = json.loads(Path(inputs['train']).read_text())
model = make_pipeline(TfidfVectorizer(analyzer='char'), LogisticRegression(class_weight='balanced', random_state=42))
model.fit([texts[r['candidate_id']] for r in train], [r['label'] for r in train])
with (directory / 'model.pkl').open('wb') as f:
    pickle.dump(model, f)
key = 'model'
'''
        elif name == "deploy":
            script += '''with Path(inputs['model']).open('rb') as f:
    model = pickle.load(f)
selected = {cid for cid, score in zip(texts, model.predict_proba(list(texts.values()))[:, 1]) if score >= 0.1}
for row in json.loads(Path(inputs['labels']).read_text()):
    (selected.add if row['label'] else selected.discard)(row['candidate_id'])
value = sorted(selected)
key = 'candidate_ids'
'''
        script += '''artifact = directory / ('model.pkl' if key == 'model' else 'table.json')
if key != 'model':
    artifact.write_text(json.dumps(value))
(directory / 'result.json').write_text(json.dumps({'artifacts': {key: str(artifact)}, 'summary': 'Synthetic execution and file checks complete'}))
'''
        paths[name] = f"{directory}/" + ("model.pkl" if name == "proxy" else "table.json")
        command = f"cat > {shlex.quote(directory + '/implementation.py')} <<'PY'\n{script}\nPY\npython {shlex.quote(directory + '/implementation.py')}"
        return tool("bash", {"command": command})

    submit = (200, tool("bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}))
    llm[1].extend([
        (200, tool("sample", {"inputs": {"table": "data.duckdb"}, "instruction": "取 ID 排序前 2 人"})),
        (200, implement), submit,
        (200, lambda _: tool("label", {"inputs": {"table": paths["sample"]}, "instruction": "标注全部样本"})),
        (200, implement), judgment("a", 1), judgment("b", 0), submit,
        (200, lambda _: tool("proxy", {"inputs": {"train": paths["label"]}, "instruction": "字符 TF-IDF + LR，无独立验证，不宣称验证指标"})),
        (200, implement), submit,
        (200, lambda _: tool("deploy", {"inputs": {"table": "data.duckdb", "model": paths["proxy"], "labels": paths["label"]}, "instruction": "全库预测，阈值 0.1，覆盖已知标签"})),
        (200, implement), submit,
        (200, lambda _: tool("finish", {"candidate_ids": paths["deploy"], "summary": "Synthetic smoke test only"})),
    ])
    result = run_lo_ph_agent(run, physical_step_limit=3, step_limit=6)
    assert result["evaluation"]["llm_calls"] == 2
    assert result["evaluation"]["recall"] == 1
    assert [r["operator"] for r in result["operators"]] == ["sample", "label", "proxy", "deploy"]
    assert len(llm[0]) == 15
    for record in result["operators"]:
        trajectory = json.loads((run / "workspace" / record["trajectory"]).read_text())
        assert trajectory["info"]["config"]["environment"]["cwd"] == str((run / "workspace").resolve())
        assert "实现正确性" in trajectory["messages"][0]["content"]
        assert "planner-only-marker" not in json.dumps(trajectory)
        assert trajectory["info"]["model_stats"]["api_calls"] == 2
    assert "计划的正确性" in json.loads((run / "logical.trajectory.json").read_text())["messages"][0]["content"]


def test_failed_physical_call_returns_to_planner(run):
    factory_calls = []

    def physical_model():
        factory_calls.append(1)
        return DeterministicModel(outputs=[make_output("Inspect", [{"command": "pwd"}])])

    planner = DeterministicModel(outputs=[make_output("Try sample", [dict(
        operator="sample", arguments={"inputs": {"table": "data.duckdb"}, "instruction": "随机取 2 人"})])])
    with pytest.raises(RuntimeError, match="Logical agent stopped: LimitsExceeded"):
        run_lo_ph_agent(run, planner_model=planner, physical_model_factory=physical_model,
                        step_limit=1, physical_step_limit=1)
    trajectory = json.loads((run / "logical.trajectory.json").read_text())
    assert factory_calls == [1] and not trajectory["info"]["logical_operators"]
    assert "sample: LimitsExceeded" in json.dumps(trajectory)
    assert not (run / "evaluation.json").exists()


def test_logical_function_rejects_missing_input_before_spawning(run):
    def forbidden():
        pytest.fail("Invalid input must not start an agent")

    ops = LogicalOperators(run / "workspace", model_factory=forbidden)
    with pytest.raises(FileNotFoundError):
        ops.sample({"table": "missing.json"}, "抽样")
    with pytest.raises(ValueError, match="workspace-relative"):
        ops.partition({"table": "../task.json"}, "分区")


def test_local_configuration_and_secret_not_serialized(tmp_path, monkeypatch):
    config = tmp_path / "llm.json"
    write_json(config, {"openai_base_url": "http://localhost/v1", "llm_name": "example-endpoint", "key": "config-test-key"})
    monkeypatch.setenv("OKAGENT_LLM_CONFIG", str(config))
    for name in ("OKAGENT_CODE_MODEL", "OPENAI_API_KEY", "OKAGENT_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    assert llm_config()["model"] == "example-endpoint"
    model = make_model()
    assert model.config.model_name == "openai/example-endpoint"
    assert model.config.model_kwargs["api_base"] == "http://localhost/v1"
    assert "config-test-key" not in json.dumps(model.serialize())


def test_label_cannot_submit_partial_or_invented_results(run, monkeypatch):
    import okagent.lo_ph_agent as module
    workspace = run / 'workspace'
    write_json(workspace / 'input.json', ['a', 'b'])

    class PartialAgent:
        def __init__(self, model, env, **kwargs):
            self.directory = kwargs['output_path'].parent
        def run(self, task):
            (self.directory / 'implementation.py').write_text('# Deliberately incomplete test artifact')
            write_json(self.directory / 'table.json', [{'candidate_id': 'a', 'label': 0}])
            write_json(self.directory / 'result.json', {'artifacts': {
                'table': str((self.directory / 'table.json').relative_to(workspace))}, 'summary': 'Incomplete'})
            return {'exit_status': 'Submitted'}

    monkeypatch.setattr(module, 'DefaultAgent', PartialAgent)
    ops = LogicalOperators(workspace, model_factory=lambda: None)
    with pytest.raises(ValueError, match='cover every input ID'):
        ops.label({'table': 'input.json'}, 'Label both IDs')
    assert not ops.completed
