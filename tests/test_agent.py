"""Exercise the real HTTP client, subprocesses and mini-swe loop with local LLM responses."""
import json
import re
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

pytest.importorskip("minisweagent")
from minisweagent.models.test_models import DeterministicModel, make_output

from okagent.agent import run_agent
from okagent.data import write_json
from okagent.evaluation import evaluate
from okagent.semantic import BudgetExceeded, SemanticOperator


@pytest.fixture
def llm(monkeypatch):
    requests, replies = [], deque()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            status, content = replies.popleft()
            if callable(content):
                content = content(body)
            response = ({"error": {"message": "test failure", "type": "server_error"}} if status != 200 else
                        {"id": "test", "object": "chat.completion", "created": 0, "model": "test",
                         "choices": [{"index": 0, "message": content if isinstance(content, dict) else
                                      {"role": "assistant", "content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OKAGENT_LABEL_MODEL", "openai/test")
    monkeypatch.setenv("OKAGENT_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield requests, replies
    server.shutdown()
    thread.join()
    server.server_close()


def judgment(cid, label):
    return 200, json.dumps({"candidate_id": cid, "is_match": {"result": bool(label)}})


def test_row_queries_cache_and_evaluation(run, llm):
    requests, replies = llm
    replies.extend([judgment("a", 1), judgment("b", 0)])
    op = SemanticOperator(run / "workspace")
    assert [op.label(cid) for cid in ["a", "b", "a"]] == [1, 0, 1]
    process = subprocess.run([sys.executable, "-c", "from okagent.semantic import SemanticOperator; "
                              "assert SemanticOperator('.').label('b') == 0"],
                             cwd=run / "workspace", capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert op.usage() == {"llm_calls": 2, "max_calls": 3, "remaining": 1}
    assert len(requests) == 2
    for request, cid in zip(requests, "ab"):
        prompt = request["messages"][-1]["content"]
        assert f"测试文本 {cid}" in prompt and "2026-07-22" in prompt and "有相关经历" in prompt
        assert "{{" not in prompt and "HIDDEN_METADATA" not in prompt
        assert re.findall(r"ID: ([a-f])", prompt) == [cid]
    write_json(run / "workspace/output/candidate_ids.json", ["a", "c"])
    write_json(run / "workspace/output/usage.json", {"llm_calls": 0})
    report = evaluate(run)
    assert report["llm_calls"] == 2 and report["usage_source"] == "semantic_operator"


@pytest.mark.parametrize("bad_response", ["not JSON", '{"candidate_id":"a","is_match":{"result":"false"}}',
                                       '{"candidate_id":"b","is_match":{"result":true}}'])
def test_failed_requests_charge_and_budget_persists(run, llm, bad_response):
    requests, replies = llm
    replies.extend([(500, ""), (200, bad_response), judgment("a", 1)])
    op = SemanticOperator(run / "workspace")
    with pytest.raises(Exception, match="test failure"):
        op.label("a")
    assert len(requests) == 1  # HTTP errors are surfaced; only malformed judgments retry.
    assert op.label("a") == 1
    with pytest.raises(BudgetExceeded):
        SemanticOperator(run / "workspace").label("b")
    assert op.label("a") == 1  # Cached successes remain available after exhaustion.
    assert len(requests) == 3 and op.usage()["remaining"] == 0
    assert json.loads((run / "workspace/output/usage.json").read_text())["llm_calls"] == 3


def test_invalid_input_never_calls_llm(run, llm):
    op = SemanticOperator(run / "workspace")
    for cid in ["unknown", 123]:
        with pytest.raises(ValueError):
            op.label(cid)
    assert not llm[0] and op.usage()["llm_calls"] == 0


def test_concurrent_calls_share_cache_and_budget(run, llm):
    llm[1].append(judgment("a", 1))
    operators = [SemanticOperator(run / "workspace") for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(lambda op: op.label("a"), operators)) == [1, 1, 1]
    assert len(llm[0]) == 1 and operators[0].usage()["llm_calls"] == 1


def test_zero_budget_and_missing_model_do_not_send(run, llm, monkeypatch):
    monkeypatch.delenv("OKAGENT_LABEL_MODEL")
    with pytest.raises(ValueError, match="OKAGENT_LABEL_MODEL"):
        SemanticOperator(run / "workspace").label("a")
    write_json(run / "workspace/settings.json", {"as_of": "2026-07-22", "max_calls": 0})
    with pytest.raises(BudgetExceeded):
        SemanticOperator(run / "workspace").label("a")
    assert not llm[0] and SemanticOperator(run / "workspace").usage()["llm_calls"] == 0


PIPELINE = '''import json, pickle
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from okagent.semantic import SemanticOperator

with duckdb.connect('data.duckdb', read_only=True) as con:
    rows = con.execute("SELECT candidate_id,string_agg(text, ' ') FROM candidate_segments GROUP BY candidate_id ORDER BY candidate_id").fetchall()
ids, texts = zip(*rows)
op = SemanticOperator('.')
y = [op.label(cid) for cid in ids[:3]]
model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1, 2)),
                      LogisticRegression(class_weight='balanced', random_state=42))
model.fit(texts[:2], y[:2])
threshold = float(model.predict_proba(texts[2:3])[0, 1]) * 0.9
predictions = model.predict_proba(texts)[:, 1] >= threshold
predictions[:3] = y
with open('output/proxy.pkl', 'wb') as f:
    pickle.dump({'model': model, 'threshold': threshold}, f)
with open('output/candidate_ids.json', 'w') as f:
    json.dump([cid for cid, selected in zip(ids, predictions) if selected], f)
with open('output/report.md', 'w') as f:
    f.write('Synthetic smoke test: 2 training samples, 1 held-out positive; not a quality estimate.')
'''


def test_code_agent_executes_training_and_evaluates(run, llm, monkeypatch):
    monkeypatch.setenv("OKAGENT_CODE_MODEL", "openai/test")
    commands = ["cat > pipeline.py <<'PY'\n" + PIPELINE + "\nPY", "python pipeline.py",
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    actions = [(200, {"role": "assistant", "content": "Run the next step", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
    ]}) for i, cmd in enumerate(commands)]
    llm[1].extend(actions[:2] + [judgment("a", 1), judgment("b", 0), judgment("c", 1)] + actions[2:])
    result = run_agent(run, step_limit=4)
    assert result["agent"]["exit_status"] == "Submitted"
    assert result["evaluation"]["llm_calls"] == 3 and result["evaluation"]["within_budget"]
    assert (run / "workspace/output/proxy.pkl").is_file()
    trajectory = json.loads((run / "agent.trajectory.json").read_text())
    assert trajectory["info"]["model_stats"]["api_calls"] == 3
    assert len(llm[0]) == 6 and sum("tools" in request for request in llm[0]) == 3
    assert any("<returncode>0</returncode>" in str(m) for m in trajectory["messages"])
    assert "local-test-key" not in (run / "agent.trajectory.json").read_text()


def test_step_limit_keeps_trajectory_and_does_not_evaluate(run):
    model = DeterministicModel(outputs=[make_output("Inspect", [{"command": "pwd"}])])
    with pytest.raises(RuntimeError, match="LimitsExceeded"):
        run_agent(run, model=model, step_limit=1)
    assert (run / "agent.trajectory.json").is_file()
    assert not (run / "evaluation.json").exists()


def test_submission_without_training_artifacts_is_incomplete(run):
    write_json(run / "workspace/output/candidate_ids.json", [])
    model = DeterministicModel(outputs=[make_output("Done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}])])
    with pytest.raises(FileNotFoundError, match="pipeline.py"):
        run_agent(run, model=model)
    assert not (run / "evaluation.json").exists()


def test_independent_queries_overlap_and_keep_budget(run, llm):
    from threading import Barrier
    gate = Barrier(3)

    def response(body):
        cid = re.search(r'ID: ([a-f])', body['messages'][-1]['content'])[1]
        gate.wait(timeout=10)  # Would deadlock if the network request still held the global lock.
        return judgment(cid, cid == 'a')[1]

    llm[1].extend([(200, response)] * 3)
    op = SemanticOperator(run / 'workspace')
    assert op.label_many(['a', 'b', 'c'], workers=3) == [1, 0, 0]
    assert op.cached_labels() == {'a': 1, 'b': 0, 'c': 0}
    assert op.get_label('d') is None
    assert op.usage()['llm_calls'] == len(llm[0]) == 3
    with pytest.raises(BudgetExceeded):
        op.label_many(['d', 'e'], workers=2)
    assert len(llm[0]) == 3


def test_parallel_reservations_cannot_overrun_budget(run, llm):
    def response(body):
        cid = re.search(r'ID: ([a-f])', body['messages'][-1]['content'])[1]
        return judgment(cid, 0)[1]

    llm[1].extend([(200, response)] * 3)
    op = SemanticOperator(run / 'workspace')
    with pytest.raises(BudgetExceeded):
        op.label_many(list('abcdef'), workers=6)
    assert op.usage()['llm_calls'] == len(llm[0]) == 3


def test_model_format_tolerance_and_failed_token_record(run, llm):
    import sqlite3
    llm[1].extend([
        (200, '```json\n{"candidate_id":"a","is_match":{"result":true},"reason":"line one\nline two"}\n```'),
        (200, '{bad JSON'),
    ])
    op = SemanticOperator(run / 'workspace')
    assert op.label('a') == 1
    with pytest.raises(ValueError):
        op.label('b')
    with sqlite3.connect(op.state) as con:
        label, response = con.execute("SELECT label,response FROM queries WHERE candidate_id='b'").fetchone()
    assert label is None and json.loads(response)['_usage']['total_tokens'] == 2
    assert op.usage()['llm_calls'] == 2


def test_optional_endpoint_pacing_applies_to_parallel_requests(run, llm, monkeypatch, tmp_path):
    import time
    config = tmp_path / 'paced.json'
    write_json(config, {'label_interval': 0.12})
    monkeypatch.setenv('OKAGENT_LLM_CONFIG', str(config))
    arrivals = []
    def response(body):
        arrivals.append(time.monotonic())
        cid = re.search(r'ID: ([a-f])', body['messages'][-1]['content'])[1]
        return judgment(cid, 0)[1]
    llm[1].extend([(200, response)] * 3)
    assert SemanticOperator(run / 'workspace').label_many(list('abc'), workers=3) == [0, 0, 0]
    assert max(arrivals) - min(arrivals) >= 0.18


def test_large_tool_output_keeps_error_and_tail_without_filling_context(llm, monkeypatch):
    from okagent.agent import make_model
    monkeypatch.setenv('OKAGENT_CODE_MODEL', 'openai/test')
    model = make_model()
    raw = 'START' + 'x' * 100000 + 'END'
    messages = model.format_observation_messages(
        {'extra': {'actions': [{'command': 'cat large.json', 'tool_call_id': 'call_test'}]}},
        [{'output': raw, 'returncode': 1, 'exception_info': 'failed'}])
    observation = messages[0]
    assert len(observation['content']) < 21000
    assert all(part in observation['content'] for part in ['START', 'END', 'truncated', 'failed', '<returncode>1'])
    assert observation['extra']['raw_output'] == raw
