"""Unbounded, one-operator-at-a-time ReAct with durable per-round traces."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def scrub(value):
    if isinstance(value, str):
        for name in ("ARK_API_KEY", "OPENAI_API_KEY"):
            secret = os.getenv(name)
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(scrub(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(scrub(value), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def parse_action(text):
    match = re.fullmatch(r"\s*<(operator|done)>(.*?)</\1>\s*", text, re.S)
    if not match:
        raise ValueError("Return exactly one <operator>{JSON}</operator> or <done>...</done>")
    kind, body = match.groups()
    if kind == "done":
        if not body.strip():
            raise ValueError("done summary must not be empty")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {"summary": body.strip()}
        return kind, result if isinstance(result, dict) else {"summary": str(result)}
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        # Some models emit Python True/False/None; literal_eval never executes code.
        try:
            raw = ast.literal_eval(body)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Invalid single operator object: " + str(exc)) from None
    if not isinstance(raw, dict):
        raise ValueError("operator must be a JSON object, not a list of operators")
    name = raw.get("operator", raw.get("name"))
    params = raw.get("params", raw.get("parameters", {}))
    if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
        raise ValueError("operator needs a nonempty name and an object params")
    return kind, {**raw, "operator": name.strip(), "params": params}


def execute_generated(code_xml, payload, directory, timeout=7200):
    from operator_agent import parse_code, validate_generated_code
    code = parse_code(code_xml)
    # Persist even rejected code so debugging traces are complete.
    (directory / "code.py").write_text(scrub(code), encoding="utf-8")
    save(directory / "payload.json", payload)
    validate_generated_code(code)
    bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name='__main__')"
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PATH": os.defpath,
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
    # Real files retain partial output after timeout, termination or a process crash.
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        result = subprocess.run(
            [sys.executable, "-I", "-c", bootstrap, str(Path(__file__).resolve().parent),
             str(directory / "code.py"), str(directory / "payload.json")],
            cwd=directory, env=env, stdout=out, stderr=err, timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError(f"Code exit={result.returncode}: " + scrub(stderr_path.read_text(encoding="utf-8")[-4000:]))
    observation = json.loads(stdout_path.read_text(encoding="utf-8"))
    if not isinstance(observation, dict):
        raise ValueError("stdout must be a single JSON observation object")
    return scrub(observation)


def run_react(planner, coder, context, run_dir, executor=execute_generated,
              observe=None, finalize=None, timeout=7200, verbose=False):
    """No round limit or fixed operator sequence. Only Planner's done ends a run."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rounds = run_dir / "rounds"
    rounds.mkdir(exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    history = []
    existing = sorted(rounds.glob("[0-9]*"))
    for directory in existing:
        path = directory / "round.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] not in {"ok", "error", "done", "interrupted", "api_error"}:
            record.update(status="interrupted", observation={"status": "error", "error": "Previous process stopped during this round; inspect its artifacts before retrying"})
            save(path, record)
            append(run_dir / "trajectory.jsonl", record)
        history.append(record)
    next_round = max([int(p.name) for p in existing] + [0]) + 1

    while True:
        number = next_round
        next_round += 1
        directory = rounds / f"{number:06d}"
        directory.mkdir()
        started = time.monotonic()
        record = {"round": number, "started_at": datetime.now(timezone.utc).isoformat(),
                  "status": "planning", "operator": None, "round_dir": str(directory)}

        def update(**fields):
            record.update(fields)
            save(directory / "round.json", record)
            append(run_dir / "events.jsonl", {"round": number, "status": record["status"],
                                             "time": datetime.now(timezone.utc).isoformat()})

        model_context = {**context, "round": number, "history": [
            {k: r.get(k) for k in ("round", "operator", "status", "observation", "round_dir")}
            for r in history]}
        update()
        save(directory / "planner_input.json", model_context)
        print(f"[Round {number}] Planner", flush=True)
        stage = "planner_api"
        final_result = None
        try:
            raw = planner.next_operator(model_context)
            (directory / "planner_response.txt").write_text(scrub(raw), encoding="utf-8")
            record["planner_response"] = scrub(raw)
            stage = "parse"
            kind, action = parse_action(raw)
            update(operator=action if kind == "operator" else {"operator": "<done>", **action}, status="planned")
            save(directory / "operator.json", record["operator"])
            if kind == "done":
                stage = "finalize"
                final_result = finalize(action, history) if finalize else {"summary": action.get("summary", "done")}
                update(status="done", observation={"status": "ok", "summary": action.get("summary", "done")})
            else:
                print(f"[Round {number}] {action['operator']} {json.dumps(action['params'], ensure_ascii=False)}", flush=True)
                canonical = "<operator>" + json.dumps(action, ensure_ascii=False) + "</operator>"
                stage = "coder_api"
                update(status="coding")
                code = coder.write_code(canonical, model_context)
                (directory / "coder_response.txt").write_text(scrub(code), encoding="utf-8")
                stage = "execute"
                update(status="executing")
                payload = {"root": context.get("root"), "run_dir": str(run_dir),
                           "round_dir": str(directory), "operator": action, "context": model_context}
                obs = executor(code, payload, directory, timeout)
                if observe and obs.get("status") not in {"error", "failed"}:
                    obs = observe(obs, directory)
                update(status="error" if obs.get("status") in {"error", "failed"} else "ok", observation=obs)
        except BaseException as exc:
            interrupted = not isinstance(exc, Exception)
            status = "interrupted" if interrupted else ("api_error" if stage.endswith("_api") else "error")
            update(status=status, observation={"status": "error", "stage": stage,
                                               "error": scrub(str(exc) or type(exc).__name__)})
            if interrupted or stage.endswith("_api"):
                record["elapsed_seconds"] = time.monotonic() - started
                save(directory / "round.json", record)
                append(run_dir / "trajectory.jsonl", record)
                raise
        record["elapsed_seconds"] = time.monotonic() - started
        save(directory / "round.json", record)
        append(run_dir / "trajectory.jsonl", record)
        history.append(record.copy())
        save(run_dir / "checkpoint.json", {"last_round": number, "history": history})
        print(f"[Round {number}] {record['status']}: {json.dumps(record.get('observation', {}), ensure_ascii=False)}", flush=True)
        if final_result is not None:
            result = {**final_result, "status": "complete", "rounds": number,
                      "run_dir": str(run_dir), "done_summary": action.get("summary", "done")}
            save(result_path, result)
            return result
