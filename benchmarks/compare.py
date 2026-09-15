"""Small, versioned experiment runner; invoke these functions from Python."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot(round_name):
    directory = ROOT / "results/comparison" / round_name
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src/okagent", directory / "runtime/okagent", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "other_methods", directory / "runtime/other_methods",
                    ignore=shutil.ignore_patterns("__pycache__", "*.zip"))
    config = json.loads((ROOT / "_config/hiring.json").read_text())
    config["raw_root"] = str((ROOT / "_config" / config["raw_root"]).resolve())
    (directory / "hiring.json").write_text(json.dumps(config))
    llm = json.loads((ROOT / "_config/llm.json").read_text())
    (directory / "label_settings.json").write_text(json.dumps({k: v for k, v in llm.items() if k != "key"}, indent=2))
    return directory


def run_case(round_name, method, job="job01"):
    directory = ROOT / "results/comparison" / round_name
    sys.path.insert(0, str(directory / "runtime"))
    os.environ["OKAGENT_LLM_CONFIG"] = str(ROOT / "_config/llm.json")
    from okagent.data import prepare, write_json
    from okagent.evaluation import evaluate

    run = directory / f"{method}-{job}"
    prepare(directory / "hiring.json", run, job=job)
    metadata = dict(round=round_name, method=method, job=job,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="running")
    write_json(run / "experiment.json", metadata)
    started = time.monotonic()
    try:
        if method == "baseline":
            from okagent.agent import run_agent
            run_agent(run)
        elif method == "lo_ph":
            from okagent.lo_ph_agent import run_lo_ph_agent
            run_lo_ph_agent(run)
        elif method == "hydra":
            from okagent.semantic import SemanticOperator
            from other_methods.hydra import Config, run_hiring
            op = SemanticOperator(run / "workspace")
            run_hiring(run, config=Config(target_recall=0.9, label_workers=8), label=lambda ids: [op.label(cid) for cid in ids])
            evaluate(run)
        else:
            raise ValueError(method)
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc())
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        write_json(run / "experiment.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata
