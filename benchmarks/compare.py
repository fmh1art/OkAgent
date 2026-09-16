"""Small, versioned experiment runner; invoke these functions from Python."""
import json
import hashlib
import importlib.metadata
import os
import shutil
import sqlite3
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
    manifest = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                    python=sys.version, packages={name: importlib.metadata.version(name) for name in
                    ("mini-swe-agent", "openai", "litellm", "duckdb", "scikit-learn", "numpy", "scipy")},
                    files={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((directory / "runtime").rglob("*.py"))})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory


def run_case(round_name, method, job="job01", *, resume=False, previous_run=None,
             validation_ids=None, validation_fixed=True, validation_sample_ids=None):
    directory = ROOT / "results/comparison" / round_name
    sys.path.insert(0, str(directory / "runtime"))
    os.environ["OKAGENT_LLM_CONFIG"] = str(ROOT / "_config/llm.json")
    from okagent.data import prepare, write_json
    from okagent.evaluation import evaluate

    run = directory / f"{method}-{job}"
    previous = None
    if resume:
        if method != "hydra":
            raise ValueError("Resume is only needed for the deterministic Hydra runner")
        previous = json.loads((run / "experiment.json").read_text())
        (run / "previous_attempt.json").write_text(json.dumps(previous, indent=2))
        shutil.copyfile(run / "error.txt", run / "previous_error.txt")
    else:
        prepare(directory / "hiring.json", run, job=job)
    initial_calls = 0
    if previous_run is not None:
        old = Path(previous_run) / "workspace/output/semantic.sqlite"
        with sqlite3.connect(old) as source, sqlite3.connect(run / "workspace/output/semantic.sqlite") as destination:
            source.backup(destination)
            initial_calls = source.execute("SELECT count(*) FROM queries").fetchone()[0]
            labels = dict(source.execute("SELECT candidate_id,label FROM queries WHERE label IS NOT NULL"))
        validation = set(validation_ids)
        workspace = run / "workspace"
        write_json(workspace / "continuation_validation_pool.json", sorted(validation))
        for name, keep in (("train", lambda cid: cid not in validation), ("validation", lambda cid: cid in validation)):
            write_json(workspace / f"continuation_{name}.json", [dict(candidate_id=cid, label=labels[cid])
                                                               for cid in sorted(labels) if keep(cid)])
        continuation = dict(initial_calls=initial_calls, train="continuation_train.json",
                            validation_labels="continuation_validation.json",
                            validation_pool="continuation_validation_pool.json", validation_fixed=validation_fixed,
                            instruction="同一2000次总预算内续跑，所有旧尝试已计入账本，不能重置。复用上述训练标签；验证池禁止用于训练。"
                            "validation_fixed=true时补齐该固定验证样本，否则在此独立池随机抽约400人并冻结为验证样本。"
                            "训练可继续主动采样全库中不在验证池且未标注的ID。不要重新划分。批量读取并缓存特征，"
                            "训练、主动采样和部署必须使用相同的特征处理。统一用workers=8，标注器已负责全局限速。")
        if validation_sample_ids is not None:
            sample = set(validation_sample_ids)
            if not sample.issubset(validation):
                raise ValueError("Fixed validation sample must belong to the original validation pool")
            write_json(workspace / "continuation_validation_sample.json", sorted(sample))
            continuation["validation_sample"] = "continuation_validation_sample.json"
            continuation["instruction"] += " validation_sample 已给定时必须复用该固定样本，不能重新抽取；整个validation_pool仍禁止训练。"
        write_json(workspace / "continuation.json", continuation)
        with (workspace / "prompt.md").open("a") as prompt:
            prompt.write("\n## 续跑要求（优先于默认重新划分步骤）\n先读取 continuation.json，严格复用其中的验证划分、已有标签和预算。\n")
    metadata = dict(round=round_name, method=method, job=job,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="running")
    metadata.update(initial_calls=initial_calls, previous_run=str(previous_run) if previous_run else None,
                    command_timeout=7200)
    write_json(run / "experiment.json", metadata)
    started = time.monotonic()
    try:
        if method == "baseline":
            from okagent.agent import run_agent
            run_agent(run, command_timeout=7200)
        elif method == "lo_ph":
            from okagent.lo_ph_agent import run_lo_ph_agent
            run_lo_ph_agent(run, command_timeout=7200)
        elif method in ("hydra", "hydra_replay"):
            from okagent.semantic import SemanticOperator
            from other_methods.hydra import Config, run_hiring
            label = None
            if method == "hydra":
                op = SemanticOperator(run / "workspace")
                label = lambda ids: [op.label(cid) for cid in ids]
            run_hiring(run, config=Config(target_recall=0.9, label_workers=8 if label else 1), label=label)
            evaluate(run)
        else:
            raise ValueError(method)
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc())
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        if previous:
            metadata["previous_attempt"] = previous
            metadata["elapsed_seconds"] += previous["elapsed_seconds"]
            metadata["resumed_from_cache"] = True
        write_json(run / "experiment.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata
