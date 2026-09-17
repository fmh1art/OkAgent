"""Prepare the resume data, labeling settings and code-agent task."""
import json
import os
import shutil
from pathlib import Path

import duckdb

from .prompts import build_prompt, proxy_variant_instruction


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def llm_config():
    path = Path(os.environ.get("OKAGENT_LLM_CONFIG", Path(__file__).resolve().parents[2] / "_config/llm.json"))
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return dict(model=config.get("llm_name"),
                base_url=os.environ.get("OKAGENT_API_BASE") or config.get("openai_base_url"),
                api_key=os.environ.get("OPENAI_API_KEY") or config.get("key"),
                label_kwargs=config.get("label_kwargs", {}),
                label_interval=float(config.get("label_interval", 0)))


def prepare(config_path, run_dir, job=None, *, proxy_variant="control"):
    proxy_variant_instruction(proxy_variant)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = (config_path.parent / config["raw_root"]).resolve()
    job = f"job{int((job or config['job']).removeprefix('job')):02d}"
    descriptions = list((raw / "job-description-20260722").glob(f"{job[3:]}_*.json"))
    if len(descriptions) != 1:
        raise ValueError(f"Expected one job description for {job}")
    databases = raw / "job-candidate-embedding-20260722"
    data = databases / f"hiring_{job}_full_segvec.db"
    labels = databases / f"hiring_{job}_llm_pass.db"
    for path in (data, labels, raw / "llm_prompt.txt"):
        if not path.is_file():
            raise FileNotFoundError(f"{job}: missing {path}")
    if type(config["max_calls"]) is not int or config["max_calls"] < 0:
        raise ValueError("max_calls must be a nonnegative integer")
    description = json.loads(descriptions[0].read_text(encoding="utf-8"), strict=False)
    with duckdb.connect(str(data), read_only=True) as con:
        count = con.execute("SELECT count(DISTINCT candidate_id) FROM candidate_segments").fetchone()[0]
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    (workspace / "output").mkdir(parents=True)
    # Link only the resume database. Historical labels stay outside the workspace.
    (workspace / "data.duckdb").symlink_to(data)
    write_json(workspace / "job.json", description)
    write_json(workspace / "settings.json", dict(as_of=config["as_of"], max_calls=config["max_calls"]))
    shutil.copyfile(raw / "llm_prompt.txt", workspace / "label_prompt.txt")
    write_json(run_dir / "task.json", dict(job=job, data=str(data), labels=str(labels),
                                          max_calls=config["max_calls"], proxy_variant=proxy_variant))
    prompt = workspace / "prompt.md"
    prompt.write_text(build_prompt(workspace, description, count, config, proxy_variant), encoding="utf-8")
    return prompt
