"""Resolve job01 data artifacts without loading the full resume corpus."""
from __future__ import annotations

import json
from pathlib import Path


def resolve_job01(config_path: Path) -> tuple[Path, Path, dict, str]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("job") != "job01":
        raise ValueError("this experiment requires config.job='job01'")
    raw = (config_path.parent / config["raw_root"]).resolve()
    descriptions = list((raw / "job-description-20260722").glob("01_*.json"))
    if len(descriptions) != 1:
        raise ValueError("expected exactly one job01 description")
    job = json.loads(descriptions[0].read_text(encoding="utf-8"))
    embedding_root = raw / "job-candidate-embedding-20260722"
    data = embedding_root / "hiring_job01_full_segvec.db"
    labels = embedding_root / "hiring_job01_llm_pass.db"
    if not data.is_file() or not labels.is_file():
        raise FileNotFoundError("job01 embedding or historical-label database is missing")
    return data, labels, job, config["as_of"]
