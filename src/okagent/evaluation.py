"""Compare returned candidate IDs with the job's historical LLM labels."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import duckdb

from .data import write_json


def evaluate(run_dir):
    run_dir = Path(run_dir)
    task = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))
    output = run_dir / "workspace/output"
    selected = json.loads((output / "candidate_ids.json").read_text(encoding="utf-8"))
    if not isinstance(selected, list) or any(not isinstance(cid, str) for cid in selected):
        raise ValueError("candidate_ids.json must be a JSON array of string IDs")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate candidate IDs")
    selected = set(selected)
    with duckdb.connect(task["data"], read_only=True) as con:
        universe = {r[0] for r in con.execute("SELECT DISTINCT candidate_id FROM candidate_segments").fetchall()}
    if not selected.issubset(universe):
        raise ValueError("Unknown candidate IDs in output")
    truth = {}
    with duckdb.connect(task["labels"], read_only=True) as con:
        for cid, label in con.execute("SELECT candidate_id,llm_pass FROM llm_pass").fetchall():
            if cid not in universe:
                continue
            if cid in truth or (label is not None and label not in (0, 1)):
                raise ValueError("Duplicate IDs or invalid values in evaluation labels")
            truth[cid] = label
    positive = {cid for cid, label in truth.items() if label == 1}
    negative = {cid for cid, label in truth.items() if label == 0}
    tp, fp = len(selected & positive), len(selected & negative)
    fn, tn = len(positive - selected), len(negative - selected)
    state = output / "semantic.sqlite"
    if state.exists():
        with closing(sqlite3.connect(f"{state.resolve().as_uri()}?mode=ro", uri=True)) as db:
            calls = db.execute("SELECT count(*) FROM queries").fetchone()[0]
        usage_source = "semantic_operator"
    else:
        calls = json.loads((output / "usage.json").read_text(encoding="utf-8"))["llm_calls"]
        usage_source = "agent_reported"
    if type(calls) is not int or calls < 0:
        raise ValueError("llm_calls must be a nonnegative integer")
    report = dict(job=task["job"], candidate_count=len(universe), selected_count=len(selected),
                  evaluated_count=len(positive | negative), unknown_gold_count=len(universe - positive - negative),
                  tp=tp, fp=fp, fn=fn, tn=tn,
                  recall=tp / (tp + fn) if tp + fn else None,
                  precision=tp / (tp + fp) if tp + fp else None,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  llm_calls=calls, max_calls=task["max_calls"], within_budget=calls <= task["max_calls"],
                  usage_source=usage_source)
    write_json(run_dir / "evaluation.json", report)
    return report
