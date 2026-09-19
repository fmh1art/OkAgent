"""Full-population paper-skill experiment with a genuinely fine-tuned Qwen proxy.

Use an already prepared hiring run with a fixed continuation validation split.
The teacher budget and successful labels remain in SemanticOperator's SQLite ledger.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import duckdb

from okagent.data import write_json
from okagent.evaluation import evaluate
from okagent.qwen_online_trainer import DEFAULT_MODEL
from okagent.semantic import SemanticOperator


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _trainer(workspace, command, *args, log):
    invocation = [sys.executable, "-m", "okagent.qwen_online_trainer", command,
                  "--workspace", str(workspace), *map(str, args)]
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(invocation, cwd=workspace, stdout=output, stderr=subprocess.STDOUT, check=True)


def minority_sample(scores, threshold, count):
    if len({row["candidate_id"] for row in scores}) != len(scores):
        raise ValueError("duplicate candidate IDs in Qwen scores")
    ordered = sorted(scores, key=lambda row: (-row["score"], row["candidate_id"]))
    stratum = [row["candidate_id"] for row in ordered if row["score"] >= threshold]
    if len(stratum) < count:
        raise ValueError(f"predicted minority stratum has only {len(stratum)} IDs; need {count}")
    return stratum[:count], len(stratum)


def _rho(rows):
    positive = sum(row["label"] for row in rows)
    negative = len(rows) - positive
    return max(positive, negative) / min(positive, negative) if min(positive, negative) else None


def run(run_dir, model_name):
    run_dir = Path(run_dir).resolve()
    workspace = run_dir / "workspace"
    output = workspace / "output/qwen_online_paper"
    output.mkdir(parents=True, exist_ok=True)
    semantic = SemanticOperator(workspace)
    validation_pool = set(_read(workspace / "continuation_validation_pool.json"))
    validation_sample = sorted(set(_read(workspace / "continuation_validation_sample.json")))
    if not set(validation_sample).issubset(validation_pool):
        raise ValueError("validation sample must be contained in the fixed validation pool")
    with duckdb.connect(str(workspace / "data.duckdb"), read_only=True) as con:
        universe = sorted(row[0] for row in con.execute(
            "SELECT DISTINCT candidate_id FROM candidate_segments ORDER BY candidate_id").fetchall())
    cached = semantic.cached_labels()
    if any(cid not in cached for cid in validation_sample):
        raise ValueError("fixed validation sample must already be fully labeled")
    validation = [{"candidate_id": cid, "label": cached[cid]} for cid in validation_sample]
    cold_start_path = output / "train_round_0.json"
    if cold_start_path.exists():
        train = _read(cold_start_path)
        if any(cached.get(row["candidate_id"]) != row["label"] for row in train):
            raise ValueError("cached cold-start labels changed")
    else:
        train = [{"candidate_id": cid, "label": cached[cid]}
                 for cid in sorted(cached) if cid not in validation_pool]
    if len({row["label"] for row in train}) < 2:
        raise ValueError("random cold-start labels need both classes before Qwen training")
    write_json(output / "validation.json", validation)
    write_json(cold_start_path, train)
    trace_path = output / "sampling_trace.json"
    if not trace_path.exists():
        write_json(trace_path, [{"strategy": "random_cold_start",
            "sample_count": len(train), "positive": sum(row["label"] for row in train),
            "validation_count": len(validation), "validation_positive": sum(row["label"] for row in validation)}])
    first = output / "round_0"
    if not (first / "metadata.json").exists():
        _trainer(workspace, "fit", "--model-name", model_name,
                 "--train", output / "train_round_0.json",
                 "--validation", output / "validation.json", "--output-dir", first,
                 "--device", "cuda", "--seed", 42, log=output / "round_0.log")
    first_meta = _read(first / "metadata.json")
    if first_meta["model_name"] != model_name or not first_meta["proxy_valid"]:
        raise ValueError("initial Qwen proxy is invalid or uses the wrong model")
    pool = [cid for cid in universe if cid not in validation_pool and cid not in cached]
    write_json(output / "unlabeled_pool.json", pool)
    pool_scores_path = output / "round_0_pool_scores.json"
    if not pool_scores_path.exists():
        _trainer(workspace, "score", "--artifact", first / "proxy.pkl",
                 "--ids", output / "unlabeled_pool.json", "--output", pool_scores_path,
                 "--device", "cuda", "--batch-size", 16, log=output / "round_0_score.log")
    scores = _read(pool_scores_path)
    if len(scores) != len(pool) or {row["candidate_id"] for row in scores} != set(pool):
        raise ValueError("Qwen pool scores must cover every unlabeled training ID exactly once")
    usage = semantic.usage()
    target_train = math.ceil(usage["max_calls"] * 0.75)
    sample_path = output / "al_sample_ids.json"
    if sample_path.exists():
        selected = _read(sample_path)
        stratum_size = sum(row["score"] >= first_meta["threshold"] for row in scores)
        by_id = {row["candidate_id"]: row["score"] for row in scores}
        if len(selected) != len(set(selected)) or any(
                by_id.get(cid, -1) < first_meta["threshold"] for cid in selected):
            raise ValueError("saved AL sample must contain unique predicted-minority IDs")
    else:
        sample_count = min(target_train - len(train), usage["remaining"])
        if sample_count <= 0:
            raise ValueError("no budget remains for minority-stratum active learning")
        selected, stratum_size = minority_sample(scores, first_meta["threshold"], sample_count)
        write_json(sample_path, selected)
    cached = semantic.cached_labels()
    pending = [cid for cid in selected if cid not in cached]
    if pending:
        semantic.label_many(pending, workers=8)
    cached = semantic.cached_labels()
    if any(cid not in cached for cid in selected):
        raise ValueError("some AL teacher labels failed; retain ledger and resume after inspection")
    al_rows = [{"candidate_id": cid, "label": cached[cid]} for cid in selected]
    write_json(output / "train_round_1.json", al_rows)
    trace = _read(trace_path)
    if not any(row["strategy"] == "al_minority" for row in trace):
        trace.append({"strategy": "al_minority", "predicted_minority_stratum": stratum_size,
                      "sample_count": len(selected), "sample_ids": selected,
                      "positive": sum(row["label"] for row in al_rows),
                      "negative": sum(not row["label"] for row in al_rows),
                      "hit_rate": sum(row["label"] for row in al_rows) / len(al_rows),
                      "cumulative_positive": sum(row["label"] for row in train + al_rows),
                      "cumulative_negative": sum(not row["label"] for row in train + al_rows),
                      "rho_before": _rho(train), "rho_after": _rho(train + al_rows)})
        write_json(trace_path, trace)
    final = output / "round_1"
    if not (final / "metadata.json").exists():
        _trainer(workspace, "fit", "--model-name", model_name,
                 "--train", output / "train_round_0.json",
                 "--train", output / "train_round_1.json",
                 "--validation", output / "validation.json", "--output-dir", final,
                 "--previous-artifact", first / "proxy.pkl", "--device", "cuda",
                 "--seed", 42, log=output / "round_1.log")
    final_meta = _read(final / "metadata.json")
    if (final_meta["model_name"] != model_name or not final_meta["proxy_valid"] or
            final_meta["train_count"] != len(train) + len(al_rows)):
        raise ValueError("final Qwen proxy is invalid, wrong-model, or missed training rounds")
    write_json(output / "universe_ids.json", universe)
    all_scores_path = output / "round_1_all_scores.json"
    if not all_scores_path.exists():
        _trainer(workspace, "score", "--artifact", final / "proxy.pkl",
                 "--ids", output / "universe_ids.json", "--output", all_scores_path,
                 "--device", "cuda", "--batch-size", 16, log=output / "round_1_score.log")
    all_scores = _read(all_scores_path)
    if len(all_scores) != len(universe) or {row["candidate_id"] for row in all_scores} != set(universe):
        raise ValueError("final Qwen scores must cover the full population")
    pure = {row["candidate_id"] for row in all_scores if row["score"] >= final_meta["threshold"]}
    write_json(workspace / "output/candidate_ids.json", sorted(pure))
    pure_metrics = evaluate(run_dir)
    write_json(output / "pure_proxy_evaluation.json", pure_metrics)
    hybrid = set(pure)
    for cid, label in semantic.cached_labels().items():
        (hybrid.add if label else hybrid.discard)(cid)
    write_json(workspace / "output/candidate_ids.json", sorted(hybrid))
    hybrid_metrics = evaluate(run_dir)
    report = {"model_name": model_name, "backend": "qwen_online_lora",
              "initial_proxy": first_meta, "final_proxy": final_meta,
              "sampling_trace": trace, "pure_proxy": pure_metrics,
              "hybrid": hybrid_metrics, "usage": semantic.usage()}
    write_json(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "usage": report["usage"],
                      "pure_proxy": pure_metrics, "hybrid": hybrid_metrics}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    args = parser.parse_args()
    run(args.run_dir, args.model_name)


if __name__ == "__main__":
    main()
