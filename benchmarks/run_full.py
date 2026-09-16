"""Reproducible entry point for one full live hiring experiment."""
import argparse
import json
from pathlib import Path

from okagent.agent import run_agent
from okagent.data import prepare
from okagent.evaluation import evaluate
from okagent.lo_ph_agent import run_lo_ph_agent
from okagent.qwen_proxy import load_qwen_features
from okagent.semantic import SemanticOperator
from other_methods.hydra import Config, run_hiring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("precompute", "baseline", "lo_ph", "hydra"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="_config/hiring.json")
    parser.add_argument("--command-timeout", type=int, default=7200)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        prepare(args.config, run_dir, job=args.job)
    task = json.loads((run_dir / "task.json").read_text(encoding="utf-8"))

    if args.method == "precompute":
        _, vectors, _, metadata = load_qwen_features(
            run_dir / "workspace/data.duckdb", run_dir / "workspace/job.json")
        result = {"candidate_count": len(vectors), "features": metadata}
    elif args.method == "baseline":
        result = run_agent(run_dir, command_timeout=args.command_timeout)
    elif args.method == "lo_ph":
        result = run_lo_ph_agent(run_dir, command_timeout=args.command_timeout)
    else:
        operator = SemanticOperator(run_dir / "workspace")

        def label(candidate_ids):
            return operator.label_many(candidate_ids, workers=min(8, len(candidate_ids)))

        config = Config.hiring_cpu(max_calls=task["max_calls"])
        method = run_hiring(run_dir, config=config, label=label, feature_backend="qwen")
        result = {"method": method["summary"], "evaluation": evaluate(run_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
