"""Launch a logged matrix of independent full experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--jobs", nargs="+", default=["job01", "job04"])
    parser.add_argument("--methods", nargs="+", default=["baseline", "lo_ph", "hydra"])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--config", default="_config/hiring.json")
    args = parser.parse_args()
    if args.max_parallel <= 0:
        raise ValueError("max-parallel must be positive")
    allowed = {"baseline", "lo_ph", "hydra"}
    if not set(args.methods) <= allowed:
        raise ValueError(f"methods must be from {sorted(allowed)}")

    root = Path(args.results_root).resolve()
    log_dir = root / f"matrix-{args.tag}-logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for job in args.jobs:
        for method in args.methods:
            run_dir = root / f"{args.tag}-{method}-{job}"
            if run_dir.exists():
                raise FileExistsError(run_dir)
            entries.append((job, method, run_dir, log_dir / f"{method}-{job}.log"))

    def execute(entry):
        job, method, run_dir, log = entry
        started = time.time()
        command = [sys.executable, "benchmarks/run_full.py", method, "--job", job,
                   "--run-dir", str(run_dir), "--config", args.config]
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=os.environ.copy(), check=False)
        return dict(job=job, method=method, run_dir=str(run_dir), log=str(log),
                    returncode=completed.returncode, seconds=time.time() - started)

    manifest = dict(tag=args.tag, jobs=args.jobs, methods=args.methods,
                    max_parallel=args.max_parallel, started=time.time(), runs=[])
    manifest_path = log_dir / "manifest.json"
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(execute, entry): entry for entry in entries}
        for future in as_completed(futures):
            result = future.result()
            manifest["runs"].append(result)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["finished"] = time.time()
    manifest["success"] = all(run["returncode"] == 0 for run in manifest["runs"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not manifest["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
