#!/usr/bin/env bash
set -euo pipefail

# One-shot, non-monitoring launcher for the already completed Qwen-Doubao 2,000
# teacher ledger. This is an architecture experiment, not an exact USA seed-11
# comparison unless the ledger hash is later shown to match the USA protocol.

repo="${OKAGENT_REPO:-/home/mengsq/projects/OkAgent}"
branch="${OKAGENT_ROUTER_BRANCH:-codex/job01-multi-proxy-router}"
worktree="${OKAGENT_ROUTER_WORKTREE:-/home/mengsq/projects/OkAgent-router-exp}"
python_bin="${OKAGENT_PYTHON:-$repo/.venv/bin/python}"
ledger="${OKAGENT_TRAIN_LEDGER:-$repo/results/comparison/qwen_doubao_segment_soft_al_full34761_mc2000_r1/workspace/output/qwen_active_full/teacher.sqlite}"
output="${OKAGENT_ROUTER_OUTPUT:-$repo/results/comparison/multi_proxy_router_qwen_ids_job01_r1}"
log="$output/run.log"
pid_file="$output/run.pid"

test -d "$repo/.git"
test -x "$python_bin"
test -f "$ledger"

mkdir -p "$output"
if [[ -f "$output/report.json" ]]; then
  echo "Router experiment is already complete: $output/report.json"
  exit 0
fi
if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "Router experiment already running with PID $(cat "$pid_file")"
  exit 0
fi

git -C "$repo" fetch origin "$branch"
expected_commit="$(git -C "$repo" rev-parse "origin/$branch")"
if [[ ! -d "$worktree/.git" && ! -f "$worktree/.git" ]]; then
  test ! -e "$worktree"
  git -C "$repo" worktree add --detach "$worktree" "$expected_commit"
fi

actual_commit="$(git -C "$worktree" rev-parse HEAD)"
if [[ "$expected_commit" != "$actual_commit" ]]; then
  if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
    echo "Existing experiment worktree is stale and dirty: $actual_commit " \
         "(expected $expected_commit)" >&2
    exit 2
  fi
  git -C "$worktree" checkout --detach "$expected_commit"
fi

"$python_bin" -c 'import duckdb,numpy,scipy,sklearn'

nohup env PYTHONPATH="$worktree/src:$worktree" PYTHONUTF8=1 \
  "$python_bin" -m benchmarks.multi_proxy_router_job01 \
  --config "$repo/_config/hiring.json" \
  --train-ledger "$ledger" \
  --expected-train-count 2000 \
  --seed 11 \
  --output "$output" \
  >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
printf 'Started PID %s\nLog: %s\nReport: %s\n' "$pid" "$log" "$output/report.json"
