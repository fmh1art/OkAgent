#!/usr/bin/env bash
set -euo pipefail

# Continue the two job01 experiments after their independent batch-label jobs
# finish. The supervisor polls local process/report state only; it performs no
# LLM calls itself and is safe to restart.

repo="${OKAGENT_REPO:-/home/mengsq/projects/OkAgent}"
python_bin="${OKAGENT_PYTHON:-$repo/.venv/bin/python}"
base="${OKAGENT_RESULTS:-$repo/results/comparison}"
round1="$base/job01_batch_multilabel_seed11"
round2="$base/job01_batch_multilabel_round2"
acquisition="$base/job01_router_acquisition_round2"
baseline="$base/multi_proxy_router_multilabel_job01_seed11"
active="$base/multi_proxy_router_multilabel_job01_round2"
supervisor_log="$base/job01_parallel_multilabel_supervisor.log"

export PYTHONPATH="$repo/src:$repo" PYTHONUTF8=1

report_complete() {
  local output="$1"
  [[ -f "$output/report.json" ]] && "$python_bin" - "$output/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("complete") is True else 1)
PY
}

process_running() {
  local output="$1"
  [[ -f "$output/run.pid" ]] && kill -0 "$(cat "$output/run.pid")" 2>/dev/null
}

require_live_or_complete() {
  local name="$1" output="$2"
  if report_complete "$output"; then
    return 0
  fi
  if process_running "$output"; then
    return 1
  fi
  printf '%s failed before producing a complete report\n' "$name" | tee -a "$supervisor_log"
  tail -n 80 "$output/run.log" 2>/dev/null | tee -a "$supervisor_log" || true
  return 2
}

launch_model() {
  local name="$1" output="$2"
  shift 2
  mkdir -p "$output"
  if [[ -f "$output/report.json" ]]; then
    printf '%s already complete\n' "$name" | tee -a "$supervisor_log"
    return
  fi
  if process_running "$output"; then
    printf '%s already running with PID %s\n' "$name" "$(cat "$output/run.pid")" \
      | tee -a "$supervisor_log"
    return
  fi
  nohup "$python_bin" -m benchmarks.multi_proxy_router_job01 \
    --config "$repo/_config/hiring.json" \
    --seed 11 \
    --comparison-protocol architecture_only \
    --output "$output" \
    "$@" >"$output/run.log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$output/run.pid"
  printf '%s started with PID %s\n' "$name" "$pid" | tee -a "$supervisor_log"
}

mkdir -p "$baseline" "$active"
printf 'Supervisor started at %s\n' "$(date --iso-8601=seconds)" >>"$supervisor_log"

while true; do
  if require_live_or_complete round1_2000 "$round1"; then
    round1_state=0
  else
    round1_state=$?
  fi
  if require_live_or_complete round2_1000 "$round2"; then
    round2_state=0
  else
    round2_state=$?
  fi
  if (( round1_state == 2 || round2_state == 2 )); then
    exit 2
  fi

  if (( round1_state == 0 )); then
    launch_model baseline_2000 "$baseline" \
      --train-ids "$round1/train_ids.json" \
      --batch-labels "$round1/labels.json" \
      --expected-train-count 2000
  fi

  if (( round1_state == 0 && round2_state == 0 )); then
    launch_model active_3000 "$active" \
      --train-ids "$round1/train_ids.json" \
      --train-ids "$acquisition/candidate_ids.json" \
      --batch-labels "$round1/labels.json" \
      --batch-labels "$round2/labels.json" \
      --expected-train-count 3000
  fi

  baseline_done=false
  active_done=false
  [[ -f "$baseline/report.json" ]] && baseline_done=true
  [[ -f "$active/report.json" ]] && active_done=true
  if [[ "$baseline_done" == true && "$active_done" == true ]]; then
    printf 'Both model experiments completed at %s\n' "$(date --iso-8601=seconds)" \
      | tee -a "$supervisor_log"
    exit 0
  fi

  for model in "$baseline" "$active"; do
    if [[ -f "$model/run.pid" && ! -f "$model/report.json" ]] && ! process_running "$model"; then
      printf 'Model process failed: %s\n' "$model" | tee -a "$supervisor_log"
      tail -n 80 "$model/run.log" 2>/dev/null | tee -a "$supervisor_log" || true
      exit 3
    fi
  done
  sleep 60
done
