#!/bin/bash
set -euo pipefail
APP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HOME/.config/local-egress/proxy.sh"
source "$HOME/.config/job01-full-agent/ark.env"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN_DIR="${1:-$APP_DIR/runs/job01_adaptive_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/run.exit"' EXIT
"$APP_DIR/.venv/bin/python" -u "$APP_DIR/operator_agent.py" \
  --full-data-dir "$HOME/datasets/20260722" \
  --run-dir "$RUN_DIR" \
  --model ep-20250612104210-ss27q \
  --base-url https://ark.cn-beijing.volces.com/api/v3 \
  --seed 42 --test-size 0.2 --validation-size 0.2 --verbose
