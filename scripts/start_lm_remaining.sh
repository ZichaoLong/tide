#!/usr/bin/env bash
# Fixed-source qualification finishes before independent compute services launch.
set -euo pipefail
experiment_root=$1
attempt=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/local_ascend.sh"
tide_load_ascend
tide_lm_environment "$experiment_root"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
/bin/bash scripts/qualify_lm_runner.sh "$experiment_root"
exec "$lm_python" -B scripts/dispatch_lm_remaining.py "$experiment_root" \
  --attempt "$attempt" --small-job assets-small-d --moe-job assets-moe-d --wait-hours 30
