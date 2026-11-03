#!/usr/bin/env bash
# Fixed-source qualification finishes before independent compute services launch.
set -euo pipefail
experiment_root=$1
attempt=$2
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
/bin/bash scripts/qualify_lm_runner.sh "$experiment_root"
exec "$experiment_root/env/transformers/bin/python" -B scripts/dispatch_lm_remaining.py "$experiment_root" \
  --attempt "$attempt" --small-job assets-small-d --moe-job assets-moe-d --wait-hours 30
