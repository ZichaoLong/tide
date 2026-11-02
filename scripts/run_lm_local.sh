#!/usr/bin/env bash
# Fixed-source service payload; public stack is loaded read-only in this shell.
set -euo pipefail
experiment_root=$1
shift
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
exec "$experiment_root/env/transformers/bin/python" -B -m tidegraph lm "$@"
