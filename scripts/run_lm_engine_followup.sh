#!/usr/bin/env bash
set -euo pipefail
experiment_root=$1
shift
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.1.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
exec "$experiment_root/env/transformers/bin/python" -B scripts/run_lm_engine_followup.py "$experiment_root" "$@"
