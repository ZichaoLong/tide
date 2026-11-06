#!/usr/bin/env bash
set -euo pipefail
experiment_root=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/local_ascend.sh"
tide_load_ascend
tide_lm_environment "$experiment_root"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
exec "$lm_python" -B scripts/run_lm_engine_followup.py "$experiment_root" "$@"
