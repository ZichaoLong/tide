#!/usr/bin/env bash
# Fixed-source service payload; public stack is loaded read-only in this shell.
set -euo pipefail
experiment_root=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/local_ascend.sh"
tide_load_ascend
tide_lm_environment "$experiment_root"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
exec "$lm_python" -B -m tidegraph lm "$@"
