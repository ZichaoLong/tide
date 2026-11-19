#!/bin/bash
set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/check-v1/source/scripts/check_native_region.py" --binary "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/build-v2/cpu/tide-regional" --device cpu --output-dir "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/capacity-bindings/artifacts/native-development/check-v1/result"
