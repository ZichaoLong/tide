#!/bin/bash
set -euo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /
env -u PYTHONPATH "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1/env/bin/python" -I "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-restore-diagnosis-v1/diagnose.py" --input "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-8a22530d-v1" --output "/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/worktrees/installed-check/artifacts/capacity-installed-restore-diagnosis-v1/result.json"
