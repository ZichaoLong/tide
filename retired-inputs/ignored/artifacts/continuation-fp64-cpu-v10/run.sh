#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-fp64-cpu-v10/source/src
export TIDE_SOURCE_COMMIT=0e0023ff773f56920da78374e26ded6f10ceac06
export TIDE_HARNESS_COMMIT=0e0023ff773f56920da78374e26ded6f10ceac06
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-fp64-cpu-v10/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-fp64-cpu-v10/status.json --source 0e0023ff773f56920da78374e26ded6f10ceac06 --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.check_continuation --device cpu --precision fp64 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-fp64-cpu-v10/result.json
