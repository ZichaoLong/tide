#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 8.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -I /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15-inputs/driver.py --source /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15/source --run /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15 --smoke /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/wheel-installed-cpu-v15-inputs/smoke.py
