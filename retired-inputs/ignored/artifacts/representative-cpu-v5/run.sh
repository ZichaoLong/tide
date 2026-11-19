#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-cpu-v5/source/src
export TIDE_SOURCE_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TIDE_HARNESS_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-cpu-v5/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-cpu-v5/status.json --source 778c65fb2961624bb416343eb3ad0f6f031deff7 --max-rss-gib 128.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.programme --suite representative --device cpu --precision fp32 bf16 --threads 32 --batch 8 --length 128 --warmup 2 --steps 5 --child-timeout 1800 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-cpu-v5/results
