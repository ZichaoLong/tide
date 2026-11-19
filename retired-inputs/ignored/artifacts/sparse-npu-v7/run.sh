#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/sparse-npu-v7/source/src
export TIDE_SOURCE_COMMIT=e2a4d39104a006feec4525e534eadae4a94cde0d
export TIDE_HARNESS_COMMIT=e2a4d39104a006feec4525e534eadae4a94cde0d
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/sparse-npu-v7/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/sparse-npu-v7/status.json --source e2a4d39104a006feec4525e534eadae4a94cde0d --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id sparse-npu-v7 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/sparse-npu-v7/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.programme --suite sparse --configs configs/fc-k.json --device npu:0 --precision fp32 bf16 --threads 4 --batch 4 --length 32 --warmup 1 --steps 3 --child-timeout 900 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/sparse-npu-v7/results
