#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-npu-fc-k-v5/source/src
export TIDE_SOURCE_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TIDE_HARNESS_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-npu-fc-k-v5/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-npu-fc-k-v5/status.json --source 778c65fb2961624bb416343eb3ad0f6f031deff7 --max-rss-gib 80.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id representative-npu-fc-k-v5 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-npu-fc-k-v5/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.programme --suite representative --configs configs/fc-k.json --device npu:0 --precision fp32 bf16 --threads 4 --batch 8 --length 128 --warmup 2 --steps 5 --child-timeout 1200 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/representative-npu-fc-k-v5/results
