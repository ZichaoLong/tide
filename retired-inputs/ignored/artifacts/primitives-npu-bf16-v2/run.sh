#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/primitives-npu-bf16-v2/source/src
export TIDE_SOURCE_COMMIT=3b9f919b560d829fbed70302749ba4c9584bbd9b
export TIDE_HARNESS_COMMIT=3b9f919b560d829fbed70302749ba4c9584bbd9b
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/primitives-npu-bf16-v2/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/primitives-npu-bf16-v2/status.json --source 3b9f919b560d829fbed70302749ba4c9584bbd9b --max-rss-gib 24.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id primitives-npu-bf16-v2 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/primitives-npu-bf16-v2/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.qualify_precision --device npu:0 --precision bf16 --primitives-only --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/primitives-npu-bf16-v2/result.json
