#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-npu-bf16-v5/source/src
export TIDE_SOURCE_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TIDE_HARNESS_COMMIT=778c65fb2961624bb416343eb3ad0f6f031deff7
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-npu-bf16-v5/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-npu-bf16-v5/status.json --source 778c65fb2961624bb416343eb3ad0f6f031deff7 --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id qualification-npu-bf16-v5 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-npu-bf16-v5/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.qualify_precision --device npu:0 --precision bf16 --width 0 --threads 4 --steps 3 --checkpoint --reference /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-fp32-v1/tensors --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-npu-bf16-v5/result.json
