#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-operators-npu-v13/source/src
export TIDE_SOURCE_COMMIT=5059ff8f751a1aeb0e65d6d1346c6a2d461676c0
export TIDE_HARNESS_COMMIT=5059ff8f751a1aeb0e65d6d1346c6a2d461676c0
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-operators-npu-v13/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-operators-npu-v13/status.json --source 5059ff8f751a1aeb0e65d6d1346c6a2d461676c0 --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id attention-operators-npu-v13 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-operators-npu-v13/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.attention_sites --device npu:0 --cases kv-unit kv-zero reception-positive reception-zero region-positive region-zero --threads 4 --warmup 0 --steps 1 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-operators-npu-v13/result.json
