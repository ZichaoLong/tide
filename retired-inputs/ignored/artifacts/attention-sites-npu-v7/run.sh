#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-npu-v7/source/src
export TIDE_SOURCE_COMMIT=e2a4d39104a006feec4525e534eadae4a94cde0d
export TIDE_HARNESS_COMMIT=e2a4d39104a006feec4525e534eadae4a94cde0d
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-npu-v7/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-npu-v7/status.json --source e2a4d39104a006feec4525e534eadae4a94cde0d --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id attention-sites-npu-v7 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-npu-v7/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.attention_sites --device npu:0 --cases kv-unit kv-zero reception-positive reception-zero region-positive region-zero --threads 4 --warmup 2 --steps 5 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-npu-v7/result.json
