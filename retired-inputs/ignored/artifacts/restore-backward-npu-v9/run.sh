#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/restore-backward-npu-v9/source/src
export TIDE_SOURCE_COMMIT=5185dc0d1d3ae4c570e0712a42826e3974161bed
export TIDE_HARNESS_COMMIT=5185dc0d1d3ae4c570e0712a42826e3974161bed
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/restore-backward-npu-v9/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/restore-backward-npu-v9/status.json --source 5185dc0d1d3ae4c570e0712a42826e3974161bed --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id restore-backward-npu-v9 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/restore-backward-npu-v9/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.diagnose_restore --device npu:0 --precision bf16 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/restore-backward-npu-v9/result.json
