#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-npu-v4/source/src
export TIDE_SOURCE_COMMIT=bdcb81de615e9aa4b21e7443871d33149b4f1553
export TIDE_HARNESS_COMMIT=bdcb81de615e9aa4b21e7443871d33149b4f1553
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-npu-v4/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-npu-v4/status.json --source bdcb81de615e9aa4b21e7443871d33149b4f1553 --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id placement-npu-v4 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-npu-v4/queue.json -- /bin/bash -c 'for tide_precision in fp32 bf16; do
 python -m scripts.check_placement --devices npu:0 npu:1 --precision ${tide_precision} --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-npu-v4/${tide_precision}.json || exit
done'
