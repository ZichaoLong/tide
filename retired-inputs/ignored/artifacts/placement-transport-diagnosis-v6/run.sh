#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-transport-diagnosis-v6/source/src
export TIDE_SOURCE_COMMIT=7bcb0df17f0613793d82511d9517e37e1870850a
export TIDE_HARNESS_COMMIT=7bcb0df17f0613793d82511d9517e37e1870850a
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-transport-diagnosis-v6/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-transport-diagnosis-v6/status.json --source 7bcb0df17f0613793d82511d9517e37e1870850a --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id placement-transport-diagnosis-v6 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-transport-diagnosis-v6/queue.json -- /bin/bash -c 'tide_failed=0
for tide_transport in direct host_stage synchronized; do
 python -m scripts.check_placement --devices npu:0 npu:1 --precision fp32 --transport ${tide_transport} --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-transport-diagnosis-v6/${tide_transport}.json || tide_failed=1
done
exit ${tide_failed}'
