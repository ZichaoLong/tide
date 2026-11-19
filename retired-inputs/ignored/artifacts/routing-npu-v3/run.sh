#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/source/src
export TIDE_SOURCE_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TIDE_HARNESS_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/status.json --source c09e5d7cf18de19d73c242a6e461c14e5dba1101 --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id routing-npu-v3 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/queue.json -- /bin/bash -c 'for tide_case in hb-lattice-k hb-exchange-x hb-multiscale-x-timed; do
 python -m scripts.diagnose_bf16 capture --device npu:0 --config configs/${tide_case}.json --tiled --trace /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/${tide_case}.pt --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/routing-npu-v3/${tide_case}.json || exit
done'
