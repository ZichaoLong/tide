#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/projections-e2e-npu-v3/source/src
export TIDE_SOURCE_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TIDE_HARNESS_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/projections-e2e-npu-v3/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/projections-e2e-npu-v3/status.json --source c09e5d7cf18de19d73c242a6e461c14e5dba1101 --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id projections-e2e-npu-v3 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/projections-e2e-npu-v3/queue.json -- /bin/bash -c 'for tide_case in fc-k hb-lattice-k; do
 for tide_precision in fp32 bf16; do
  for tide_mode in inference train; do
   python -m benchmarks.hotspots --config configs/${tide_case}.json --device npu:0 --precision ${tide_precision} --mode ${tide_mode} --threads 4 --batch 8 --length 128 --candidates compact head_query both --warmup 2 --steps 5 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/projections-e2e-npu-v3/${tide_case}-${tide_precision}-${tide_mode}.json || exit
  done
 done
done'
