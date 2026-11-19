#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/hotspots-cpu-v3/source/src
export TIDE_SOURCE_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TIDE_HARNESS_COMMIT=c09e5d7cf18de19d73c242a6e461c14e5dba1101
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/hotspots-cpu-v3/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/hotspots-cpu-v3/status.json --source c09e5d7cf18de19d73c242a6e461c14e5dba1101 --max-rss-gib 64.0 -- /bin/bash -c 'for tide_case in fc-k hb-lattice-k; do
 for tide_precision in fp32 bf16; do
  for tide_mode in inference train; do
   python -m benchmarks.hotspots --config configs/${tide_case}.json --device cpu --precision ${tide_precision} --mode ${tide_mode} --threads 32 --batch 8 --length 128 --candidates argsort compact bucketed head_query both --warmup 1 --steps 3 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/hotspots-cpu-v3/${tide_case}-${tide_precision}-${tide_mode}.json || exit
  done
 done
done'
