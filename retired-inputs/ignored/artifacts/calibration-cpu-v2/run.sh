#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/calibration-cpu-v2/source/src
export TIDE_SOURCE_COMMIT=3b9f919b560d829fbed70302749ba4c9584bbd9b
export TIDE_HARNESS_COMMIT=3b9f919b560d829fbed70302749ba4c9584bbd9b
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/calibration-cpu-v2/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/calibration-cpu-v2/status.json --source 3b9f919b560d829fbed70302749ba4c9584bbd9b --max-rss-gib 64.0 -- /bin/bash -c 'for tide_case in fc-k hb-lattice-k; do
  for tide_threads in 4 8 16 32; do
    python -m benchmarks.run --config configs/${tide_case}.json --device cpu --precision fp32 --threads ${tide_threads} --batch 8 --length 128 --subjects tide --modes inference train --warmup 1 --steps 3 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/calibration-cpu-v2/${tide_case}-${tide_threads}.json || exit
  done
done'
