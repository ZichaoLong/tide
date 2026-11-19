#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-bf16-v1/source/src
export TIDE_SOURCE_COMMIT=050dafa305d45162b2c86e736253455d7b476d27
export TIDE_HARNESS_COMMIT=050dafa305d45162b2c86e736253455d7b476d27
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-bf16-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-bf16-v1/status.json --source 050dafa305d45162b2c86e736253455d7b476d27 --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.qualify_precision --device cpu --precision bf16 --width 0 --threads 8 --steps 3 --checkpoint --reference /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-fp32-v1/tensors --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-bf16-v1/result.json
