#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-attention-npu-v1/source/src
export TIDE_SOURCE_COMMIT=050dafa305d45162b2c86e736253455d7b476d27
export TIDE_HARNESS_COMMIT=050dafa305d45162b2c86e736253455d7b476d27
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-attention-npu-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-attention-npu-v1/status.json --source 050dafa305d45162b2c86e736253455d7b476d27 --max-rss-gib 24.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id qualification-attention-npu-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-attention-npu-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python tests/device_attention.py --device npu:0 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-attention-npu-v1/result.json
