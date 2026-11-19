#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/source/src
export TIDE_SOURCE_COMMIT=9a1f600f2b712aaebf28a1a689d864b2358a5fa6
export TIDE_HARNESS_COMMIT=9a1f600f2b712aaebf28a1a689d864b2358a5fa6
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/status.json --source 9a1f600f2b712aaebf28a1a689d864b2358a5fa6 --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import subprocess,sys; base=[sys.executable,"-m","scripts.qualify_precision","--device","cpu","--width","0","--batch","2","--length","5","--steps","3","--threads","8","--checkpoint"]; a=subprocess.call(base+["--precision","fp32","--save-tensors","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/tensors","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/fp32.json"]); b=subprocess.call(base+["--precision","bf16","--reference","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/tensors","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/qualification-cpu-v8/bf16.json"]); sys.exit(int(bool(a or b)))'
