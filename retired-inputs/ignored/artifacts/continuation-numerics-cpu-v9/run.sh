#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-numerics-cpu-v9/source/src
export TIDE_SOURCE_COMMIT=5185dc0d1d3ae4c570e0712a42826e3974161bed
export TIDE_HARNESS_COMMIT=5185dc0d1d3ae4c570e0712a42826e3974161bed
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-numerics-cpu-v9/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-numerics-cpu-v9/status.json --source 5185dc0d1d3ae4c570e0712a42826e3974161bed --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import subprocess,sys; codes=[subprocess.call([sys.executable,"-m","scripts.check_continuation","--device","cpu","--precision",p,"--threads","4","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuation-numerics-cpu-v9/"+p+".json"]) for p in ("fp32","bf16","fp64")]; sys.exit(int(any(codes)))'
