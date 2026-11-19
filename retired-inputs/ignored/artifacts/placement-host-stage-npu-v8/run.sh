#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-host-stage-npu-v8/source/src
export TIDE_SOURCE_COMMIT=20062c170e85ef0c5fd91130b6df724f90622913
export TIDE_HARNESS_COMMIT=20062c170e85ef0c5fd91130b6df724f90622913
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-host-stage-npu-v8/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-host-stage-npu-v8/status.json --source 20062c170e85ef0c5fd91130b6df724f90622913 --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id placement-host-stage-npu-v8 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-host-stage-npu-v8/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,subprocess,sys; os.environ.pop("ASCEND_LAUNCH_BLOCKING",None); codes=[subprocess.call([sys.executable,"-m","scripts.check_placement","--devices","npu:0","npu:1","--transport","host_stage","--precision",p,"--steps","3","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-host-stage-npu-v8/"+p+".json"]) for p in ("fp32","bf16")]; sys.exit(int(any(codes)))'
