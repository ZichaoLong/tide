#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 160.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/dependency.json --seconds 14400 --unit tide-representative-cpu-v5.service --unit tide-scale-two-npu-hb-17b-v12.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id scale-four-npu-hb-9b-train-v14 --project tide-instances --priority 0 --npu-count 4 --max-wait-seconds 1800 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import subprocess,sys; codes=[subprocess.call([sys.executable,"-m","scripts.check_placement","--devices","npu:0","npu:1","npu:2","npu:3","--mode","train","--config","configs/scale/hb-9b.json","--precision",p,"--threads","16","--batch","1","--length","8","--steps","2","--no-foreach","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-four-npu-hb-9b-train-v14/"+p+".json"]) for p in ("fp32","bf16")]; sys.exit(int(any(codes)))'
