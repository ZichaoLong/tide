#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/source/src
export TIDE_SOURCE_COMMIT=c8026e47ee5b6fbe03b9d8c5e84e2bfde70f8267
export TIDE_HARNESS_COMMIT=c8026e47ee5b6fbe03b9d8c5e84e2bfde70f8267
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/status.json --source c8026e47ee5b6fbe03b9d8c5e84e2bfde70f8267 --max-rss-gib 192.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/dependency.json --seconds 10800 --unit tide-representative-cpu-v5.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id scale-two-npu-hb-17b-v12 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 1200 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import subprocess,sys; codes=[subprocess.call([sys.executable,"-m","scripts.check_placement","--devices","npu:0","npu:1","--mode","inference","--config","configs/scale/hb-17b.json","--precision",p,"--threads","16","--batch","1","--length","8","--warmup","1","--steps","3","--output","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/scale-two-npu-hb-17b-v12/"+p+".json"]) for p in ("fp32","bf16")]; sys.exit(int(any(codes)))'
