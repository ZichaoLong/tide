#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-npu-fc-k-v8/source/src
export TIDE_SOURCE_COMMIT=9a1f600f2b712aaebf28a1a689d864b2358a5fa6
export TIDE_HARNESS_COMMIT=9a1f600f2b712aaebf28a1a689d864b2358a5fa6
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-npu-fc-k-v8/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-npu-fc-k-v8/status.json --source 9a1f600f2b712aaebf28a1a689d864b2358a5fa6 --max-rss-gib 80.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id continuous-npu-fc-k-v8 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 1200 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-npu-fc-k-v8/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import subprocess,sys; codes=[subprocess.call([sys.executable,"-m","benchmarks.programme","--suite","continuous","--configs","configs/fc-k.json","--device","npu:0","--precision","fp32","bf16","--threads","4","--batch",str(b),"--length","128","--prefill-chunk","64","--decode","8","--warmup","1","--steps","3","--continuous-joint","A","B","C","D","--checkpoints","--child-timeout","900","--output-dir","/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/continuous-npu-fc-k-v8/b"+str(b)]) for b in (1,4)]; sys.exit(int(any(codes)))'
