#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/source/src
export TIDE_SOURCE_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TIDE_HARNESS_COMMIT=0992b467584c1f603314ed04d8efe8596a14f576
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/status.json --source 0992b467584c1f603314ed04d8efe8596a14f576 --max-rss-gib 48.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/dependency.json --seconds 18000 --unit tide-continuous-cpu-v16.service -- numactl --membind=7 --physcpubind=280-311 /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,pathlib,subprocess
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16")
command=['"'"'/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python'"'"', '"'"'-m'"'"', '"'"'benchmarks.attention_sites'"'"', '"'"'--device'"'"', '"'"'cpu'"'"', '"'"'--cases'"'"', '"'"'kv-unit'"'"', '"'"'kv-zero'"'"', '"'"'reception-positive'"'"', '"'"'reception-zero'"'"', '"'"'region-positive'"'"', '"'"'region-zero'"'"', '"'"'--threads'"'"', '"'"'32'"'"', '"'"'--warmup'"'"', '"'"'2'"'"', '"'"'--steps'"'"', '"'"'5'"'"', '"'"'--output'"'"', '"'"'/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/attention-sites-cpu-v16/result.json'"'"']
atomic(root/"numa-policy.json",dict(policy='"'"'--membind=7'"'"',affinity=sorted(os.sched_getaffinity(0)),maps=pathlib.Path("/proc/self/numa_maps").read_text(),status=pathlib.Path("/proc/self/status").read_text(),numactl=subprocess.check_output(["numactl","--show"],text=True),supersedes='"'"'attention-sites-cpu-v14'"'"'))
os.execv(command[0],command)'
