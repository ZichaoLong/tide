#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee
export TIDE_HARNESS_COMMIT=8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/status.json --source 8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee --max-rss-gib 256.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/dependency.json --seconds 4200 --unit tide-native-regional-perf-npu-v1.service --unit tide-execution-perf-timed-bf16-v1.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id capacity-region-17b-npu-v1 --project tide-instances --priority 0 --npu-count 6 --max-wait-seconds 1200 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,sys,subprocess
from pathlib import Path
from scripts.job import atomic
out=Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-17b-npu-v1"); rows=[]
for precision in ("fp32","bf16"):
 command=[sys.executable,"-m","scripts.check_placement","--config","configs/scale/hb-17b.json","--devices","npu:0","npu:1","npu:2","npu:3","npu:4","npu:5","--partition-kind","region","--mode","train","--precision",precision,"--batch","1","--length","8","--steps","2","--no-foreach","--threads","8","--output",str(out/(precision+".json"))]
 with (out/(precision+".log")).open("w") as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
 rows.append(dict(precision=precision,command=command,exit_code=result.returncode));atomic(out/"children.json",dict(source=os.environ["TIDE_SOURCE_COMMIT"],children=rows))
 if result.returncode: break
sys.exit(any(r["exit_code"] for r in rows))
'
