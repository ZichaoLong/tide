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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1/status.json --source 8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id regional-rounding-npu-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,sys,subprocess
from pathlib import Path
from scripts.job import atomic
out=Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-rounding-npu-v1");rows=[]
for kind in ("settle","timed"):
 command=[sys.executable,"-m","scripts.diagnose_region_batch","--config","configs/hb-multiscale-x-"+kind+".json","--device","npu:0","--precision","bf16","--threads","1","--output-dir",str(out/kind)]
 with (out/(kind+".log")).open("w") as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 rows.append(dict(kind=kind,command=command,exit_code=result.returncode));atomic(out/"children.json",dict(source=os.environ["TIDE_SOURCE_COMMIT"],children=rows))
sys.exit(any(r["exit_code"] for r in rows))
'
