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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1/status.json --source 8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee --max-rss-gib 96.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id region-placement-perf-npu-v1 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 1200 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,sys,json
from pathlib import Path
from scripts.job import atomic,enable_subreaper
from benchmarks.programme import run_child
out=Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/region-placement-perf-npu-v1");record=dict(source=os.environ["TIDE_SOURCE_COMMIT"],physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),state="running",children=[])
persist=lambda:atomic(out/"programme.json",record)
enable_subreaper()
for config,batch,length in (("settle",8,256),("timed",4,128)):
 for precision in ("fp32","bf16"):
  for label,devices,policy in (("single",["npu:0","npu:0"],"serial"),("dual_serial",["npu:0","npu:1"],"serial"),("dual_per_device",["npu:0","npu:1"],"per_device")):
   name=f"{config}-{precision}-{label}"
   command=[sys.executable,"-m","scripts.check_placement","--config","configs/hb-multiscale-x-"+config+".json","--devices",*devices,"--partition-kind","region","--region-concurrency",policy,"--transport","host_stage","--mode","train","--precision",precision,"--batch",str(batch),"--length",str(length),"--steps","6","--threads","4","--output",str(out/(name+".json"))]
   row=dict(identity=name,command=command,output=str(out/(name+".json")),log=str(out/(name+".log")),timeout_seconds=600,max_rss_bytes=80*2**30)
   record["children"].append(row);persist();run_child(row,persist,supervisor_pid=os.getpid())
record["state"]="completed" if all(r["state"]=="completed" for r in record["children"]) else "failed"
persist();sys.exit(record["state"]!="completed")
'
