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
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1/status.json --source 8a22530dae69d9f1fa5ead4a01d1798fd1dfe0ee --max-rss-gib 16.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id native-regional-npu-pool-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 1200 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,sys,importlib.util
from pathlib import Path
from scripts.job import atomic,enable_subreaper
from benchmarks.programme import run_child
root=Path("/mi/data2T/zlong/home/llm/graph-execution-foundation");out=Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/native-regional-npu-pool-v1")
spec=importlib.util.spec_from_file_location("native_reference",Path.cwd()/"native/regional/reference.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
check=root/"artifacts/native-regional-npu-same-v1/checks"
binary=root/"artifacts/worktrees/capacity-bindings/artifacts/native-89ce08ff-cpu/build-NPU/tide-regional"
launcher=root/"artifacts/native-sdk-launcher-89ce08ff.sh"
record=dict(source=os.environ["TIDE_SOURCE_COMMIT"],state="running",physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),children=[])
persist=lambda:atomic(out/"programme.json",record)
enable_subreaper()
for dtype in ("float32","bfloat16"):
 for workers in (2,4):
  for resumed in (False,True):
   name=f"{dtype}-w{workers}-"+("restored" if resumed else "train")
   path=out/name
   command=[str(launcher),str(binary),"--device","npu:0","--dtype",dtype,"--mode","train","--workers",str(workers),"--inner-threads","1","--streams","pool","--warmup","1","--repeats","1" if resumed else "3","--input-dir",str(check/"fixture"),"--output-dir",str(path)]
   if resumed:command+=["--resume",str(out/f"{dtype}-w{workers}-train"/"checkpoint.pt")]
   row=dict(identity=name,command=command,output=str(path/"result.json"),log=str(out/(name+".log")),timeout_seconds=120,max_rss_bytes=12*2**30)
   record["children"].append(row);persist();run_child(row,persist,supervisor_pid=os.getpid())
   if row["state"]!="completed":
    record["state"]="execution_failed";persist();sys.exit(1)
   reference=check/(f"native-{dtype}-restored-w1" if resumed else f"python-{dtype}-train")/"tensors"
   row["reference"]=str(reference)
   row["comparison"]=ref.compare(ref.read_bundle(path/"tensors"),ref.read_bundle(reference),rtol=2e-5,atol=2e-6)
   persist()
record["state"]="passed" if all(r["comparison"]["state"]=="passed" for r in record["children"]) else "failed"
persist();sys.exit(record["state"]!="passed")
'
