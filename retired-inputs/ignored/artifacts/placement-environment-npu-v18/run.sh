#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=ca6199d427ea452492422bb7be21c03ce45aab5c
export TIDE_HARNESS_COMMIT=ca6199d427ea452492422bb7be21c03ce45aab5c
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18/status.json --source ca6199d427ea452492422bb7be21c03ce45aab5c --max-rss-gib 24.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id placement-environment-npu-v18 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18/queue.json -- numactl --membind=2 --physcpubind=80-83 /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import pathlib,subprocess,sys
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/placement-environment-npu-v18")
rows=[]
for transport,precision in (("auto","bf16"),("synchronized","fp32"),("direct","fp32")):
    command=[sys.executable,"-m","scripts.check_placement","--devices","npu:0","npu:1","--precision",precision,"--transport",transport,"--threads","4","--output",str(root/(transport+"-"+precision+".json"))]
    code=subprocess.call(command)
    rows.append(dict(transport=transport,precision=precision,command=command,exit_code=code))
    atomic(root/"attempts.json",dict(state="running",rows=rows))
atomic(root/"attempts.json",dict(state="completed",rows=rows,scope="fresh processes after CANN Python path correction; original failures retained"))
sys.exit(int(any(row["exit_code"] for row in rows)))
'
