#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=7b77457d594662e50acf0a002ca18d0a241e0ce3
export TIDE_HARNESS_COMMIT=7b77457d594662e50acf0a002ca18d0a241e0ce3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-budgets-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-budgets-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-budgets-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-budgets-v1/status.json --source 7b77457d594662e50acf0a002ca18d0a241e0ce3 --max-rss-gib 8.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'from benchmarks.capacity import placement_budget,scale_configs
from scripts.job import atomic
import os
rows=[]
for name,count in (("hb-9b",4),("hb-17b",5),("hb-17b",6),("hb-17b",2)):
 for precision in ("fp32","bf16"):
  rows.append(dict(name=name,parts=count,precision=precision,budget=placement_budget(scale_configs()[name],batch=1,length=8,precision=precision,mode="train",partitions=count,partition_kind="region")))
atomic("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-region-budgets-v1/budgets.json",dict(source=os.environ["TIDE_SOURCE_COMMIT"],rows=rows))
'
