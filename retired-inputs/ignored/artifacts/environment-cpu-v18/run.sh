#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=ca6199d427ea452492422bb7be21c03ce45aab5c
export TIDE_HARNESS_COMMIT=ca6199d427ea452492422bb7be21c03ce45aab5c
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/environment-cpu-v18/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/environment-cpu-v18/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/environment-cpu-v18/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/environment-cpu-v18/status.json --source ca6199d427ea452492422bb7be21c03ce45aab5c --max-rss-gib 4.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util,json,os,pathlib,sys
import tbe,tide
from scripts.job import atomic
root=pathlib.Path("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/environment-cpu-v18")
expected=root/"source/src/tide/__init__.py"
assert pathlib.Path(tide.__file__)==expected
assert importlib.util.find_spec("tbe") is not None
assert "/opt/software/ascend-runtime/9.2.0-r1/python" in os.environ["PYTHONPATH"].split(":")
atomic(root/"result.json",dict(state="passed",tide=tide.__file__,tbe=tbe.__file__,python=sys.executable,pythonpath=os.environ["PYTHONPATH"],scope="actual generated service environment; CPU import only, no accelerator qualification"))
'
