#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=301a327569aa75aefddd03cac9ca839589db5672
export TIDE_HARNESS_COMMIT=301a327569aa75aefddd03cac9ca839589db5672
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1/status.json --source 301a327569aa75aefddd03cac9ca839589db5672 --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
from scripts.job import atomic
run,source=map(Path,sys.argv[1:3])
build=run/"build-source";shutil.copytree(source,build)
venv=run/"venv"
subprocess.run([sys.executable,"-m","venv","--system-site-packages",str(venv)],check=True)
python=str(venv/"bin/python")
environment=dict(os.environ);environment.pop("PYTHONPATH",None)
subprocess.run([python,"-m","pip","wheel","--no-cache-dir","--no-deps","--no-build-isolation","--wheel-dir",str(run/"wheels"),str(build)],check=True,env=environment,cwd="/")
wheel=next((run/"wheels").glob("*.whl"))
subprocess.run([python,"-m","pip","install","--no-deps","--no-cache-dir",str(wheel)],check=True,env=environment,cwd="/")
atomic(run/"installation.json",dict(source=os.environ["TIDE_SOURCE_COMMIT"],wheel=str(wheel),sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),venv=str(venv)))
result=subprocess.run([python,"-I",str(source/"scripts/check_installed_memory.py"),"--output-dir",str(run/"results"),"--expected-package-root",str(venv)],env=environment,cwd="/")
raise SystemExit(result.returncode)
' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1 /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/memory-installed-v1/source
