#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=1fbe15c9b0ce900bfffbf5c148ca61c81cb34046
export TIDE_HARNESS_COMMIT=1fbe15c9b0ce900bfffbf5c148ca61c81cb34046
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1/status.json --source 1fbe15c9b0ce900bfffbf5c148ca61c81cb34046 --max-rss-gib 32.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import hashlib,json,os,pathlib,shutil,subprocess,sys
from scripts.job import atomic
source,run=map(pathlib.Path,sys.argv[1:]); build=run/'"'"'build'"'"';shutil.copytree(source,build)
manifest=dict(source=os.environ['"'"'TIDE_SOURCE_COMMIT'"'"'],files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((source/'"'"'src/tide'"'"').glob('"'"'*.py'"'"'))})
atomic(run/'"'"'source-manifest.json'"'"',manifest)
commands=[]
def execute(argv,**kw):
 commands.append(argv);atomic(run/'"'"'commands.json'"'"',commands);subprocess.run(argv,check=True,**kw)
execute([sys.executable,'"'"'-m'"'"','"'"'pip'"'"','"'"'wheel'"'"','"'"'--no-deps'"'"','"'"'--no-build-isolation'"'"','"'"'--no-index'"'"',str(build),'"'"'-w'"'"',str(run/'"'"'wheels'"'"')])
execute([sys.executable,'"'"'-m'"'"','"'"'venv'"'"','"'"'--system-site-packages'"'"',str(run/'"'"'venv'"'"')])
python=str(run/'"'"'venv/bin/python'"'"');wheel=next((run/'"'"'wheels'"'"').glob('"'"'*.whl'"'"'))
execute([python,'"'"'-m'"'"','"'"'pip'"'"','"'"'install'"'"','"'"'--no-deps'"'"','"'"'--no-index'"'"',str(wheel)])
env={k:v for k,v in os.environ.items() if k!='"'"'PYTHONPATH'"'"'}
execute([python,'"'"'-I'"'"',str(source/'"'"'scripts/check_installed_capacity.py'"'"'),'"'"'--execution-variant'"'"','"'"'reception'"'"','"'"'--output-dir'"'"',str(run/'"'"'installed'"'"'),'"'"'--wheel'"'"',str(wheel),'"'"'--venv'"'"',str(run/'"'"'venv'"'"'),'"'"'--source-manifest'"'"',str(run/'"'"'source-manifest.json'"'"')],cwd='"'"'/'"'"',env=env)
execute([python,'"'"'-I'"'"',str(source/'"'"'examples/train.py'"'"'),'"'"'--config'"'"',str(source/'"'"'configs/agg-inference/settle-incoming_light_64-optimized.json'"'"'),'"'"'--parameter-indexing'"'"','"'"'static_view'"'"','"'"'--ssd-scan-backend'"'"','"'"'doubling'"'"','"'"'--batch'"'"','"'"'1'"'"','"'"'--length'"'"','"'"'2'"'"','"'"'--steps'"'"','"'"'3'"'"','"'"'--checkpoint'"'"',str(run/'"'"'example.pt'"'"')],cwd='"'"'/'"'"',env=env)
execute([python,'"'"'-I'"'"',str(source/'"'"'examples/train.py'"'"'),'"'"'--resume'"'"',str(run/'"'"'example.pt'"'"'),'"'"'--parameter-indexing'"'"','"'"'static_view'"'"','"'"'--batch'"'"','"'"'1'"'"','"'"'--length'"'"','"'"'2'"'"','"'"'--steps'"'"','"'"'1'"'"'],cwd='"'"'/'"'"',env=env)
atomic(run/'"'"'installation-complete.json'"'"',dict(state='"'"'completed'"'"',source=manifest['"'"'source'"'"'],wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),commands=commands))
' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1/source /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-installed-final-v1
