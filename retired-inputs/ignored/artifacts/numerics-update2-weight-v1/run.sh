#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=aab9a4d2ef6595269b2521e8632cbfad0a5a6648
export TIDE_HARNESS_COMMIT=aab9a4d2ef6595269b2521e8632cbfad0a5a6648
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/status.json --source aab9a4d2ef6595269b2521e8632cbfad0a5a6648 --max-rss-gib 128.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/dependency.json --seconds 1800 --unit tide-numerics-update2-baseline-v1.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.replay_saved_training_update --qualification-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-medium-cpu-qualification-v1/qualification --verified-inputs /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-saved-update-v1/diagnosis/result.json --owner weight_access --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/numerics-update2-weight-v1/replay.json
