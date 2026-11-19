#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=a58536608f7276a771c56703b48e23ea82d2ca65
export TIDE_HARNESS_COMMIT=a58536608f7276a771c56703b48e23ea82d2ca65
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-large-cpu-timing-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-large-cpu-timing-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-large-cpu-timing-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-large-cpu-timing-v1/status.json --source a58536608f7276a771c56703b48e23ea82d2ca65 --max-rss-gib 192.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.execution_suite --config configs/memory-candidates/9b-kv128.json --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-large-cpu-timing-v1/results --variants baseline --include-transformer --fixed-large-transformer --mode inference --device cpu --batch 8 --length 128 --chunk 32 --decode 16 --threads 32 --repeats 3 --samples 2 --warmup 1 --child-timeout 2700 --total-seconds 10200 --observe-first
