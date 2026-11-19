#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=46317a8dd73d0611ff0f7b18f4057e97c5baa630
export TIDE_HARNESS_COMMIT=46317a8dd73d0611ff0f7b18f4057e97c5baa630
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TIDE_RESOURCE_IDENTITY='pinned_cpu; shared host'
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-infer-cpu-stages-baseline-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-infer-cpu-stages-baseline-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-infer-cpu-stages-baseline-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-infer-cpu-stages-baseline-v1/status.json --source 46317a8dd73d0611ff0f7b18f4057e97c5baa630 --max-rss-gib 128.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.inference_breakdown --subject tide --config configs/memory-candidates/9b-kv128.json --device cpu --batch 8 --prefill 128 --prefill-chunk 32 --decode 16 --warmup 1 --samples 1 --threads 32 --allocated-budget-gib 56 --reserved-budget-gib 56 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-infer-cpu-stages-baseline-v1/breakdown.json
