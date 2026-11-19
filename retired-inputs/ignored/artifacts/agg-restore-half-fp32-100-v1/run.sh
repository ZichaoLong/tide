#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=1fbe15c9b0ce900bfffbf5c148ca61c81cb34046
export TIDE_HARNESS_COMMIT=1fbe15c9b0ce900bfffbf5c148ca61c81cb34046
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/status.json --source 1fbe15c9b0ce900bfffbf5c148ca61c81cb34046 --max-rss-gib 96.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id agg-restore-half-fp32-100-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 4500 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.restore --checkpoint /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/state.pt --checkpoint-source 6570d0610e389c42f548c0afffa281b536df01db --expected-checkpoint-sha256 b98db421399ac3976a0f02b21780581fa526d18d35e1b64fc990691259da1c89 --device npu:0 --precision fp32 --batch 4 --length 128 --threads 4 --device-budget-gib 48 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-restore-half-fp32-100-v1/restore.json
