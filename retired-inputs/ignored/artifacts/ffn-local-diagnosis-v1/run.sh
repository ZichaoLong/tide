#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=a58536608f7276a771c56703b48e23ea82d2ca65
export TIDE_HARNESS_COMMIT=a58536608f7276a771c56703b48e23ea82d2ca65
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/status.json --source a58536608f7276a771c56703b48e23ea82d2ca65 --max-rss-gib 16.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/source/scripts/wait_units.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/dependency.json --seconds 2400 --unit tide-ffn-medium-npu-timing-v1.service -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id ffn-local-diagnosis-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/queue.json --candidate-ids 9 -- timeout --signal=TERM --kill-after=30 300 python -m scripts.diagnose_grouped_ffn --qualification /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-npu-v1/qualification.json --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/ffn-local-diagnosis-v1/diagnosis.json --device npu:0 --threads 4
