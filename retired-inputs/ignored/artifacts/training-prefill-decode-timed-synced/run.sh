#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=f8235fe48ff794f0e5ac60bf7565d7f918479e6c
export TIDE_HARNESS_COMMIT=f8235fe48ff794f0e5ac60bf7565d7f918479e6c
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/status.json --source f8235fe48ff794f0e5ac60bf7565d7f918479e6c --max-rss-gib 80.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id training-prefill-decode-timed-synced --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 900 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.programme --suite continuous --configs /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-timed-fp32/search/config.json --device npu:0 --precision fp32 --batch 1 --length 128 --prefill-chunk 128 --decode 8 --steps 3 --warmup 1 --threads 4 --modes inference --continuous-joint A B C D --child-timeout 300 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-prefill-decode-timed-synced/results
