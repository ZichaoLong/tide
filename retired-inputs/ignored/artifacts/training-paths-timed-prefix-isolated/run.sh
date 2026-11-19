#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a
export TIDE_HARNESS_COMMIT=22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/status.json --source 22527cf472fc4d6ea9ca7ce22528b7e4ca9cc96a --max-rss-gib 64.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id training-paths-timed-prefix-isolated --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 900 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.check_training_paths --config configs/hb-multiscale-x-timed.json --device npu:0 --precision bf16 --candidates baseline prefix --candidate-overrides '{"prefix":{"ssd_summary_backend":"prefix"}}' --batch 1 --length 8 --chunk 4 --steps 2 --threads 1 --witness-elements 12000000 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-paths-timed-prefix-isolated/results
