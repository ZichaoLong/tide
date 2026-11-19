#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=ae247dee82c60b63495e9abea7f703abd84ac3c7
export TIDE_HARNESS_COMMIT=ae247dee82c60b63495e9abea7f703abd84ac3c7
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/status.json --source ae247dee82c60b63495e9abea7f703abd84ac3c7 --max-rss-gib 100.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id training-selected-settle-bf16 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 1500 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.search --config configs/hb-multiscale-x-settle.json --config-overrides '{"region_batch_size":4,"checkpoint_reception":true,"ssd_summary_backend":"prefix"}' --table capacity --batch 16 --length 256 --max-axis-value 32 --max-points 2 --subjects tide --device npu:0 --precision bf16 --mode train --threads 4 --warmup 2 --steps 5 --device-budget-gib 48 --rss-gib 90 --seconds 1400 --child-timeout 600 --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/training-selected-settle-bf16/search
