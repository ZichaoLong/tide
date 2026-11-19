#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=6570d0610e389c42f548c0afffa281b536df01db
export TIDE_HARNESS_COMMIT=6570d0610e389c42f548c0afffa281b536df01db
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/status.json --source 6570d0610e389c42f548c0afffa281b536df01db --max-rss-gib 96.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id agg-train-half-fp32-100-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 3000 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.continuous --config configs/scale/hb-0p5b.json --config-overrides '{"region_batch_size": 4, "checkpoint_reception": true, "ssd_summary_backend": "prefix", "ssd_scan_backend": "doubling"}' --device npu:0 --precision fp32 --mode train --batch 4 --prefill 128 --prefill-chunk 128 --decode 0 --updates 100 --warmup 2 --threads 4 --device-budget-gib 48 --checkpoint /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/state.pt --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-train-half-fp32-100-v1/continuous.json
