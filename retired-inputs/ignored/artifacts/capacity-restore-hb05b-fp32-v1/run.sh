#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=062a8ab6d6ba0faa191d8900420b697711b7d21d
export TIDE_HARNESS_COMMIT=062a8ab6d6ba0faa191d8900420b697711b7d21d
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/status.json --source 062a8ab6d6ba0faa191d8900420b697711b7d21d --max-rss-gib 96.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id capacity-restore-hb05b-fp32-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.restore --checkpoint /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-continuous-hb05b-v3/results/002-hb-0p5b-fp32-tide-train.pt --checkpoint-source efca7198570e8ee7a21ddccd81165eb36bdb269a --device npu:0 --precision fp32 --batch 1 --length 128 --prefill-chunk 64 --decode 8 --threads 4 --device-budget-gib 52 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/capacity-restore-hb05b-fp32-v1/restore.json
