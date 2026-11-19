#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=1553b301bf960ec6337fbc38e8e3cada26205eb2
export TIDE_HARNESS_COMMIT=1553b301bf960ec6337fbc38e8e3cada26205eb2
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/status.json --source 1553b301bf960ec6337fbc38e8e3cada26205eb2 --max-rss-gib 80.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id agg-inference-timed-long-fp32-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 3000 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.inference --config configs/agg-inference/timed-incoming_light_64-optimized.json --subjects tide A B C D --devices npu:0 --precision fp32 --batch 4 --prefill 2048 --prefill-chunk 128 --decode 256 --stream-positions 512 --stream-chunk 128 --samples 2 --warmup 1 --no-cold --transformer-chunk 256 --threads 4 --allocated-budget-gib 48 --reserved-budget-gib 56 --reference-output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/reference.json --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-timed-long-fp32-v1/inference.json
