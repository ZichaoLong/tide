#!/bin/bash
set -eo pipefail
source /usr/share/Modules/init/bash
module purge
unset PYTHONPATH
module load ascend/dev-workspace-9.2.0
source /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/activate
export TIDE_SOURCE_COMMIT=28941994880a277495b315f0fec3213239ef0b49
export TIDE_HARNESS_COMMIT=28941994880a277495b315f0fec3213239ef0b49
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TIDE_RESOURCE_IDENTITY='exclusive_cooperative_queue; external jobs not controlled'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/status.json --source 28941994880a277495b315f0fec3213239ef0b49 --max-rss-gib 192.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id agg-inference-hb17b-fp32-v1 --project tide-instances --priority 0 --npu-count 2 --max-wait-seconds 900 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m benchmarks.inference --config configs/scale/hb-17b.json --config-overrides '{"region_batch_size":4,"ssd_scan_backend":"doubling","ssd_summary_backend":"prefix"}' --subjects tide A B C D --devices npu:0,npu:1 --placement region --region-concurrency per_device --prefill 128 --prefill-chunk 32 --decode 16 --samples 2 --warmup 1 --threads 4 --transformer-chunk 256 --allocated-budget-gib 48 --reserved-budget-gib 56 --output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/inference.json --reference-output /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-inference-hb17b-fp32-v1/reference.json
