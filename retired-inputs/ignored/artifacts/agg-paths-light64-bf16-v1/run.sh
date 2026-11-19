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
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/status.json --source 1553b301bf960ec6337fbc38e8e3cada26205eb2 --max-rss-gib 80.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id agg-paths-light64-bf16-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 3000 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -m scripts.check_training_paths --config configs/agg-inference/settle-incoming_light_64.json --device npu:0 --precision bf16 --batch 1 --length 4 --steps 3 --chunk 2 --threads 4 --candidates baseline optimized --candidate-overrides '{"optimized": {"region_batch_size": 4, "checkpoint_reception": true, "ssd_summary_backend": "prefix", "ssd_scan_backend": "doubling", "reception_qkv_backend": "fused", "reception_mask_padding": true}}' --output-dir /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/agg-paths-light64-bf16-v1/paths
