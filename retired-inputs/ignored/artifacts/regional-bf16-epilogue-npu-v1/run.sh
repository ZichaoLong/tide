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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/source/src"${PYTHONPATH:+:$PYTHONPATH}"
/mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import importlib.util, os, sys
from scripts.job import atomic
names = ('"'"'PYTHONPATH'"'"', '"'"'PATH'"'"', '"'"'LD_LIBRARY_PATH'"'"', '"'"'ASCEND_HOME_PATH'"'"', '"'"'ASCEND_OPP_PATH'"'"', '"'"'LOADEDMODULES'"'"', '"'"'VIRTUAL_ENV'"'"', '"'"'TORCH_DEVICE_BACKEND_AUTOLOAD'"'"')
modules = {name: importlib.util.find_spec(name) for name in ('"'"'tide'"'"', '"'"'tbe'"'"')}
atomic(sys.argv[1], dict(executable=sys.executable, sys_path=sys.path, environment={name: os.environ.get(name) for name in names}, module_origins={name: None if spec is None else spec.origin for name, spec in modules.items()}))' /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/environment.json
exec /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/source/scripts/job.py --status /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/status.json --source 062a8ab6d6ba0faa191d8900420b697711b7d21d --max-rss-gib 8.0 -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python /mi/data2T/zlong/home/.agents/skills/use-local-ascend/scripts/run_on_free_npu.py --job-id regional-bf16-epilogue-npu-v1 --project tide-instances --priority 0 --npu-count 1 --max-wait-seconds 600 --state-file /mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/queue.json -- /mi/data2T/zlong/home/llm/graph-execution-foundation/.venv-instances/bin/python -c 'import os,torch,json
import torch_npu
from tide.precision import precision_context
from scripts.job import atomic
torch.npu.set_device("npu:0"); torch.npu.matmul.allow_hf32=False; torch.set_num_threads(1)
x=torch.zeros(2,6,32,device="npu:0"); w=torch.zeros(2,128,32,device="npu:0"); b=torch.full((2,128),1.001,device="npu:0")
with precision_context("npu:0","bf16"):
 serial=torch.stack([torch.nn.functional.linear(x[i],w[i],b[i]) for i in range(2)])
 batched=torch.baddbmm(b[:,None],x,w.transpose(1,2))
 separate=torch.bmm(x,w.transpose(1,2))+b[:,None]
torch.npu.synchronize()
record=dict(source=os.environ["TIDE_SOURCE_COMMIT"],physical_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
serial_dtype=str(serial.dtype),baddbmm_dtype=str(batched.dtype),separate_dtype=str(separate.dtype),
serial_value=float(serial.flatten()[0]),baddbmm_value=float(batched.flatten()[0]),separate_value=float(separate.flatten()[0]),
max_abs=float((serial-batched).abs().max()))
atomic("/mi/data2T/zlong/home/llm/graph-execution-foundation/artifacts/regional-bf16-epilogue-npu-v1/probe.json",record)
print(json.dumps(record)); assert record["serial_dtype"]==record["baddbmm_dtype"]=="torch.bfloat16" and record["max_abs"]==0
'
