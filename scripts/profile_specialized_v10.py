#!/usr/bin/env python3
"""One real-shape NPU operator trace, separate from uninstrumented timing."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now


def main(a):
    import torch
    import torch_npu
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.v10_config import hb
    from tide_lab.campaign_runtime import make_model,optimizer_for,update,synthetic_batch
    device=runtime('npu:0',2,11);a.output.mkdir(parents=True,exist_ok=False)
    c=hb('v10-profile',128,64,.03125,micro=8)
    model=make_model(c,device);opt=optimizer_for(model,c);b=synthetic_batch(8,256,193,device)
    warm=[update(model,opt,b,c,8*2**20,device) for _ in range(2)]
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,profile_memory=False,with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'trace'))) as prof:
        row=update(model,opt,b,c,8*2**20,device);prof.step()
    # TorchNPU exports CSV; its profiler object does not expose Torch's
    # key_averages API. The original failed exporter/trace remains preserved.
    file=next((a.output/'trace').glob('*/ASCEND_PROFILER_OUTPUT/operator_details.csv'))
    with file.open() as f:operations=list(csv.DictReader(f))
    atomic_json(a.output/'result.json',dict(schema='tide-v10-operator-trace-v1',state='completed',at=now(),
        source=source_identity(),installation=installed_identity(),config=c,uninstrumented_warmups=warm,
        instrumented_update=row,operations=operations,scope='one fixed shape trace; instrumented time is not a speed estimate'))
    print(json.dumps([o for o in operations if any(k in o['Name'].lower() for k in ('fusion_attention','select_backward','unbind'))]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
