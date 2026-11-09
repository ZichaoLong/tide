#!/usr/bin/env python3
"""Bounded application-level complete training-step calibration, no quality claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.campaign_runtime import make_model,optimizer_for,update,memory,synthetic_batch
from tide_lab.language import trainable_inventory


def main(a):
    c=json.loads(a.config.read_text());c['device']=a.device;c['threads']=a.threads
    c['training'].update(batch=a.batch,length=a.length)
    if a.native_library:
        c['core'].update(eager_implementation='native',native_library=str(a.native_library))
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();metrics=None
    state=dict(schema='tide-v6-calibration-v1',state='running',started_at=now(),updates=[],
        resolved=c,source=source_identity(),config_sha256=digest(a.config),
        physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        timing_scope='synthetic data; full model/loss/backward/AdamW/synchronization; warmup separate; no speedup qualification')
    atomic_json(a.output/'result.json',state)
    try:
        device=runtime(a.device,a.threads,c['seed']);state['installation']=installed_identity()
        model=make_model(c,device);opt=optimizer_for(model,c)
        state['inventory']=trainable_inventory(model);atomic_json(a.output/'result.json',state)
        for step in range(a.updates):
            b=synthetic_batch(a.batch,a.length,991+step,device)
            row=update(model,opt,b,c,step*a.batch*a.length,device)
            row.update(step=step,warmup=step==0,memory=memory(device))
            state['updates'].append(row);atomic_json(a.output/'result.json',state)
            print(json.dumps(row),flush=True)
        state['state']='completed'
    except BaseException as e:
        state.update(state='failed',error=repr(e));raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start)
        atomic_json(a.output/'result.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','npu:0'],required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--length',type=int,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--updates',type=int,default=3)
    p.add_argument('--native-library',type=Path)
    main(p.parse_args())
