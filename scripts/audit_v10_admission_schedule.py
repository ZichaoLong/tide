#!/usr/bin/env python3
"""Separate same-schedule CPU/NPU audit; never mutate frozen training checks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now,digest


def main(a):
    import gc
    import torch
    from tide_lab.train import runtime,source_identity,installed_identity
    from tide_lab.campaign_runtime import make_model
    from scripts.campaign_stage import admission
    dev=runtime(a.device,2,11);a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='tide-v10-schedule-audit-v1',state='running',started_at=now(),
        source=source_identity(),installation=installed_identity(),device=str(dev),cases=[],
        scope='separate aligned-schedule independent checks; original frozen main checks/failures unchanged')
    try:
        for d,r,tag in [(64,28,'q10'),(64,64,'q3'),(128,28,'q10'),(128,64,'q3')]:
            cid=f'hb-d{d}-r{r}-{tag}-s11';path=a.config_root/(cid+'.json');c=json.loads(path.read_text())
            out=a.output/cid;out.mkdir();model=make_model(c,dev)
            value=admission(model,c,dev,out)
            row=dict(id=cid,config_sha256=digest(path),state='completed',**value)
            result['cases'].append(row);atomic_json(a.output/'result.json',result)
            print(json.dumps(dict(id=cid,forward=value['checks']['forward'],
                strict_vjp_failed=sum(v['state']=='failed' for v in value['checks']['parameter_vjp'].values()))),flush=True)
            del model;gc.collect()
            if dev.type=='npu':torch.npu.empty_cache()
        result['state']='completed'
    except BaseException as e:result.update(state='failed',error=repr(e));raise
    finally:result['finished_at']=now();atomic_json(a.output/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','npu:0'),required=True)
    main(p.parse_args())
