#!/usr/bin/env python3
"""Finite CPU/NPU batch and CPU-thread comparison before depth training."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from tide_lab.language_driver import StageRunner

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','data','output','asset-path','recording-cli'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--route',choices=['a','b'],default='a')
    p.add_argument('--cache',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    cfg=plan.get('backend_calibration',{})
    if a.route=='b' and a.cache is None:raise ValueError('B calibration requires an immutable prefix cache')
    seconds=cfg.get('compute_seconds',1740)
    r=StageRunner(a,plan,{'calibration':seconds},seconds)
    cases=cfg.get('cases')
    if cases is None:
        cases=[(4,8),(4,16),(4,32)]
        if a.cpu_only:cases.insert(0,(1,8))
    r.state['purpose']='time to complete controlled depth studies; no learning or exclusive-backend claim'
    r.publish()
    try:
        for threads,batch in cases:
            flags=['--kind','tide-dense','--depth',str(cfg.get('depth',4)),'--length','32',
                   '--batch',str(batch),'--threads',str(threads)]
            if a.cache is not None:flags+=['--cache',str(a.cache)]
            r.stage(f't{threads}-b{batch}','calibrate','calibration',cfg.get('case_seconds',480),flags,record=False)
        failed=any(s['state']!='completed' for s in r.state['steps']) or bool(r.state['skipped'])
        r.finish('failed' if failed else 'completed')
        if failed:raise SystemExit(1)
    except BaseException as error:r.finish('failed',repr(error));raise
