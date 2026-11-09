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
    a=p.parse_args();a.route='a';plan=json.loads(a.plan.read_text())
    r=StageRunner(a,plan,{'calibration':1740},1740)
    cases=[(4,8),(4,16),(4,32)]
    if a.cpu_only:cases.insert(0,(1,8))
    r.state['purpose']='time to complete controlled depth studies; no learning or exclusive-backend claim'
    r.publish()
    try:
        for threads,batch in cases:
            r.stage(f't{threads}-b{batch}','calibrate','calibration',480,
                ['--kind','tide-dense','--depth','4','--length','32','--batch',str(batch),'--threads',str(threads)],record=False)
        r.finish('completed')
    except BaseException as error:r.finish('failed',repr(error));raise
