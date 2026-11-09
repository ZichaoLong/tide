#!/usr/bin/env python3
"""One bounded shallow-control timing, using the ordinary stage and ledger."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.language_driver import StageRunner

p=argparse.ArgumentParser()
for key in ('plan','data','output','asset-path','recording-cli'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--route',default='a');p.add_argument('--deadline-epoch',type=float,required=True)
p.add_argument('--cpu-only',action='store_true');a=p.parse_args()
plan=json.loads(a.plan.read_text());runner=StageRunner(a,plan,{'admission':540},540)
try:
    _,r=runner.stage('calibrate-tide-dense','calibrate','admission',500,
        ['--kind','tide-dense','--depth','1','--batch','8','--length','32'],record=False)
    if r['state']!='completed':raise RuntimeError('calibration failed')
    runner.finish()
except BaseException as e:runner.finish('failed',repr(e));raise
