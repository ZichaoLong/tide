#!/usr/bin/env python3
"""Evaluate frozen older weights on v5 heldout documents; no training."""
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
    p.add_argument('--route',choices=['a','b'],required=True);p.add_argument('--cache',type=Path)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--cpu-only',action='store_true')
    a=p.parse_args();plan=json.loads(a.plan.read_text());runner=StageRunner(a,plan,{'final':1740},1740)
    try:
        for item in plan['reference_checkpoints']:
            flags=['--kind',item['kind'],'--depth','1','--length','32','--batch','8']
            if a.cache:flags+=['--cache',str(a.cache)]
            flags+=['--checkpoint',item['checkpoint']] if item.get('checkpoint') else ['--baseline-only']
            path,result=runner.stage('test-'+item['label'],'evaluate','final',600,flags)
            if result['state']!='completed':raise RuntimeError('reference evaluation failed '+item['label'])
        runner.finish()
    except BaseException as error:runner.finish('failed',repr(error));raise
