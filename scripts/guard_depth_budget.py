#!/usr/bin/env python3
"""Enforce one shared actual-compute budget across named v5 attempts only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from scripts.job import inspect


def usage(root,names,at):
    rows=[]
    for name in names:
        path=root/'runs'/name/'stage.json';job=root/'artifacts/jobs'/name
        life=json.loads((job/'state.json').read_text()) if (job/'state.json').exists() else {}
        effective_at=min(at,datetime.fromisoformat(life['ended_at']).timestamp()) if life.get('ended_at') else at
        used=0.;phase='not_submitted'
        if path.exists():
            r=json.loads(path.read_text());phase=r['state']
            used=r.get('allocated_elapsed_seconds',max(0,effective_at-datetime.fromisoformat(r['started_at']).timestamp()))
        elif (job/'worker-state.json').exists():
            r=json.loads((job/'worker-state.json').read_text());phase=r['state']
            # Data preparation has no StageRunner. Dependency/resource waits do
            # not allocate a compute worker and are recorded separately.
            if name=='lm-depth-data-v5':used=r.get('elapsed_seconds',max(0,effective_at-datetime.fromisoformat(r['started_at']).timestamp()))
            else:phase='waiting_or_starting'
        if (job/'state.json').exists():
            life=json.loads((job/'state.json').read_text())
            if life['state'] in ['completed','failed','stopped','timed_out']:phase=life['state']
        rows.append(dict(job_id=name,compute_seconds=used,phase=phase))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    while True:
        rows=usage(a.root,cfg['jobs'],time.time());total=sum(r['compute_seconds'] for r in rows)+cfg.get('prior_compute_seconds',0)
        state=dict(schema='tide-depth-budget-guard-v1',state='monitoring',at=now(),rows=rows,used_compute_seconds=total,
            limit_compute_seconds=cfg['compute_seconds'],deadline_epoch=cfg['deadline_epoch'],
            scope='v5 data/calibration/study/reference compute; waits separate; v4 excluded; stop before cap with shutdown reserve')
        if total>=cfg['compute_seconds']-120 or time.time()>=cfg['deadline_epoch']-30:
            state['state']='budget_stop';state['stops']=[]
            for name in cfg['jobs']:
                job=a.root/'artifacts/jobs'/name/'job.json'
                if not job.exists():continue
                spec=json.loads(job.read_text());subprocess.run(['systemctl','--user','stop',spec['unit']],capture_output=True)
                state['stops'].append(inspect(job))
            atomic_json(a.output/'state.json',state);break
        if all(r['phase'] in ['completed','failed','stopped','timed_out'] for r in rows):
            state['state']='completed';atomic_json(a.output/'state.json',state);break
        atomic_json(a.output/'state.json',state);time.sleep(10)
