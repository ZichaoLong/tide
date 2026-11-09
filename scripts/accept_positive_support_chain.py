#!/usr/bin/env python3
"""Finite acceptance orchestration; independent FC evaluation survives HB failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json, now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','foundation'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--old-python',type=Path);p.add_argument('--recording-cli')
    p.add_argument('--device',choices=['cpu','npu:0'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cfg=json.loads(a.config.read_text())
    deadline=min(cfg['deadline_epoch'],time.time()+(2700 if a.device=='cpu' else 600))
    state=dict(schema='tide-language-chain-v1',state='running',started_at=now(),steps=[],device=a.device,
        deadline_epoch=deadline,config=str(a.config),study_id='TL-LM-EXEC')
    def publish():atomic_json(a.output/'stage.json',state)
    def phase(name,limit,python=sys.executable,extra=()):
        limit=min(limit,deadline-time.time()-15)
        if limit<5:
            state['steps'].append(dict(name=name,state='skipped',reason='budget'));publish();return False
        dest=a.output/name;row=dict(name=name,state='running',output=str(dest),started_at=now(),limit_seconds=limit)
        state['steps'].append(row);publish();start=time.monotonic()
        command=[python,'scripts/accept_positive_support.py',name,'--config',str(a.config),'--output',str(dest),
            '--device',a.device,'--foundation',str(a.foundation),*map(str,extra)]
        wrapper=[sys.executable,'scripts/bounded_workload.py','--max-seconds',str(limit),
            '--deadline-epoch',str(deadline),'--output',str(a.output/(name+'-wrapper')),'--',*command]
        with (a.output/(name+'.log')).open('x') as log:
            proc=subprocess.run(wrapper,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else dict(state='failed',error='missing inner result')
        ok=proc.returncode==0 and result['state']=='completed'
        row.update(state='completed' if ok else 'failed',exit_code=proc.returncode,elapsed_seconds=time.monotonic()-start,finished_at=now(),error=result.get('error'))
        publish();print(json.dumps(row),flush=True);return ok
    publish()
    if phase('foundation',300):
        if phase('task',300) and a.device=='cpu':
            if phase('export',60,python=str(a.old_python)):
                extra=['--material',a.output/'export','--recording-cli',a.recording_cli]
                if phase('hb',900,extra=extra):
                    phase('resume',300,extra=[*extra,'--prior',a.output/'hb'])
                phase('fc',600,extra=extra)
    failures=[r for r in state['steps'] if r['state']!='completed']
    state.update(state='failed' if failures else 'completed',finished_at=now(),allocated_elapsed_seconds=time.monotonic()-started)
    publish();return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
