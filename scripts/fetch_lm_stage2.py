#!/usr/bin/env python3
"""Wait for first download termination and one finite calibrated main point, then fetch stage 2."""
import argparse
from pathlib import Path
import time
import subprocess
import sys
from tidegraph.lm.assets import fetch
from tidegraph.experiment.records import read_json, write_json, now

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root');p.add_argument('--wait-hours',type=float,default=12)
p.add_argument('--submit-stage2',action='store_true')
a=p.parse_args();root=Path(a.root);deadline=time.monotonic()+a.wait_hours*3600
record=dict(state='waiting',started=now(),condition='stage1 terminal and a completed qwen15 main point with finite logits')
path=root/'stage2-download.json'
write_json(path,record)
while time.monotonic()<deadline:
    first=root/'jobs'/'assets1-a'/'status.json'
    terminal=first.exists() and read_json(first)['state'] in {'passed','failed','cancelled'}
    witnesses=[]
    for result in (root/'runs').glob('*/main-qwen15-*-b1-l512/run/result.json'):
        value=read_json(result)
        if value.get('execution_status')=='completed' and value.get('device_peak',[]) is not None:
            witnesses.append(str(result))
    if terminal and witnesses:
        record.update(state='running',calibration_witnesses=witnesses,results=[]);write_json(path,record);break
    time.sleep(30)
else:
    record.update(state='not_started',reason='bounded calibration wait expired',finished=now());write_json(path,record)
    raise SystemExit(3)
for key in ('qwen3','qwen7','qwenmoe'):
    try:
        value=fetch(key,root/'models',root,project_gib=160,account_gib=300)
        record['results'].append(dict(model=key,state=value['state']))
    except Exception as error:
        record['results'].append(dict(model=key,state='failed',error=str(error)[:1000]))
    write_json(path,record)
if a.submit_stage2 and any(x['state']=='complete' for x in record['results']):
    record['compute_submissions']=[]
    for backend in ('cpu','npu'):
        command=[sys.executable,'-B','scripts/submit_lm_job.py','--unit',f'gef-lm-{backend}-stage2-20261102-a',
                 '--output',str(root/'jobs'/(backend+'-stage2-a')),'--seconds','86400','--memory-gib','384','--',
                 '/bin/bash','scripts/run_lm_local.sh',str(root),'queue','--root',str(root),'--backend',backend,
                 '--phase','second','--npu-helper',str(root/'site'/'run_on_free_npu.py')]
        process=subprocess.run(command,capture_output=True,text=True)
        record['compute_submissions'].append(dict(backend=backend,exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr))
        write_json(path,record)
record.update(state='completed',finished=now());write_json(path,record)
raise SystemExit(0 if all(x['state']=='complete' for x in record['results']) else 1)
