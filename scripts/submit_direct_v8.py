#!/usr/bin/env python3
"""Submit a bounded explicit catalog using existing immutable worker lifecycle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now


def main(a):
    root=a.root.resolve();source=a.source.resolve();cfg=a.catalog.parent.resolve()
    catalog=json.loads(a.catalog.read_text());cases=catalog['cases']
    budget=root/'configs/studies/language/direct-v8/budget.json'
    policy=root/'configs/studies/language/direct-v8/resource.json'
    deadline=json.loads(budget.read_text())['deadline_epoch']
    dest=root/'runs'/('v8-'+a.phase+'-submission');dest.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for index in range(min(a.workers,len(cases))):
        selected=cases[index::a.workers];records=[]
        for cid in selected:
            path=cfg/(cid+'.json');c=json.loads(path.read_text())
            records.append(dict(id=cid,config=str(path),output=str(root/'runs'/('v8-'+a.phase+'-'+cid)),
                max_rss_gib=c['resources']['max_rss_gib'],timeout_seconds=c['training']['timeout_seconds'],
                retire_probe_checkpoints=True,train_arguments=['--run-id','tl-v8-'+a.phase+'-'+cid,'--recording-group',a.group]))
        plan=dest/f'w{index}.json';atomic_json(plan,dict(schema='tide-v6-worker-plan-v1',cases=records));plan.chmod(0o444)
        cap=min(int(deadline-time.time()-30),sum(c['timeout_seconds'] for c in records)+300)
        if cap<180:raise TimeoutError('campaign deadline exhausted')
        device=json.loads(Path(records[0]['config']).read_text())['device']
        if any(json.loads(Path(c['config']).read_text())['device']!=device for c in records):raise ValueError('mixed worker devices')
        job=f'v8-{a.phase}-w{index}'
        cmd=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
            '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
            '--env',f'TIDE_LAB_VENV={root}/.venv-positive-floor-v1',
            '--env',f'TIDE_LAB_RESOURCE_POLICY={policy}','--device','npu' if device.startswith('npu') else 'cpu',
            '--threads','2','--memory-gib','32','--timeout',str(cap)]
        if device.startswith('npu'):cmd+=['--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(min(1800,cap-60))]
        for path in [plan,a.data/'data-lock.json',budget,policy,*[Path(c['config']) for c in records]]:cmd+=['--input',str(path)]
        cmd+=['--','python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(a.data.resolve()),
            '--output',str(root/'runs'/(job+'-worker')),'--recording-cli',a.lens_cli,'--deadline-epoch',str(deadline)]
        reply=subprocess.run(cmd,capture_output=True,text=True)
        if reply.returncode:raise RuntimeError(reply.stderr+' '+reply.stdout)
        jobs.append(dict(job=job,cases=selected,plan=str(plan)))
        atomic_json(dest/'submission.json',dict(state='submitting',jobs=jobs,updated_at=now()))
        print(json.dumps(jobs[-1]),flush=True)
    atomic_json(dest/'submission.json',dict(state='submitted',jobs=jobs,updated_at=now(),
        scientific_review='unchecked; inspect inner stages and actual services'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source','catalog','data'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('phase','group','lens-cli'):p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=8);main(p.parse_args())
