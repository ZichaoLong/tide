#!/usr/bin/env python3
"""Read-only v6 status; no graph or training record writes."""
import argparse
import collections
import json
from pathlib import Path


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def status(root):
    manifest=read(root/'.local/v6-submission/submission.json');rows=[]
    corrections=read(root/'.local/v6-submission/corrections.json');jobs={j['name']:j for j in manifest.get('jobs',[])}
    for job in corrections.get('jobs',[]):jobs[job['name']]=job
    attempts=collections.defaultdict(list)
    for job in jobs.values():
        lifecycle=read(root/'artifacts/jobs'/job['name']/'state.json')
        worker_path=Path(job.get('worker_output',root/'runs'/job['name']))
        worker=read(worker_path/'state.json');cases={c['id']:c for c in worker.get('cases',[])}
        queue=read(root/'artifacts/jobs'/job['name']/'queue-state.json')
        for cid in job['cases']:
            case=cases.get(cid,{});base=Path(job.get('case_outputs',{}).get(cid,root/'runs'/('v6-'+cid)))
            train=read(base/'train/result.json');progress=read(base/'train/progress.json')
            state=case.get('state','queued' if lifecycle else 'planned')
            if lifecycle.get('state')=='failed' and state in ('queued','running'):state='failed'
            stage=next((s['name'] for s in reversed(case.get('stages',[])) if s['state']=='running'),None)
            attempts[cid].append(dict(id=cid,run_id='tl-v6-'+cid,worker=job['name'],state=state,stage=stage,
                inner_training_state=train.get('state'),step=progress.get('step'),tokens=progress.get('tokens'),
                device=job['device'],physical_id=queue.get('physical_id'),reason=case.get('reason'),
                validation_nll=train.get('final_validation_nll'),record=str((base/'train/result.json').relative_to(root)),
                source_commit=job.get('source_commit',manifest.get('source_identity',{}).get('commit'))))
    for cid,history in attempts.items():
        rows.append(dict(history[-1],previous_attempts=history[:-1]))
    budget=read(root/'runs/v6-budget-v2/state.json')
    return dict(counts=dict(collections.Counter(r['state'] for r in rows)),
        training=sum(r['inner_training_state']=='running' for r in rows),
        checking=sum(r['state']=='running' and r['inner_training_state'] is None for r in rows),
        allocated_cpu_core_hours=budget.get('usage',{}).get('cpu_core_seconds',0)/3600,
        allocated_npu_device_hours=budget.get('usage',{}).get('npu_device_seconds',0)/3600,
        budget_state=budget.get('state'),budget_stop_reasons=budget.get('stop_reasons'),cases=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--brief',action='store_true')
    a=p.parse_args();s=status(a.root)
    if a.brief:s.pop('cases')
    print(json.dumps(s,ensure_ascii=False,indent=2))
