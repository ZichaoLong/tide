#!/usr/bin/env python3
"""Read-only live v12 state; never creates or modifies workload records."""
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now
from tide_lab.quota import account_quota

def read(p):return json.loads(p.read_text()) if p.is_file() else {}

def review(root):
    base=root/'configs/studies/language/hb-v12';table=read(base/'table.json');rows=[]
    for case in table['cases']:
        job='v12-main-'+case['id'];out=root/'runs'/job;record=root/'artifacts/jobs'/job/'job.json'
        observed=inspect(record) if record.exists() else {}
        worker=read(root/'runs'/(job+'-worker')/'state.json');stages=worker.get('cases',[{}])[-1].get('stages',[]) if worker.get('cases') else []
        stage=stages[-1]['name'] if stages else None
        r=read(out/stage/'result.json') if stage else {};progress=read(out/stage/'progress.json') if stage else {}
        train=read(out/'train/result.json');validation=read(out/'train/validation-final.json');confirm=read(out/'confirmation/result.json')
        rows.append(dict(id=case['id'],group=case['study_id'],parameters=case['parameters'],
            lifecycle=observed.get('state',{}).get('state','not_submitted'),worker=worker.get('state','not_started'),
            stage=stage,stage_state=r.get('state'),tokens=progress.get('tokens',r.get('progress',{}).get('tokens')),
            physical_device=observed.get('queue',{}).get('physical_id') if observed.get('queue') else None,
            unit=observed.get('systemd',{}),termination_verified=observed.get('termination_verified_now'),
            source_commit=read(record).get('source',{}).get('commit'),error=r.get('error',worker.get('error')),
            endpoint_reached=train.get('state')=='completed' and train.get('progress',{}).get('tokens')==128*2**20,
            validation_nll=validation.get('nll'),confirmation_state=confirm.get('state','unchecked'),
            confirmation_nll=confirm.get('nll'),science='awaiting scoped human review',
            run_id='tl-v12-main-'+case['id'],record=str((out/'train/result.json').relative_to(root))))
    return dict(schema='tide-hb-v12-live-review-v1',observed_at=now(),budget=read(base/'budget.json'),quota=account_quota(),
        training=rows,preparation=[dict(job=p.parent.name,**{k:v for k,v in read(p).items() if k in ('state','error','selected','reproduced','centered_solver_check')})
            for p in sorted((root/'runs').glob('v12-*/result.json'))],
        historical_controls=read(root/'runs/v12-controls/state.json'),
        scope='live read-only metadata; completed/failed/strict numeric/recovery/learning are separate; no scientific support from exit codes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path);a=p.parse_args()
    r=review(a.root.resolve())
    if a.output:write_review_json(a.output,r)
    else:print(json.dumps(r,indent=2,ensure_ascii=False))
