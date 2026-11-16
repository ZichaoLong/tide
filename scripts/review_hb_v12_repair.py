#!/usr/bin/env python3
"""Read-only repair lifecycle; original failures never become successes here."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now


def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def review(root):
    control=read(root/'runs/v12-repair-controller/state.json');rows=[]
    for name,item in control.get('jobs',{}).items():
        job=item['job'];obs=inspect(root/'artifacts/jobs'/job/'job.json');worker=read(root/'runs'/(job+'-worker')/'state.json')
        case=worker.get('cases',[{}])[-1] if worker.get('cases') else {};stages=case.get('stages',[]);stage=stages[-1]['name'] if stages else None
        path=root/'runs'/job;r=read(path/stage/'result.json') if stage else {};progress=read(path/stage/'progress.json') if stage else {}
        train=read(path/'train/result.json');confirm=read(path/'confirmation/result.json')
        rows.append(dict(id=name,job=job,lifecycle=obs['state']['state'],stage=stage,stage_state=r.get('state'),
            physical_device=obs.get('queue',{}).get('physical_id'),termination_verified=obs.get('termination_verified_now'),
            tokens=train.get('progress',{}).get('tokens') if train.get('state')=='completed' else progress.get('tokens',r.get('progress',{}).get('tokens')),
            recovered_parent_progress=case.get('recovered_parent_progress'),prefix_progress=case.get('prefix_progress'),
            parent_checkpoint=item['parent_checkpoint'],parent_sha256=item['parent_sha256'],
            validation_nll=train.get('final_validation_nll'),confirmation_nll=confirm.get('nll'),
            error=r.get('error',worker.get('error')),science='awaiting scoped review'))
    checks=[]
    for dev in ('cpu','npu'):
        p=root/'runs'/('v12-repair-qualification-'+dev);r=read(p/'result.json')
        checks.append(dict(device=dev,state=r.get('state','not_started'),error=r.get('error'),allocator_time_ratio=r.get('allocator_time_ratio'),
            lens=read(p/'projection.json').get('status'),record=str((p/'result.json').relative_to(root))))
    return dict(schema='tide-hb-v12-repair-live-review-v1',observed_at=now(),controller=control,qualification=checks,repairs=rows,
        scope='Original failures remain in review_hb_v12.py; new source migration is not a new seed or old-source exact continuation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path);a=p.parse_args()
    r=review(a.root.resolve())
    if a.output:write_review_json(a.output,r)
    else:print(json.dumps(r,indent=2,ensure_ascii=False))
