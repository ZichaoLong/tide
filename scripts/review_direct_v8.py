#!/usr/bin/env python3
"""Read-only v8 lifecycle/evidence review; optional writes only to review paths."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from scripts.job import inspect
from tide_lab.records import digest,now


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def failures(obj):
    if isinstance(obj,dict):return int(obj.get('state')=='failed')+sum(failures(v) for v in obj.values())
    if isinstance(obj,list):return sum(failures(v) for v in obj)
    return 0


def review(root):
    jobs=[];cases=[]
    for path in sorted((root/'artifacts/jobs').glob('v8-*/job.json')):
        obs=inspect(path);record=read(path)
        jobs.append(dict(id=obs['job_id'],state=obs['state'].get('state'),terminated=obs['termination_verified_now'],
            source_commit=record['source']['commit'],unit=record['unit'],slice=obs.get('systemd',{}).get('Slice'),
            device=record['device'],mapping=read(path.parent/'queue-state.json').get('physical_ids'),
            record=str(path.relative_to(root))))
    for submission in sorted((root/'runs').glob('v8-*-submission/submission.json')):
        for job in read(submission).get('jobs',[]):
            for row in read(Path(job['plan'])).get('cases',[]):
                base=Path(row['output']);config=read(Path(row['config']));train=read(base/'train/result.json');check=read(base/'check/result.json')
                recovery=read(base/'recovery.json');val=read(base/'train/validation-final.json');projection=read(base/'train/projection.json')
                manifest=read(base/'train/manifest.json');progress=read(base/'train/progress.json')
                state=train.get('state','pending')
                if not train and check.get('state')=='failed':state='failed_admission'
                if not train and recovery.get('state')=='failed':state='failed_recovery'
                if not train and check.get('state')=='completed' and not recovery:state='qualifying'
                cases.append(dict(id=row['id'],job=job['job'],kind=config['kind'],seed=config['seed'],
                    family=config.get('candidate',{}).get('group'),state=state,
                    tokens=train.get('progress',progress).get('tokens',0),target=config['training']['target_tokens'],
                    validation_nll=val.get('nll'),validation_tokens=val.get('tokens'),
                    validation_state=val.get('state'),recovery=recovery.get('state','unchecked'),
                    strict_failure_count=failures(check.get('checks',{})),error=train.get('error',check.get('error')),
                    axes=train.get('axes',check.get('axes',{})),parameters=train.get('inventory',{}).get('trainable_parameters'),
                    inference=train.get('inference',{}).get('state','unchecked'),
                    source_commit=train.get('identity',check.get('identity',{})).get('source_commit'),
                    config=str(Path(row['config']).relative_to(root)),config_sha256=digest(row['config']),
                    record=str((base/'train/result.json').relative_to(root)),
                    result_sha256=digest(base/'train/result.json') if train else None,
                    run_id=manifest.get('run_id'),group=manifest.get('group',config['study_id']),
                    lens=dict(status=projection.get('status','unchecked'),readback_matches=projection.get('readback_matches'),
                              destination_name=projection.get('destination_name'),destination_run_id=projection.get('destination_run_id'))))
    return dict(schema='tide-direct-v8-review-v1',observed_at=now(),jobs=jobs,cases=cases,
        counts=dict(Counter(c['state'] for c in cases)),resources=read(root/'runs/v8-budget/state.json'),
        confirmation=read(root/'runs/v8-confirmation-controller/state.json'),
        scientific_scope='Lifecycle and measurements only. Scientific support requires explicit review; complete does not mean strict equivalence or real sparse execution.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);p.add_argument('--full',action='store_true');a=p.parse_args()
    r=review(a.root.resolve())
    if a.output:write_review_json(a.output,r)
    print(json.dumps(r if a.full else dict(observed_at=r['observed_at'],counts=r['counts'],
        active_jobs=[j for j in r['jobs'] if not j['terminated']],
        confirmation=r['confirmation'].get('state','not_submitted'),
        resources={k:r['resources'].get('usage',{}).get(k) for k in ('cpu_core_seconds','npu_device_seconds','active_threads')}),ensure_ascii=False,indent=2))
