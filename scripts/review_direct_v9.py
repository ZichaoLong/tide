#!/usr/bin/env python3
"""Read-only current v9 observations; optional publication outside run records."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now,digest


def read(p):return json.loads(p.read_text()) if p.exists() else {}


def review(root):
    jobs=[]
    for p in sorted((root/'artifacts/jobs').glob('v9-*/job.json')):
        x=inspect(p);spec=read(p)
        jobs.append(dict(id=x['job_id'],state=x['state'].get('state'),unit=spec['unit'],
            physical_id=(x.get('queue') or {}).get('physical_id'),queue_status=(x.get('queue') or {}).get('status'),
            main_pid=x['systemd'].get('MainPID') if x['systemd'] else None,
            termination_verified=x['termination_verified_now'],source=spec['source']['commit'],
            config_inputs=spec['inputs'],log=str(p.parent/'job.log')))
    stages=[]
    for p in sorted((root/'runs').glob('v9-*/train/result.json')):
        r=read(p);progress=read(p.parent/'progress.json');projection=read(p.parent/'projection.json')
        stages.append(dict(id=r['config_id'],run_id=read(p.parent/'manifest.json').get('run_id'),
            execution=r['state'],axes=r.get('axes'),tokens=progress.get('tokens'),
            last_metrics=progress.get('last_update'),final_validation_nll=r.get('final_validation_nll'),
            error=r.get('error'),record=str(p.relative_to(root)),result_sha256=digest(p),
            lens_name=projection.get('destination_name'),lens_status=projection.get('status'),
            lens_readback_matches=projection.get('readback_matches')))
    return dict(schema='tide-direct-v9-observation-v1',observed_at=now(),
        campaign=read(root/'runs/v9-campaign/state.json'),jobs=jobs,training_stages=stages,
        scientific_review='awaiting; execution/numerical/discrete/recovery/learning remain separate',
        data=read(root/'runs/v9-data-r3/state.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--write',type=Path);a=p.parse_args();r=review(a.root)
    if a.write:write_review_json(a.write,r)
    print(json.dumps(dict(observed_at=r['observed_at'],campaign=r['campaign'].get('state'),data=r['data'],
        jobs=[{k:j[k] for k in ('id','state','physical_id','queue_status','termination_verified')} for j in r['jobs']],
        stages=[{k:s[k] for k in ('id','execution','tokens','final_validation_nll','error')} for s in r['training_stages']]),ensure_ascii=False,indent=2))
