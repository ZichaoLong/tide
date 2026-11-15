#!/usr/bin/env python3
"""Read-only MoE status: never write runs/artifacts and never infer quality."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from tide_lab.records import now
from tide_lab.supervisor import TERMINAL
from scripts.review_direct_v10 import application_state


def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def review(root):
    current='moe-controller-r2' if (root/'runs/moe-controller-r2/state.json').is_file() else 'moe-controller'
    state=read(root/'runs'/current/'state.json');rows=[]
    for phase,value in state.get('phases',{}).items():
        for case in value.get('cases',[]):
            path=Path(case['output']);training=path if phase=='confirm' else path/'train'
            result=read(training/'result.json');progress=read(training/'progress.json')
            record=root/'artifacts/jobs'/case['job']/'job.json';obs=inspect(record)
            worker=read(root/'runs'/(case['job']+'-worker')/'state.json')
            inner=next((r for r in worker.get('cases',[]) if r.get('id')==case['id']),{})
            app=application_state(obs['state'].get('state'),result.get('state'),inner.get('state'))
            stages=inner.get('stages',[]);latest=stages[-1] if stages else {}
            projection=read(training/'projection.json');recovery=read(path/'recovery.json');check=read(path/'check/result.json')
            v=read(training/'validation-final.json')
            rows.append(dict(id=case['id'],phase=phase,job=case['job'],application_state=app,
                service_state=obs['state'].get('state'),stage=latest.get('name'),
                physical_id=(obs.get('queue') or {}).get('physical_id'),
                terminal=obs['termination_verified_now'] and obs['state'].get('state') in TERMINAL,
                tokens=progress.get('tokens',result.get('progress',{}).get('tokens')),nll=v.get('nll',result.get('nll')),
                error=result.get('error',inner.get('reason')),recovery=recovery.get('state','unchecked'),
                numerical=check.get('axes',{}).get('strict_numerical','unchecked'),
                last_update=progress.get('last_update',{}),
                lens_status=projection.get('status'),lens_readback_matches=projection.get('readback_matches'),
                config=case.get('config'),record=str(training/'result.json')))
    counts={}
    for row in rows:counts[row['application_state']]=counts.get(row['application_state'],0)+1
    return dict(schema='tide-moe-review-v1',observed_at=now(),controller=state.get('state','not_submitted'),
        controller_job=current,source=state.get('source'),deadline_epoch=state.get('deadline_epoch'),counts=counts,cases=rows,
        original_attempts=state.get('original_attempts',[]),
        selections={k:state.get(k) for k in ('selection8','selection32')},resources=state.get('resources'),
        reused_dense=state.get('reused_dense',[]),scope='read-only execution/metrics; scientific conclusions await review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--json',action='store_true')
    a=p.parse_args();r=review(a.root)
    if a.json:print(json.dumps(r,ensure_ascii=False,indent=2))
    else:
        print(f"controller={r['controller']} source={r['source']} counts={r['counts']}")
        for c in r['cases']:
            exposure=c['tokens']/2**20 if c['tokens'] is not None else None
            print(c['phase'],c['id'],c['application_state'],'stage='+str(c['stage']),'device='+str(c['physical_id']),
                  'Mi='+str(round(exposure,3) if exposure is not None else None),'nll='+str(c['nll']),
                  'recovery='+c['recovery'],'lens='+str(c['lens_status']))
