#!/usr/bin/env python3
"""Read-only v10 handoff; execution states never create scientific support."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.review_io import write_review_json
from tide_lab.records import now


ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def application_state(service_state,result_state,case_state=None):
    """A worker can exit normally after recording an inner training failure."""
    if result_state in ('failed','budget_truncated'):return result_state
    if case_state=='failed':return 'failed'
    if service_state=='completed':
        return 'completed' if result_state=='completed' else 'unchecked'
    return service_state


def review():
    controller=read(ROOT/'runs/v10-controller/state.json');rows=[]
    for phase,record in controller.get('phases',{}).items():
        for c in record.get('cases',[]):
            job=c['job'];output=Path(c['output']);spec=ROOT/'artifacts/jobs'/job/'job.json'
            observed=inspect(spec);state=observed['state'];queue=observed.get('queue') or {}
            training=read(output/'train/progress.json');worker=read(ROOT/'runs'/(job+'-worker')/'state.json')
            case=(worker.get('cases') or [{}])[-1];stages=case.get('stages',[])
            result=read(output/('result.json' if phase.startswith('confirm') else 'train/result.json'))
            projection=read(output/('projection.json' if phase.startswith('confirm') else 'train/projection.json'))
            checks=read(output/'check/checks.json');strict=[v.get('state') for v in checks.get('parameter_vjp',{}).values()]
            rows.append(dict(id=c['id'],job=job,phase=phase,
                execution=application_state(state.get('state','unchecked'),result.get('state'),case.get('state')),
                service_state=state.get('state','unchecked'),
                queue=queue.get('status'),physical_id=queue.get('physical_id'),
                stage=stages[-1]['name'] if stages else None,tokens=training.get('tokens'),step=training.get('step'),
                last_update=training.get('last_update'),result_state=result.get('state'),error=result.get('error'),
                recovery=read(output/'recovery.json').get('state','reused_or_unchecked'),
                strict_forward=checks.get('forward',{}).get('state','unchecked'),strict_vjp_failures=strict.count('failed'),
                lens_run='tl-'+job,lens_status=projection.get('status','not_yet_recorded'),
                source_commit=read(spec).get('source',{}).get('commit'),
                descendants_terminated=observed['termination_verified_now']))
    supplemental=[]
    for name in ('v10-admission-aligned',):
        spec=ROOT/'artifacts/jobs'/name/'job.json'
        if spec.is_file():
            observed=inspect(spec);result=read(ROOT/'runs'/name/'result.json')
            supplemental.append(dict(job=name,execution=observed['state'].get('state'),
                physical_id=(observed.get('queue') or {}).get('physical_id'),result_state=result.get('state','unchecked'),
                source_commit=read(spec).get('source',{}).get('commit'),
                cases=[dict(id=c['id'],state=c['state'],forward=c['checks']['forward'],
                    strict_vjp_failures=sum(v['state']=='failed' for v in c['checks'].get('parameter_vjp',{}).values()))
                    for c in result.get('cases',[])],descendants_terminated=observed['termination_verified_now']))
    return dict(schema='tide-v10-observation-v1',at=now(),controller_state=controller.get('state','not_submitted'),
        source_commit=controller.get('source'),deadline_epoch=controller.get('deadline_epoch'),
        resources=controller.get('resources'),jobs=rows,supplemental_checks=supplemental,
        scientific_review='awaiting; strict numerical, discrete mechanisms, recovery and learning remain separate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',type=Path);a=p.parse_args();value=review()
    if a.write:write_review_json(a.write,value)
    summary=dict(value,jobs=[{k:v for k,v in r.items() if k!='last_update'} for r in value['jobs']])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
