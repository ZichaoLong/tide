#!/usr/bin/env python3
"""Read-only observation of repair chains; original failures remain separate."""
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


def review(root=ROOT):
    manifest=read(root/'configs/studies/language/direct-v10/repair.json');rows=[]
    for case in manifest['cases']:
        cid=case['id'];job='v10-repair-'+cid;out=root/'runs'/job
        spec=root/'artifacts/jobs'/job/'job.json';observed=inspect(spec) if spec.is_file() else {}
        worker=read(root/'runs'/(job+'-worker')/'state.json');current=(worker.get('cases') or [{}])[-1]
        checks=read(out/'check/checks.json');pilot=read(out/'pilot/result.json');trained=read(out/'train/result.json')
        stage=(current.get('stages') or [{}])[-1].get('name')
        progress=read(out/'train/progress.json') or read(out/'pilot/progress.json')
        recovery=read(out/'recovery-compare/recovery.json');migration=read(out/'migration/result.json')
        projection=read(out/'train/projection.json') or read(out/'pilot/projection.json')
        original=read(root/'runs'/('v10-main-'+cid)/'train/result.json')
        rows.append(dict(id=cid,job=job,original_run='tl-v10-main-'+cid,
            original_state=original.get('state','unchecked'),original_error=original.get('error'),
            service_state=observed.get('state',{}).get('state','not_submitted'),
            application_state=current.get('state','unchecked'),stage=stage,
            source_commit=read(spec).get('source',{}).get('commit'),
            physical_id=(observed.get('queue') or {}).get('physical_id'),
            slice=observed.get('systemd',{}).get('Slice'),
            descendants_terminated=observed.get('termination_verified_now'),
            strict_operator=read(out/'witness/result.json').get('state','unchecked'),
            strict_forward=checks.get('forward',{}).get('state','unchecked'),
            strict_vjp_failures=sum(v.get('state')=='failed' for v in checks.get('parameter_vjp',{}).values()),
            discrete=read(out/'check/result.json').get('axes',{}).get('discrete_semantics','unchecked'),
            migration_state=migration.get('state','unchecked'),ancestry=migration.get('ancestry'),
            recovery_state=recovery.get('state','unchecked'),recovery_scope=recovery.get('scope'),
            pilot_state=pilot.get('state','unchecked'),pilot_updates=pilot.get('updates'),
            pilot_initial_nll=pilot.get('initial_validation_nll'),pilot_final_nll=pilot.get('final_validation_nll'),
            pilot_max_gradient_norm=pilot.get('max_gradient_norm'),
            gates_passed_at=current.get('gates_passed_at'),tokens=progress.get('tokens'),step=progress.get('step'),
            last_update=progress.get('last_update'),training_state=trained.get('state','unchecked'),
            final_nll=trained.get('final_validation_nll'),confirmation=read(out/'confirmation/result.json'),
            lens_run='tl-'+job,lens_status=projection.get('status','unchecked'),
            lens_readback_matches=projection.get('readback_matches'),
            error=current.get('error'),resources=worker.get('resources')))
    return dict(schema='tide-v10-repair-observation-v1',observed_at=now(),
        deadline_epoch=manifest['deadline_epoch'],cases=rows,
        scientific_review='No new independent seed. Repair continuation does not erase original failures or establish long-term numerical/learning qualification.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',type=Path);a=p.parse_args()
    result=review()
    if a.write:write_review_json(a.write,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
