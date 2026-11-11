#!/usr/bin/env python3
"""Predeclared bounded seed22 follow-up; selection is not scientific support."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from scripts.job import inspect
from tide_lab.supervisor import TERMINAL


def main(a):
    root=a.root.resolve();source=Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True,exist_ok=False)
    budget=root/'configs/studies/language/direct-v8/budget.json';deadline=json.loads(budget.read_text())['deadline_epoch']
    state=dict(schema='tide-direct-v8-confirmation-controller-v1',state='waiting_coverage',started_at=now(),
        policy='At most20 seed22 cases: anchor plus each named bucket winner per topology; complete finite 1M validation only; sending_control merges into sending. No new scale/sparsity campaign.',
        dependencies=a.jobs,selection=[],scientific_review='awaiting manual review')
    stopped=[];signal.signal(signal.SIGTERM,lambda s,f:stopped.append(s))
    def publish():atomic_json(a.output/'state.json',state)
    publish()
    try:
        while not stopped and time.time()<deadline-300:
            statuses=[inspect(root/'artifacts/jobs'/name/'job.json') for name in a.jobs]
            state['observed']=[dict(id=o['job_id'],state=o['state'].get('state'),terminated=o['termination_verified_now']) for o in statuses];publish()
            if all(o['state'].get('state') in TERMINAL and o['termination_verified_now'] for o in statuses):break
            time.sleep(15)
        if stopped:raise InterruptedError('controller stopped; no automatic new confirmation')
        if time.time()>=deadline-300:raise TimeoutError('campaign deadline leaves no confirmation budget')
        cases=[]
        for phase in ('anchor','cover'):
            submission=json.loads((root/'runs'/f'v8-{phase}-submission/submission.json').read_text())
            for job in submission['jobs']:
                plan=json.loads(Path(job['plan']).read_text());cases.extend(plan['cases'])
        eligible=[];rejected=[]
        for row in cases:
            out=Path(row['output']);files=[out/'train/result.json',out/'train/validation-final.json',out/'recovery.json',out/'check/result.json']
            if not all(p.exists() for p in files):rejected.append(dict(id=row['id'],reason='missing required evidence'));continue
            result,val,recovery,check=[json.loads(p.read_text()) for p in files]
            if not (result.get('state')=='completed' and result.get('progress',{}).get('tokens')==1048576 and
                    val.get('state')=='completed' and val.get('tokens')==32768 and math.isfinite(val.get('nll',float('nan'))) and
                    recovery.get('state')=='passed' and check.get('admitted_for_finite_exploration')):
                rejected.append(dict(id=row['id'],reason='incomplete/failed execution, validation or qualification'));continue
            c=json.loads(Path(row['config']).read_text());group=c['candidate']['group']
            if group=='sending_control':group='sending'
            eligible.append(dict(row,topology=c['candidate']['recipe']['topology'],bucket=group,nll=val['nll'],
                validation_sha256=digest(files[1])))
        winners=[]
        for topology,bucket in sorted({(r['topology'],r['bucket']) for r in eligible}):
            winners.append(min((r for r in eligible if (r['topology'],r['bucket'])==(topology,bucket)),key=lambda r:(r['nll'],r['id'])))
        if len(winners)>20:raise ValueError('predeclared confirmation cap exceeded')
        cfg=root/'configs/studies/language/direct-v8/confirmation';cfg.mkdir(parents=True,exist_ok=False);ids=[]
        for winner in winners:
            c=json.loads(Path(winner['config']).read_text());c.update(seed=22,id=c['id'][:-3]+'s22',study_id='TL-DIRECT-CONFIRM-v8')
            c['selection_ancestry']=dict(seed11_config_sha256=digest(winner['config']),validation_sha256=winner['validation_sha256'],
                meaning='configuration chosen by validation; fresh initialization, no checkpoint inheritance')
            p=cfg/(c['id']+'.json');atomic_json(p,c);p.chmod(0o444);ids.append(c['id'])
        catalog=cfg/'catalog.json';atomic_json(catalog,dict(schema='tide-direct-v8-confirmation-table-v1',cases=ids));catalog.chmod(0o444)
        state.update(state='submitting',selection=winners,rejected=rejected,eligible=len(eligible),configurations=ids);publish()
        if winners:
            cmd=[sys.executable,str(source/'scripts/submit_direct_v8.py'),'--root',str(root),'--source',str(source),
                '--catalog',str(catalog),'--data',str(root/'runs/v6-data-final'),'--phase','confirm','--group','TL-DIRECT-CONFIRM-v8',
                '--workers','10','--lens-cli',a.lens_cli]
            reply=subprocess.run(cmd,text=True,capture_output=True)
            (a.output/'submission.log').write_text(reply.stdout+'\n'+reply.stderr)
            if reply.returncode:raise RuntimeError('confirmation submission failed; see durable log')
        state.update(state='submitted' if winners else 'no_eligible_candidates',finished_at=now());publish()
    except BaseException as e:
        state.update(state='failed',error=repr(e),finished_at=now());publish();raise
    finally:
        atomic_json(root/'runs/v8-submission-closed.json',dict(closed=True,state=state['state'],at=now(),
            scope='No further automatic submission; existing jobs retain independent lifecycle.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
