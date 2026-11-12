#!/usr/bin/env python3
"""Recover only the stopped/unstarted original v9 cases, on their frozen trainer."""
import argparse
import copy
import math
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_direct_v9 import Campaign,read
from scripts.job import inspect
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import verify_source


def remaining_cases(original):
    remaining=[]
    for case in original['phases']['main']['cases']:
        result=read(Path(case['output'])/'train/result.json')
        state=result.get('state','not_started')
        if state=='completed':continue
        if state not in ('not_started','budget_truncated'):
            raise ValueError(f'not an orchestration interruption: {case["id"]}: {state}')
        if state=='budget_truncated' and not (Path(case['output'])/'train/latest.pt').is_file():
            raise ValueError(f'no optimizer checkpoint: {case["id"]}')
        remaining.append((case,state))
    return remaining


def outcome(case, qualifications):
    p=Path(case['output']);r=read(p/'train/result.json');v=read(p/'train/validation-final.json')
    progress=read(p/'train/progress.json');target=int(case['train_arguments'][1])
    return dict(id=case['id'],config=case['config'],output=str(p),qualifications=qualifications,
        eligible=(r.get('state')=='completed' and progress.get('tokens')==target
                  and v.get('state')=='completed' and math.isfinite(v.get('nll',float('nan')))),
        execution=r.get('state','not_started'),tokens=progress.get('tokens'),validation_nll=v.get('nll'),
        last_work=progress.get('last_update',{}),validation_work=v.get('work',{}))


class Recovery(Campaign):
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=a.training_source.resolve()
        trainer=verify_source(self.source)
        if trainer['commit']!='6e7bd49eb5e35c15445826cac3a9b6ddda75aff4':
            raise ValueError('recovery requires the original frozen training source')
        self.original=read(a.original/'state.json')
        if self.original.get('state')!='failed' or 'quota' not in self.original.get('error',''):
            raise ValueError('expected retained quota-controller failure')
        for job in self.original['jobs']:
            if not inspect(self.root/'artifacts/jobs'/job/'job.json')['termination_verified_now']:
                raise ValueError(f'original worker still alive: {job}')
        self.cfg=self.source/'configs/studies/language/direct-v9';self.budget=read(self.cfg/'budget.json')
        self.deadline=self.budget['deadline_epoch'];self.stopped=False;self.guard_last=0
        if self.deadline!=self.original['deadline_epoch']:raise ValueError('deadline changed')
        a.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-direct-v9-recovery-v1',state='running',started_at=now(),
            original=str(a.original),original_state_sha256=digest(a.original/'state.json'),
            training_source=trainer['commit'],controller_source=read(Path(__file__).resolve().parents[1]/'.tide-source.json')['commit'],
            deadline_epoch=self.deadline,jobs=[],phases={},scientific_review='awaiting human review')
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
        self.publish()

    def dispatch(self,name,records):
        if not records:return
        if len(records)>12:raise ValueError('original concurrent-worker bound exceeded')
        dest=self.a.output/name;dest.mkdir()
        phase=dict(state='submitting',cases=records,jobs=[],started_at=now())
        self.state['phases'][name]=phase;self.publish()
        for i,case in enumerate(records):
            plan=dest/f'w{i}.json';atomic_json(plan,dict(cases=[case]));plan.chmod(0o444)
            inputs=[plan,Path(case['config']),self.a.data/'data-lock.json']
            if case.get('reuse_admission'):
                inputs += [Path(case['reuse_admission']),Path(case['reuse_recovery'])]
            if '--resume' in case['train_arguments']:
                inputs.append(Path(case['train_arguments'][case['train_arguments'].index('--resume')+1]))
            job=f'v9-recover-{name}-w{i}'
            argv=['python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(self.a.data),
                '--output',str(self.root/'runs'/(job+'-worker')),'--recording-cli',self.a.lens_cli,
                '--deadline-epoch',str(self.deadline)]
            phase['jobs'].append(self.submit(job,argv,inputs));self.publish()
        phase['state']='running';self.publish();self.wait(phase['jobs'])
        phase.update(state='terminal',finished_at=now());self.publish()

    def run(self):
        self.guard();records=[];resolved={};quals={}
        original_cases=self.original['phases']['main']['cases']
        for case in original_cases:
            resolved[case['id']]=case;quals[case['id']]=case['output']
        for old,state in remaining_cases(self.original):
            c=copy.deepcopy(old);ident=c['id'];parent=Path(c['output'])
            c['output']=str(self.root/'runs'/('v9-recover-main-'+ident))
            c['timeout_seconds']=self.deadline-time.time()-180
            args=c['train_arguments'];args[args.index('--run-id')+1]='tl-v9-recover-main-'+ident
            if state=='budget_truncated':
                c.update(reuse_admission=str(parent/'check/result.json'),reuse_recovery=str(parent/'recovery.json'))
                args+=['--resume',str(parent/'train/latest.pt')]
            else:quals[ident]=c['output']
            resolved[ident]=c;records.append(c)
        self.state['resumed_ids']=[c['id'] for c in records if '--resume' in c['train_arguments']]
        self.state['first_started_ids']=[c['id'] for c in records if '--resume' not in c['train_arguments']]
        self.state['resolved_main_cases']=list(resolved.values());self.publish()
        self.dispatch('main',records)
        results={i:outcome(c,quals[i]) for i,c in resolved.items()}
        self.state['main_results']=list(results.values());self.publish()
        extensions=[]
        scale=[f'{name}-s{s}' for name in ('hb88-pool','hb88-quad','tf-hb88') for s in (11,22)]
        if all(results[i]['eligible'] for i in scale):extensions.extend(scale)
        activity=[];decisions=[]
        for width,kind in ((32,'q10'),(64,'q3')):
            ids=[f'r{width}-h2-{k}-s{s}' for k in (kind,'bern','pool') for s in (11,22)]
            band=(.08,.12) if width==32 else (.025,.04)
            # Preserve the original controller's gate, including its last-update denominator.
            fractions=[results[f'r{width}-h2-{kind}-s{s}']['last_work'].get('activity/body_full_all_fraction',1.) for s in (11,22)]
            eligible=all(results[i]['eligible'] for i in ids) and all(band[0]<=f<=band[1] for f in fractions)
            decisions.append(dict(shape=width,band=band,last_update_fractions=fractions,eligible=eligible,
                                  scope='original gate retained; not an independent scientific review'))
            if eligible:activity=ids;break
        extensions+=activity
        self.state['extension_decision']=dict(ids=extensions,activity=decisions,
            remaining_seconds=self.deadline-time.time(),main_complete=sum(r['eligible'] for r in results.values()))
        self.publish();records=[]
        if extensions and self.deadline-time.time()>3600:
            for ident in extensions:
                old=resolved[ident];cfg=read(Path(old['config']));parent=Path(old['output']);qualification=Path(quals[ident])
                records.append(dict(id=ident,config=old['config'],output=str(self.root/'runs'/('v9-recover-extend-'+ident)),
                    max_rss_gib=48,timeout_seconds=self.deadline-time.time()-180,
                    reuse_admission=str(qualification/'check/result.json'),reuse_recovery=str(qualification/'recovery.json'),
                    train_arguments=['--until-tokens',str(cfg['training']['target_tokens']),'--resume',str(parent/'train/latest.pt'),
                        '--run-id','tl-v9-recover-extend-'+ident,'--recording-group',cfg['study_id']]))
            self.dispatch('extend',records)
        self.state.update(state='completed',finished_at=now(),completion_scope='orchestration terminal; individual axes retained')
        self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','data','original','output','training-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);a=p.parse_args();driver=Recovery(a)
    try:driver.run()
    except BaseException as error:
        driver.state.update(state='failed',error=repr(error),finished_at=now());driver.publish();driver.stop_children();raise
    finally:atomic_json(a.output/'submission-closed.json',dict(closed=True,state=driver.state['state'],at=now()))
