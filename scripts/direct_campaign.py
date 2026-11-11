#!/usr/bin/env python3
"""Fixed v7 selection/progression policy over the existing durable job launcher."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from scripts.prepare_direct_v7 import config
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import TERMINAL


class Campaign:
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=Path(__file__).resolve().parents[1]
        self.cfg=self.source/'configs/studies/language/direct-v7'
        self.catalog=json.loads((self.cfg/'catalog.json').read_text())
        self.deadline=json.loads((self.cfg/'budget.json').read_text())['deadline_epoch']
        a.output.mkdir(parents=True,exist_ok=False);self.stopped=False
        self.state=dict(schema='tide-direct-v7-campaign-v1',state='running',started_at=now(),
            source=json.loads((self.source/'.tide-source.json').read_text())['commit'],
            catalog_sha256=digest(self.cfg/'catalog.json'),phases={},scientific_review='awaiting human review')
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
        self.publish()

    def stop(self,sig,frame):self.stopped=True
    def publish(self):atomic_json(self.a.output/'state.json',self.state)

    def phase(self,name,entries,group,until):
        jobs=[];records=[]
        for entry in entries:
            path=Path(entry['config']);c=json.loads(path.read_text())
            out=self.root/'runs'/('v7-'+name+'-'+c['id'])
            row=dict(id=c['id'],config=str(path),output=str(out),max_rss_gib=24,
                timeout_seconds=900 if name=='screen' else 3600,retire_probe_checkpoints=True,
                train_arguments=['--until-tokens',str(until),'--run-id','tl-v7-'+name+'-'+c['id'],
                                 '--recording-group',group])
            if entry.get('parent'):
                parent=Path(entry['parent'])
                row.update(reuse_admission=str(parent/'check/result.json'),reuse_recovery=str(parent/'recovery.json'))
                row['train_arguments'] += ['--resume',str(parent/'train/tokens-262144.pt')]
            records.append(row)
        phase=dict(state='submitting',cases=records,jobs=jobs,started_at=now());self.state['phases'][name]=phase;self.publish()
        for index in range(min(4,len(records))):
            cases=records[index::4];job=f'v7-{name}-w{index}'
            plan=self.a.output/(job+'-plan.json');atomic_json(plan,dict(schema='tide-v6-worker-plan-v1',cases=cases));plan.chmod(0o444)
            cap=min(int(self.deadline-time.time()-30),sum(c['timeout_seconds'] for c in cases)+180)
            if self.stopped or cap<180:raise TimeoutError('campaign stopped or deadline exhausted')
            cmd=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',job,'--source',str(self.source),
                '--output',str(self.root/'artifacts/jobs'/job),'--env-script',str(self.source/'scripts/env.sh'),
                '--env',f'TIDE_LAB_VENV={self.root}/.venv-positive-floor-v1',
                '--env',f'TIDE_LAB_RESOURCE_POLICY={self.cfg}/resource.json',
                '--device','npu','--threads','2','--memory-gib','28','--timeout',str(cap),
                '--queue-helper',str(self.source/'scripts/run_on_light_device.py'),
                '--wait-seconds',str(min(1200,cap-60))]
            inputs=[plan,self.a.data/'data-lock.json',self.cfg/'budget.json',self.cfg/'resource.json',
                    *[Path(c['config']) for c in cases]]
            for c in cases:
                if c.get('reuse_admission'):
                    inputs += [Path(c['reuse_admission']),Path(c['reuse_recovery']),
                               Path(c['train_arguments'][-1])]
            for p in inputs:cmd+=['--input',str(p)]
            cmd += ['--','python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(self.a.data),
                '--output',str(self.root/'runs'/(job+'-worker')),'--recording-cli',self.a.lens_cli,'--deadline-epoch',str(self.deadline)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr+' '+result.stdout)
            jobs.append(job);self.publish();print(json.dumps(dict(submitted=job,cases=[c['id'] for c in cases])),flush=True)
        phase['state']='running';self.publish()
        while not self.stopped and time.time()<self.deadline:
            observations=[inspect(self.root/'artifacts/jobs'/j/'job.json') for j in jobs]
            if all(o['termination_verified_now'] and o['state'].get('state') in TERMINAL for o in observations):break
            phase['jobs_observed']=[dict(id=o['job_id'],state=o['state']['state']) for o in observations]
            self.publish();time.sleep(10)
        if self.stopped:raise InterruptedError('controller stopped; individual durable jobs retain their own lifecycle')
        if time.time()>=self.deadline:raise TimeoutError('campaign deadline')
        reviewed=[]
        for c in records:
            path=Path(c['output'])/'train/result.json';p=Path(c['output'])/'train/projection.json'
            r=json.loads(path.read_text()) if path.exists() else {}
            projection=json.loads(p.read_text()) if p.exists() else {}
            vpath=path.parent/'validation-final.json';v=json.loads(vpath.read_text()) if vpath.exists() else {}
            ok=(r.get('state')=='completed' and r.get('progress',{}).get('tokens')==until
                and v.get('state')=='completed' and v.get('tokens')==32768 and math.isfinite(v.get('nll',float('nan'))))
            reviewed.append(dict(id=c['id'],config=c['config'],output=c['output'],eligible=ok,
                execution=r.get('state','not_started'),tokens=r.get('progress',{}).get('tokens'),
                validation_nll=v.get('nll'),lens_verified=projection.get('status')=='verified' and projection.get('readback_matches') is True,
                result_sha256=digest(path) if path.exists() else None))
        phase.update(state='terminal',results=reviewed,finished_at=now(),
            eligible_cases=sum(r['eligible'] for r in reviewed),
            failed_or_incomplete_cases=sum(not r['eligible'] for r in reviewed));self.publish()
        return reviewed

    def run(self):
        screen=self.phase('screen',[dict(config=str(self.cfg/(i+'.json'))) for i in self.catalog['screen']],
                          'TL-DIRECT-RECIPE-v7',262144)
        if not any(r['eligible'] and r['lens_verified'] for r in screen):raise RuntimeError('no admitted learning/verified Lens record')
        by={r['id']:r for r in screen};scores=[]
        for init in ('std015','fan_in'):
            for norm in (False,True):
                ids=[config('direct_graph',m,initialization=init,norm=norm)['id'] for m in ('hb40','hb88')]
                if all(by[i]['eligible'] for i in ids):scores.append(dict(initialization=init,norm=norm,
                    nll=sum(by[i]['validation_nll'] for i in ids)/2,ids=ids))
        best=min(scores,key=lambda x:x['nll']) if scores else None
        preferred=next((s for s in scores if s['initialization']=='fan_in' and s['norm']),None)
        if preferred and preferred['nll']<=best['nll']+.02:best=preferred
        tfs=[]
        for pos in ('original','balanced'):
            ids=[f'{m}-{pos}-s11' for m in ('tf12','tf28')]
            if all(by[i]['eligible'] for i in ids):tfs.append(dict(position=pos,nll=sum(by[i]['validation_nll'] for i in ids)/2))
        tf=min(tfs,key=lambda x:x['nll']) if tfs else None
        preferred=next((s for s in tfs if s['position']=='balanced'),None)
        if preferred and preferred['nll']<=tf['nll']+.02:tf=preferred
        if best is None and tf is None:raise RuntimeError('no common recipe/control eligible')
        self.state['selection']=dict(hb=best,transformer=tf,hb_scores=scores,transformer_scores=tfs,
            scope='Predeclared validation screening rule, not scientific support or test acceptance.')
        self.publish()
        cfgout=self.root/'configs/studies/language/direct-v7/selected';cfgout.mkdir(parents=True,exist_ok=False)
        entries=[]
        for seed in (11,22):
            for model in ('hb40','hb88','tf12','tf28','fc16'):
                if (model.startswith('tf') and tf is None) or (not model.startswith('tf') and best is None):continue
                c=config('transformer' if model.startswith('tf') else 'direct_graph',model,
                         initialization=best['initialization'] if best else 'fan_in',norm=best['norm'] if best else True,
                         position=tf['position'] if tf else 'balanced',seed=seed)
                previous=by.get(c['id'])
                if previous:
                    entries.append(dict(config=previous['config'],parent=previous['output']))
                else:
                    path=cfgout/(c['id']+'.json');atomic_json(path,c);path.chmod(0o444);entries.append(dict(config=str(path)))
        main=self.phase('main',entries,'TL-DIRECT-SCALE-v7',4194304)
        gaps={}
        for n in (1048576,4194304):
            vals={}
            if best is None:continue
            for model in ('hb40','hb88'):
                for seed in (11,22):
                    cid=config('direct_graph',model,initialization=best['initialization'],norm=best['norm'],seed=seed)['id']
                    row=next(r for r in main if r['id']==cid)
                    p=Path(row['output'])/'train'/f'validation-{n}.json'
                    v=json.loads(p.read_text()) if p.exists() else {}
                    if v.get('state')=='completed' and v.get('tokens')==32768:vals[(model,seed)]=v['nll']
            if len(vals)==4:gaps[str(n)]=sum(vals[('hb88',s)]-vals[('hb40',s)] for s in (11,22))/2
        self.state['conditional_decision']=dict(gaps=gaps,trigger=len(gaps)==2 and all(g>.05 for g in gaps.values()))
        self.publish()
        if self.state['conditional_decision']['trigger']:
            entries=[]
            for m in ('hb40','hb88'):
                c=config('direct_graph',m,initialization=best['initialization'],norm=best['norm'],residual=True)
                path=cfgout/(c['id']+'.json');atomic_json(path,c);path.chmod(0o444);entries.append(dict(config=str(path)))
            self.phase('residual',entries,'TL-DIRECT-STRUCTURE-v7',4194304)
        outcomes=[r for phase in self.state['phases'].values() for r in phase.get('results',[])]
        self.state.update(state='completed',finished_at=now(),
            completion_scope='orchestration terminal; individual scientific acceptance requires review',
            eligible_stages=sum(r['eligible'] for r in outcomes),
            failed_or_incomplete_stages=sum(not r['eligible'] for r in outcomes));self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);a=p.parse_args();driver=Campaign(a)
    try:driver.run()
    except BaseException as error:
        driver.state.update(state='failed',error=repr(error),finished_at=now());driver.publish();raise
    finally:atomic_json(a.output/'submission-closed.json',dict(closed=True,state=driver.state['state'],at=now()))
