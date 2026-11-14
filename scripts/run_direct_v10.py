#!/usr/bin/env python3
"""Finite single-case durable scheduling and matched-group continuation."""
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
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import TERMINAL
from tide_lab.quota import account_quota

M=2**20


def read(path):return json.loads(path.read_text()) if path.is_file() else {}


def makespan(durations,workers=12):
    """Conservative LPT estimate, not an exclusive-device timing promise."""
    loads=[0.]*workers
    for seconds in sorted(durations,reverse=True):
        i=min(range(workers),key=loads.__getitem__);loads[i]+=seconds
    return max(loads,default=0.)


def continuation_plan(groups,rows,main_target,remaining_seconds,workers=12):
    by={r['id']:r for r in rows};selected=[];decisions=[];durations=[]
    for group in groups:
        group_rows=[by[i] for i in group['ids']]
        eligible=all(r['eligible'] and isinstance(r.get('training_seconds'),(int,float))
            and math.isfinite(r['training_seconds']) and r['training_seconds']>0 for r in group_rows)
        extra=[r['training_seconds']/main_target*(256*M-main_target) for r in group_rows] if eligible else []
        estimate=makespan(durations+extra,workers)*1.35+1800 if eligible else None
        fits=eligible and estimate<remaining_seconds
        decisions.append(dict(group=group['id'],eligible=eligible,estimated_cumulative_seconds=estimate,
            deadline_fits=fits,decision='extend_group' if fits else 'not_extended',
            selection='finite execution and available time only; confirmation/quality not used'))
        if fits:selected+=group['ids'];durations+=extra
    return selected,decisions


class Campaign:
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=Path(__file__).resolve().parents[1]
        self.cfg=self.source/'configs/studies/language/direct-v10';self.budget=read(self.cfg/'budget.json')
        self.catalog=read(self.cfg/'table/catalog.json');self.deadline=self.budget['deadline_epoch'];self.stopped=False
        a.output.mkdir(parents=True,exist_ok=False);self.last_guard=0
        self.state=dict(schema='tide-direct-v10-campaign-v1',state='running',started_at=now(),
            source=read(self.source/'.tide-source.json')['commit'],deadline_epoch=self.deadline,
            jobs=[],phases={},scientific_review='awaiting human review; execution does not create support')
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop);self.publish()

    def stop(self,sig,frame):self.stopped=True
    def publish(self):atomic_json(self.a.output/'state.json',self.state)

    def guard(self):
        if self.stopped:raise InterruptedError('controller stopped')
        if time.time()>=self.deadline-180:raise TimeoutError('shared48h deadline')
        if time.time()-self.last_guard<120:return
        self.last_guard=time.time()
        paths=[p for base in ('runs','artifacts/jobs') for p in (self.root/base).glob('v10-*')]
        used=0.
        if paths:
            raw=subprocess.check_output(['du','-s','-B1',*map(str,paths)],text=True)
            used=sum(int(x.split()[0]) for x in raw.splitlines())/2**30
        quota=account_quota()
        self.state['resources']=dict(at=now(),new_disk_gib=used,quota=quota);self.publish()
        if used>self.budget['max_new_disk_gib'] or quota['hard_quota_headroom_gib']<self.budget['minimum_quota_headroom_gib']:
            raise RuntimeError('v10 disk/headroom guard')

    def submit(self,phase,c,until,parent=None):
        self.guard();cid=c['id'];job='v10-'+phase+'-'+cid
        dest=self.a.output/phase;dest.mkdir(exist_ok=True)
        config=self.cfg/'table/main'/(cid+'.json');out=self.root/'runs'/job
        args=['--until-tokens',str(until),'--run-id','tl-'+job,'--recording-group',c['study_id']]
        row=dict(id=cid,config=str(config),output=str(out),max_rss_gib=64,
            timeout_seconds=self.deadline-time.time()-240,retire_probe_checkpoints=True,train_arguments=args)
        inputs=[config,self.cfg/'budget.json',self.cfg/'table/catalog.json',self.a.data/'data-lock.json']
        if parent:
            p=Path(parent['output']);row.update(reuse_admission=str(p/'check/result.json'),reuse_recovery=str(p/'recovery.json'))
            args+=['--resume',str(p/'train/latest.pt')]
            inputs += [Path(row['reuse_admission']),Path(row['reuse_recovery']),Path(args[-1])]
        plan=dest/(cid+'-worker.json');atomic_json(plan,dict(cases=[row]));plan.chmod(0o444);inputs.append(plan)
        policy=read(self.cfg/'resource.json')
        policy['minimum_free_hbm_mib']=min(56000,max(12288,int(c['resource_admission']['peak_mib']*1.15+8192)))
        resource=dest/(cid+'-resource.json');atomic_json(resource,policy);resource.chmod(0o444);inputs.append(resource)
        cap=int(self.deadline-time.time()-30)
        argv=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',job,'--source',str(self.source),
            '--output',str(self.root/'artifacts/jobs'/job),'--env-script',str(self.source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(self.root/'.venv-positive-floor-v1'),'--env','TIDE_LAB_RESOURCE_POLICY='+str(resource),
            '--device','npu','--threads','4','--memory-gib','72','--timeout',str(cap),'--wait-seconds',str(cap-120),
            '--queue-helper',str(self.source/'scripts/run_on_light_device.py')]
        for p in inputs:argv+=['--input',str(p)]
        argv+=['--','python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(self.a.data),
            '--output',str(self.root/'runs'/(job+'-worker')),'--recording-cli',self.a.lens_cli,'--deadline-epoch',str(self.deadline)]
        result=subprocess.run(argv,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr+' '+result.stdout)
        self.state['jobs'].append(job);self.publish()
        print(json.dumps(dict(submitted=job,until_tokens=until)),flush=True)
        return dict(id=cid,job=job,config=str(config),output=str(out),until_tokens=until)

    def confirm(self,stage,endpoints):
        name='confirm-'+stage;phase=dict(state='submitting',cases=[],skipped=[],started_at=now())
        self.state['phases'][name]=phase;self.publish()
        for row in endpoints:
            if not row['eligible']:
                phase['skipped'].append(dict(id=row['id'],reason='endpoint execution incomplete; no quality exclusion'))
                continue
            self.guard();cid=row['id'];job='v10-'+name+'-'+cid;config=Path(row['config'])
            checkpoint=Path(row['output'])/'train/latest.pt';out=self.root/'runs'/job
            resource=self.a.output/stage/(cid+'-resource.json');cap=int(self.deadline-time.time()-30)
            argv=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',job,'--source',str(self.source),
                '--output',str(self.root/'artifacts/jobs'/job),'--env-script',str(self.source/'scripts/env.sh'),
                '--env','TIDE_LAB_VENV='+str(self.root/'.venv-positive-floor-v1'),
                '--env','TIDE_LAB_RESOURCE_POLICY='+str(resource),'--device','npu','--threads','4',
                '--memory-gib','64','--timeout',str(cap),'--wait-seconds',str(cap-120),
                '--queue-helper',str(self.source/'scripts/run_on_light_device.py')]
            for p in (config,checkpoint,resource,self.a.data/'data-lock.json'):argv+=['--input',str(p)]
            argv+=['--','python','scripts/evaluate_direct_v10.py','--config',str(config),'--checkpoint',str(checkpoint),
                '--data',str(self.a.data),'--output',str(out),'--lens-cli',self.a.lens_cli,'--run-id','tl-'+job,
                '--deadline-epoch',str(self.deadline)]
            r=subprocess.run(argv,capture_output=True,text=True)
            if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
            self.state['jobs'].append(job);phase['cases'].append(dict(id=cid,job=job,output=str(out)));self.publish()
        phase['state']='running';self.publish();self.wait_phase(phase)
        phase['results']=[dict(id=c['id'],job=c['job'],**{k:v for k,v in read(Path(c['output'])/'result.json').items()
            if k in ('state','nll','tokens','error','exposure','checkpoint_sha256')}) for c in phase['cases']]
        phase.update(state='terminal',finished_at=now());self.publish()

    def wait_phase(self,phase):
        while True:
            self.guard();observed=[]
            for c in phase['cases']:
                o=inspect(self.root/'artifacts/jobs'/c['job']/'job.json')
                observed.append(dict(id=c['id'],job=c['job'],state=o['state'].get('state'),
                    terminal=o['termination_verified_now'] and o['state'].get('state') in TERMINAL,
                    physical_id=(o.get('queue') or {}).get('physical_id')))
            phase['observed']=observed;self.publish()
            if all(x['terminal'] for x in observed):return
            time.sleep(20)

    def phase(self,name,configs,until,parents=None):
        phase=dict(state='submitting',started_at=now(),cases=[],results=[]);self.state['phases'][name]=phase;self.publish()
        # Longest measured trajectories request resources first. Each service
        # owns one case, so a short/failed case never pins a strided work list.
        for c in sorted(configs,key=lambda x:x['resource_admission']['targets_per_second']):
            phase['cases'].append(self.submit(name,c,until,parents.get(c['id']) if parents else None));self.publish()
        phase['state']='running';self.publish()
        self.wait_phase(phase)
        for c in phase['cases']:
            p=Path(c['output']);r=read(p/'train/result.json');v=read(p/'train/validation-final.json');pr=read(p/'train/progress.json')
            eligible=(r.get('state')=='completed' and r.get('progress',{}).get('tokens')==until and v.get('state')=='completed'
                and v.get('tokens')==131072 and isinstance(v.get('nll'),(int,float)) and math.isfinite(v['nll'])
                and r.get('inference',{}).get('state')=='completed')
            phase['results'].append(dict(c,eligible=eligible,execution=r.get('state','not_started'),
                tokens=r.get('progress',{}).get('tokens'),nll=v.get('nll'),last_update=pr.get('last_update',{}),
                training_seconds=r.get('training_seconds'),ancestry=r.get('ancestry')))
        phase.update(state='terminal',finished_at=now());self.publish();return phase['results']

    def run(self):
        configs=[read(self.cfg/'table/main'/(cid+'.json')) for cid in self.catalog['cases']]
        main_target=self.catalog['main_target_tokens'];main=self.phase('main',configs,main_target)
        by={x['id']:x for x in main}
        ids,decisions=continuation_plan(self.catalog['continuation_groups'],main,main_target,self.deadline-time.time(),
            self.budget['max_npu_workers'])
        self.state['continuation_decisions']=decisions;self.publish()
        # Decisions are frozen before observing the fresh confirmation split.
        # Every eligible common endpoint is evaluated, including negatives.
        self.confirm('main',main)
        if ids:
            extended=self.phase('extend',[c for c in configs if c['id'] in ids],256*M,parents=by)
            self.confirm('extend',extended)
        self.state.update(state='completed',finished_at=now(),scope='finite execution terminal; individual failures retained; scientific review remains separate');self.publish()

    def stop_children(self):
        for job in self.state['jobs']:
            subprocess.run([sys.executable,str(self.source/'scripts/job.py'),'stop',str(self.root/'artifacts/jobs'/job)],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);a=p.parse_args();driver=Campaign(a)
    try:driver.run()
    except BaseException as e:
        driver.state.update(state='failed',error=repr(e),finished_at=now());driver.publish();driver.stop_children();raise
    finally:atomic_json(a.output/'submission-closed.json',dict(state=driver.state['state'],closed=True,at=now()))
