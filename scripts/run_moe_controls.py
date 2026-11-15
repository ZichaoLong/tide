#!/usr/bin/env python3
"""Bounded calibration->recipe selection->fresh-seed MoE/dense->confirmation.

Project-owned producer. An outer service exit never upgrades an inner failure.
All generated worker plans are immutable and live outside the source snapshot.
"""
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
from scripts.job import inspect
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import TERMINAL
from tide_lab.quota import account_quota

M=2**20


def read(path):return json.loads(path.read_text()) if path.is_file() else {}


def eligible_result(result,validation,until):
    return (result.get('state')=='completed' and result.get('progress',{}).get('tokens')==until
        and validation.get('state')=='completed' and validation.get('tokens')==131072
        and isinstance(validation.get('nll'),(int,float)) and math.isfinite(validation['nll'])
        and result.get('inference',{}).get('state')=='completed')


def select_within_shape(rows,number):
    selected=[];groups={}
    for row in rows:groups.setdefault(row['shape'],[]).append(row)
    for shape,items in sorted(groups.items()):
        selected.extend(sorted((r for r in items if r['eligible']),key=lambda r:(r['nll'],r['id']))[:number])
    return selected


class Campaign:
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=Path(__file__).resolve().parents[1]
        self.cfg=self.source/'configs/studies/language/moe-controls';self.budget=read(self.cfg/'budget.json')
        self.catalog=read(self.cfg/'table/catalog.json');self.deadline=self.budget['deadline_epoch']
        self.stopped=False;self.last_guard=0;a.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-moe-campaign-v1',state='running',started_at=now(),
            source=read(self.source/'.tide-source.json')['commit'],deadline_epoch=self.deadline,
            jobs=[],phases={},scientific_review='awaiting review; execution is not scientific support')
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop);self.publish()

    def stop(self,sig,frame):self.stopped=True
    def publish(self):atomic_json(self.a.output/'state.json',self.state)

    def guard(self):
        if self.stopped:raise InterruptedError('controller stopped')
        if time.time()>=self.deadline-180:raise TimeoutError('shared MoE48h deadline')
        if time.time()-self.last_guard<120:return
        self.last_guard=time.time()
        paths=[p for base in ('runs','artifacts/jobs') for p in (self.root/base).glob('moe-*')]
        used=0.
        if paths:
            raw=subprocess.check_output(['du','-s','-B1',*map(str,paths)],text=True)
            used=sum(int(line.split()[0]) for line in raw.splitlines())/2**30
        quota=account_quota();self.state['resources']=dict(at=now(),new_disk_gib=used,quota=quota);self.publish()
        if used>self.budget['max_new_disk_gib'] or quota['hard_quota_headroom_gib']<self.budget['minimum_quota_headroom_gib']:
            raise RuntimeError('MoE disk/headroom guard')

    def observe(self,row):
        o=inspect(self.root/'artifacts/jobs'/row['job']/'job.json')
        return dict(id=row['id'],job=row['job'],state=o['state'].get('state'),
            terminal=o['termination_verified_now'] and o['state'].get('state') in TERMINAL,
            physical_id=(o.get('queue') or {}).get('physical_id'))

    def launch(self,job,inputs,resource,command):
        self.guard();cap=int(self.deadline-time.time()-30)
        argv=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',job,'--source',str(self.source),
            '--output',str(self.root/'artifacts/jobs'/job),'--env-script',str(self.source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(self.root/'.venv-positive-floor-v1'),
            '--env','TIDE_LAB_RESOURCE_POLICY='+str(resource),'--device','npu','--threads','4','--memory-gib','72',
            '--timeout',str(cap),'--wait-seconds',str(cap-120),'--queue-helper',str(self.source/'scripts/run_on_light_device.py')]
        for p in inputs:argv+=['--input',str(p)]
        result=subprocess.run(argv+['--',*command],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr+' '+result.stdout)
        self.state['jobs'].append(job);self.publish()
        print(json.dumps(dict(submitted=job,at=now())),flush=True)

    def submit(self,phase,c,until,parent=None):
        cid=c['id'];job='moe-'+phase+'-'+cid;dest=self.a.output/phase;dest.mkdir(exist_ok=True)
        config=self.cfg/'table/configs'/(cid+'.json');out=self.root/'runs'/job
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
        policy['minimum_free_hbm_mib']=max(16384,int(c['resource_admission']['peak_mib']*1.5+8192))
        resource=dest/(cid+'-resource.json');atomic_json(resource,policy);resource.chmod(0o444);inputs.append(resource)
        self.launch(job,inputs,resource,['python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(self.a.data),
            '--output',str(self.root/'runs'/(job+'-worker')),'--recording-cli',self.a.lens_cli,'--deadline-epoch',str(self.deadline)])
        return dict(id=cid,shape=c['selection_key'],job=job,config=str(config),output=str(out),until_tokens=until,resource=str(resource))

    def phase(self,name,configs,until,parents=None):
        phase=dict(state='running',started_at=now(),cases=[],results=[],not_submitted=[c['id'] for c in configs])
        self.state['phases'][name]=phase;self.publish();pending=list(configs)
        while pending or any(not self.observe(r)['terminal'] for r in phase['cases']):
            self.guard();observed=[self.observe(r) for r in phase['cases']]
            live=sum(not r['terminal'] for r in observed)
            while pending and live<self.budget['max_npu_workers']:
                c=pending.pop(0);phase['cases'].append(self.submit(name,c,until,parents.get(c['id']) if parents else None))
                phase['not_submitted']=[c['id'] for c in pending];live+=1;self.publish()
            phase['observed']=[self.observe(r) for r in phase['cases']];self.publish()
            if pending or any(not r['terminal'] for r in phase['observed']):time.sleep(20)
        for c in phase['cases']:
            p=Path(c['output']);r=read(p/'train/result.json');v=read(p/'train/validation-final.json')
            worker=read(self.root/'runs'/(c['job']+'-worker')/'state.json')
            failed_case=next((x for x in worker.get('cases',[]) if x['id']==c['id']),{})
            phase['results'].append(dict(c,eligible=eligible_result(r,v,until),execution=r.get('state',failed_case.get('state','not_started')),
                failure=r.get('error',failed_case.get('reason')),tokens=r.get('progress',{}).get('tokens'),
                nll=v.get('nll'),training_seconds=r.get('training_seconds')))
        phase.update(state='terminal',finished_at=now());self.publish();return phase['results']

    def retire_calibration_optimizers(self,rows):
        # New-batch, ended calibration only. Scientific weights and failures stay.
        receipt=[]
        for row in rows:
            if not row.get('eligible') or not self.observe(row)['terminal']:continue
            train=Path(row['output'])/'train'
            weights=train/('tokens-'+str(row['until_tokens'])+'.pt')
            if not weights.is_file():continue
            for name in ('latest.pt','previous.pt'):
                path=train/name
                if path.is_file():
                    receipt.append(dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,
                        retained_weights=str(weights),meaning='completed calibration optimizer no longer an ancestry input'))
        manifest=self.a.output/'calibration-optimizer-retirement.json'
        atomic_json(manifest,dict(state='approved_by_batch_retention_policy',files=receipt,at=now()))
        for row in receipt:
            path=Path(row['path'])
            if digest(path)!=row['sha256']:raise ValueError('retirement identity changed')
            path.unlink()
        atomic_json(manifest,dict(state='completed',files=receipt,at=now()))

    def reused_dense(self,c):
        """Narrow exact-recipe reuse: immutable v10 small dense128Mi endpoint."""
        if c['selection_key']!='small-dense' or c['training']['lr']!=.001 or c['training']['batch']!=64:return None
        cid=f"tf-d192-l6-s{c['seed']}";base=self.root/'runs'/('v10-main-'+cid)
        old_path=self.root/'.local/source-v10-execution/configs/studies/language/direct-v10/table/main'/(cid+'.json')
        old=read(old_path);result=read(base/'train/result.json');val=read(base/'train/validation-final.json')
        checkpoint=base/'train'/f'tokens-{128*M}.pt'
        keys=('global_batch','batch','length','lr','betas','eps','weight_decay','decay_policy','clip_norm',
              'warmup_tokens','lr_decay_tokens','lr_floor_fraction','optimizer_foreach')
        effective=lambda t,k:t.get(k,False) if k=='optimizer_foreach' else t.get(k)
        same=old and old['transformer']==c['transformer'] and old['seed']==c['seed'] and all(effective(old['training'],k)==effective(c['training'],k) for k in keys)
        if not same or not checkpoint.is_file() or not eligible_result(result,val,128*M):return None
        if result['identity']['data_lock_sha256']!=digest(self.a.data/'data-lock.json'):return None
        return dict(id=c['id'],shape=c['selection_key'],job='v10-main-'+cid,config=str(old_path),output=str(base),
            until_tokens=128*M,checkpoint=str(checkpoint),eligible=True,execution='completed',tokens=128*M,
            nll=val['nll'],training_seconds=result['training_seconds'],reused=True,
            original_identity=result['identity'],recipe_equality_keys=list(keys),
            scope='existing dense recipe/result retains original identity; not a new independent replicate')

    def confirm(self,rows):
        phase=dict(state='running',started_at=now(),cases=[],results=[],skipped=[r['id'] for r in rows if not r['eligible']])
        self.state['phases']['confirm']=phase;self.publish();pending=[r for r in rows if r['eligible']]
        while pending or any(not self.observe(r)['terminal'] for r in phase['cases']):
            self.guard();live=sum(not self.observe(r)['terminal'] for r in phase['cases'])
            while pending and live<self.budget['max_npu_workers']:
                row=pending.pop(0);cid=row['id'];job='moe-confirm-'+cid
                config=Path(row['config']);checkpoint=Path(row.get('checkpoint') or str(Path(row['output'])/'train/latest.pt'))
                out=self.root/'runs'/job;resource=self.cfg/'resource.json'
                self.launch(job,[config,checkpoint,resource,self.a.data/'data-lock.json'],resource,
                    ['python','scripts/evaluate_direct_v10.py','--config',str(config),'--checkpoint',str(checkpoint),
                     '--data',str(self.a.data),'--output',str(out),'--lens-cli',self.a.lens_cli,'--run-id','tl-'+job,
                     '--recording-group','TL-MOE-CONFIRM','--deadline-epoch',str(self.deadline)])
                phase['cases'].append(dict(id=cid,job=job,output=str(out)));live+=1;self.publish()
            phase['observed']=[self.observe(r) for r in phase['cases']];self.publish()
            if pending or any(not r['terminal'] for r in phase['observed']):time.sleep(20)
        for row in phase['cases']:
            r=read(Path(row['output'])/'result.json')
            phase['results'].append(dict(row,**{k:r[k] for k in ('state','nll','tokens','error') if k in r}))
        phase.update(state='terminal',finished_at=now());self.publish()

    def run(self):
        configs=[read(self.cfg/'table/configs'/(cid+'.json')) for cid in self.catalog['screen_cases']]
        by={c['id']:c for c in configs}
        screen=self.phase('screen8',configs,8*M)
        promoted=select_within_shape(screen,2);parents={r['id']:r for r in promoted}
        self.state['selection8']=dict(selected=[r['id'] for r in promoted],method='lowest complete validation NLL within each of six shapes; max2; no confirmation')
        self.publish()
        extended=self.phase('screen32',[by[r['id']] for r in promoted],32*M,parents)
        chosen=select_within_shape(extended,1)
        self.state['selection32']=dict(selected=[r['id'] for r in chosen],method='lowest complete32Mi validation NLL within shape; seed33 only; no confirmation')
        self.publish()
        self.retire_calibration_optimizers(screen+extended)
        main_configs=[];reused=[]
        for row in chosen:
            for cid in self.catalog['main_by_recipe'][row['id']]:
                c=read(self.cfg/'table/configs'/(cid+'.json'));prior=self.reused_dense(c)
                if prior:reused.append(prior)
                else:main_configs.append(c)
        self.state['reused_dense']=reused;self.publish()
        main=self.phase('main128',main_configs,128*M)
        # Freeze decisions before using the independent split. No automatic extension.
        self.confirm(main+reused)
        failures=[dict(phase=p,id=r['id'],state=r.get('execution',r.get('state'))) for p,v in self.state['phases'].items()
            for r in v.get('results',[]) if (not r.get('eligible',False) if p!='confirm' else r.get('state')!='completed')]
        self.state.update(state='completed',finished_at=now(),retained_failures=failures,
            scope='bounded execution terminal, failures preserved; scientific quality and strict numerics require review')
        self.publish()

    def stop_children(self):
        for job in self.state['jobs']:
            try:
                if self.observe(dict(id=job,job=job))['terminal']:continue
                subprocess.run([sys.executable,str(self.source/'scripts/job.py'),'stop',str(self.root/'artifacts/jobs'/job)],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
            except Exception as e:print('stop observation: '+repr(e),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);a=p.parse_args();driver=Campaign(a)
    try:driver.run()
    except BaseException as e:
        driver.state.update(state='failed',error=repr(e),finished_at=now());driver.publish();driver.stop_children();raise
    finally:atomic_json(a.output/'submission-closed.json',dict(state=driver.state['state'],closed=True,at=now()))
