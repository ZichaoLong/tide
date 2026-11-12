#!/usr/bin/env python3
"""Bounded v9 orchestration: measured batches, finite recipes, main and conditional continuations."""
import argparse
import copy
import hashlib
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


def read(p):return json.loads(p.read_text()) if p.is_file() else {}


def batch_key(c):
    if c['kind']=='transformer':
        t=c['transformer'];return f'tf-{t["width"]}-{t["layers"]}'
    p=c['direct']['plan'];g=p['graph']
    return f'direct-{g["nodes"]}-{p["width"]}-{p["memories"][0]["ffn_width"]}'


class Campaign:
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=Path(__file__).resolve().parents[1]
        self.cfg=self.source/'configs/studies/language/direct-v9';self.budget=read(self.cfg/'budget.json')
        self.deadline=self.budget['deadline_epoch'];self.stopped=False
        a.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-direct-v9-campaign-v1',state='running',started_at=now(),
            source=read(self.source/'.tide-source.json')['commit'],deadline_epoch=self.deadline,
            jobs=[],phases={},scientific_review='awaiting human review')
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
        self.publish();self.guard_last=0

    def stop(self,sig,frame):self.stopped=True
    def publish(self):atomic_json(self.a.output/'state.json',self.state)
    def guard(self):
        if self.stopped:raise InterruptedError('controller stop requested')
        if time.time()>=self.deadline-180:raise TimeoutError('shared48h deadline')
        if time.time()-self.guard_last<120:return
        self.guard_last=time.time()
        paths=[p for base in ('runs','artifacts/jobs') for p in (self.root/base).glob('v9-*')]
        sizes=subprocess.check_output(['du','-s','-B1',*[str(p) for p in paths]],text=True)
        used=sum(int(s.split()[0]) for s in sizes.splitlines())/2**30
        quota=account_quota();headroom=quota['hard_quota_headroom_gib']
        self.state['resources']=dict(at=now(),new_disk_gib=used,hard_quota_headroom_gib=headroom,quota=quota)
        self.publish()
        if used>self.budget['max_new_disk_gib'] or headroom<200:
            raise RuntimeError('v9 disk/headroom bound reached')

    def submit(self,name,argv,inputs,device='npu',threads=4,memory=56):
        self.guard();cap=int(self.deadline-time.time()-30)
        cmd=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',name,'--source',str(self.source),
            '--output',str(self.root/'artifacts/jobs'/name),'--env-script',str(self.source/'scripts/env.sh'),
            '--env',f'TIDE_LAB_VENV={self.root}/.venv-positive-floor-v1',
            '--env',f'TIDE_LAB_RESOURCE_POLICY={self.cfg}/resource.json',
            '--device',device,'--threads',str(threads),'--memory-gib',str(memory),'--timeout',str(cap)]
        if device=='npu':cmd+=['--queue-helper',str(self.source/'scripts/run_on_light_device.py'),
                              '--wait-seconds',str(cap-120)]
        for p in [self.cfg/'budget.json',self.cfg/'resource.json',*inputs]:cmd+=['--input',str(p)]
        result=subprocess.run(cmd+['--',*argv],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr+' '+result.stdout)
        self.state['jobs'].append(name);self.publish();print(json.dumps(dict(submitted=name)),flush=True)
        return name

    def wait(self,jobs):
        while True:
            self.guard();states=[inspect(self.root/'artifacts/jobs'/j/'job.json') for j in jobs]
            if all(x['termination_verified_now'] and x['state'].get('state') in TERMINAL for x in states):return
            self.state['observed_jobs']=[dict(id=x['job_id'],state=x['state'].get('state'),
                physical_id=(x.get('queue') or {}).get('physical_id')) for x in states];self.publish();time.sleep(15)

    def calibrated_config(self,c,batches):
        x=copy.deepcopy(c);key=batch_key(c)
        if key not in batches:return None
        b=batches[key];x['training'].update(batch=b,eval_batch=b)
        x['batch_calibration']=dict(key=key,microbatch=b,global_batch=64)
        return x

    def phase(self,name,rows,workers=12,parents=None):
        dest=self.a.output/name;dest.mkdir();configs=dest/'configs';configs.mkdir()
        records=[]
        for c in rows:
            if c.get('direct'):
                plan=c['direct']['plan'];plan.pop('sha256',None)
                plan['sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
            path=configs/(c['id']+'.json');atomic_json(path,c);path.chmod(0o444)
            row=dict(id=c['id'],config=str(path),output=str(self.root/'runs'/('v9-'+name+'-'+c['id'])),
                max_rss_gib=48,timeout_seconds=max(900,self.deadline-time.time()-180),retire_probe_checkpoints=True,
                train_arguments=['--until-tokens',str(c['stage_until_tokens']),
                    '--run-id','tl-v9-'+name+'-'+c['id'],'--recording-group',c['study_id']])
            if parents and c['id'] in parents:
                parent=Path(parents[c['id']]);row.update(reuse_admission=str(parent/'check/result.json'),
                    reuse_recovery=str(parent/'recovery.json'))
                row['train_arguments']+=['--resume',str(parent/'train/latest.pt')]
            records.append(row)
        phase=dict(state='submitting',cases=records,jobs=[],started_at=now());self.state['phases'][name]=phase;self.publish()
        # One case at a time per worker; failed cases never stop unrelated entries.
        for i in range(min(workers,len(records))):
            cases=records[i::workers];plan=dest/f'w{i}.json';atomic_json(plan,dict(cases=cases));plan.chmod(0o444)
            inputs=[plan,self.a.data/'data-lock.json',*[Path(c['config']) for c in cases]]
            for c in cases:
                if c.get('reuse_admission'):inputs += [Path(c['reuse_admission']),Path(c['reuse_recovery']),Path(c['train_arguments'][-1])]
            job='v9-'+name+'-w'+str(i)
            argv=['python','scripts/campaign_worker.py','--plan',str(plan),'--data',str(self.a.data),
                '--output',str(self.root/'runs'/(job+'-worker')),'--recording-cli',self.a.lens_cli,
                '--deadline-epoch',str(self.deadline)]
            phase['jobs'].append(self.submit(job,argv,inputs));self.publish()
        phase['state']='running';self.publish();self.wait(phase['jobs'])
        results=[]
        for c in records:
            p=Path(c['output']);r=read(p/'train/result.json');v=read(p/'train/validation-final.json')
            progress=read(p/'train/progress.json');proj=read(p/'train/projection.json')
            cfg=read(Path(c['config']));until=cfg['stage_until_tokens']
            eligible=(r.get('state')=='completed' and r.get('progress',{}).get('tokens')==until
                and v.get('state')=='completed' and math.isfinite(v.get('nll',float('nan'))))
            results.append(dict(id=c['id'],config=c['config'],output=c['output'],eligible=eligible,
                execution=r.get('state','not_started'),validation_nll=v.get('nll'),tokens=r.get('progress',{}).get('tokens'),
                last_work=progress.get('last_update',{}),lens_verified=proj.get('status')=='verified' and proj.get('readback_matches') is True))
        phase.update(state='terminal',results=results,finished_at=now());self.publish();return results

    def run(self):
        mains=[read(p) for p in sorted((self.cfg/'table/main').glob('*.json')) if p.name!='catalog.json']
        screens=[read(p) for p in sorted((self.cfg/'table/screen').glob('*.json')) if p.name!='catalog.json']
        reps={}
        for c in sorted(mains,key=lambda c:('pool' not in c['id'],'quad' not in c['id'] and '-q' not in c['id'])):
            reps.setdefault(batch_key(c),c)
        jobs=[];batch_paths={}
        for key,c in reps.items():
            p=self.cfg/'table/main'/(c['id']+'.json');out=self.root/'runs'/('v9-batch-'+key)
            jobs.append(self.submit('v9-batch-'+key,['python','scripts/calibrate_direct_v9.py','--config',str(p),
                '--output',str(out),'--lens-cli',self.a.lens_cli,'--deadline-epoch',str(min(self.deadline,time.time()+6*3600))],[p]))
            batch_paths[key]=out
        self.wait(jobs);batches={}
        for key,out in batch_paths.items():
            r=read(out/'result.json')
            if r.get('state')=='completed':batches[key]=r['selected_batch']
        self.state['batches']=batches;self.publish()
        if not any(read(p/'projection.json').get('readback_matches') is True for p in batch_paths.values()):
            raise RuntimeError('no small calibration record verified in Lens')
        while not (self.a.data/'data-lock.json').exists():
            self.guard();s=read(self.a.data/'state.json')
            if s.get('state')=='failed':raise RuntimeError('expanded data failed; no silent old-pool substitution')
            self.state['waiting_for']='expanded data lock';self.publish();time.sleep(15)
        self.state.pop('waiting_for',None)
        screen=[self.calibrated_config(c,batches) for c in screens];screen=[c for c in screen if c]
        result=self.phase('screen',screen)
        byid={r['id']:r for r in result};selection={};activity={}
        for key in sorted({c['selection_key'] for c in screen if c.get('selection_key')}):
            good=[(c,byid[c['id']]) for c in screen if c.get('selection_key')==key and byid[c['id']]['eligible']]
            if not good:continue
            best=min(r['validation_nll'] for c,r in good)
            c,r=min((x for x in good if x[1]['validation_nll']<=best+.01),key=lambda x:(x[0]['training']['lr'],x[1]['validation_nll']))
            selection[key]=dict(id=c['id'],lr=c['training']['lr'],transformer=c.get('transformer'),nll=r['validation_nll'])
        for key in sorted({c['activity_selection_key'] for c in screen if c.get('activity_selection_key')}):
            band=(.025,.04) if key.endswith('q3') else (.08,.12)
            good=[(c,byid[c['id']]) for c in screen if c.get('activity_selection_key')==key and byid[c['id']]['eligible']]
            if not good:continue
            def rank(x):
                c,r=x;f=r['last_work'].get('activity/body_full_all_fraction',1.)
                gap=max(band[0]-f,0,f-band[1]);return (gap>0,gap,r['validation_nll'])
            c,r=min(good,key=rank);tau=c['direct']['plan']['regions'][0]['allocation_temperature']
            activity[key]=dict(id=c['id'],temperature=tau,nll=r['validation_nll'],
                observed_body_fraction=r['last_work'].get('activity/body_full_all_fraction'),band=band,
                meaning='short target-budget screen; final long-trajectory activity is not guaranteed')
        self.state.update(recipe_selection=selection,activity_selection=activity);self.publish()
        main=[];skipped=[]
        for c in mains:
            x=self.calibrated_config(c,batches)
            if not x:skipped.append(dict(id=c['id'],reason='no safe measured batch'));continue
            key=c.get('selection_key');ak=c.get('activity_selection_key')
            if key:
                if key not in selection:skipped.append(dict(id=c['id'],reason='all recipe screens failed'));continue
                chosen=selection[key];x['training']['lr']=chosen['lr']
                if chosen['transformer']:x['transformer']=chosen['transformer']
                x['selected_recipe']=chosen
            if ak:
                if ak not in activity:skipped.append(dict(id=c['id'],reason='all activity screens failed'));continue
                for s in x['direct']['plan']['regions']:s['allocation_temperature']=activity[ak]['temperature']
                x['selected_activity']=activity[ak]
            main.append(x)
        self.state['not_started']=skipped;self.publish()
        result=self.phase('main',main);by={r['id']:r for r in result};configs={c['id']:c for c in main}
        extensions=[];parents={}
        ids=[f'{name}-s{seed}' for name in ('hb88-pool','hb88-quad','tf-hb88') for seed in (11,22)]
        if all(by.get(i,{}).get('eligible') for i in ids):
            for i in ids:extensions.append(configs[i]);parents[i]=by[i]['output']
        for r,kind in ((32,'q10'),(64,'q3')):
            ids=[f'r{r}-h2-{k}-s{s}' for k in (kind,'bern','pool') for s in (11,22)]
            band=(.08,.12) if r==32 else (.025,.04)
            if all(by.get(i,{}).get('eligible') for i in ids) and all(
                band[0]<=by[f'r{r}-h2-{kind}-s{s}']['last_work'].get('activity/body_full_all_fraction',1.)<=band[1] for s in (11,22)):
                for i in ids:extensions.append(configs[i]);parents[i]=by[i]['output']
                break
        # Same config/optimizer identity; exact restore, low LR floor after initial horizon.
        if extensions and self.deadline-time.time()>3600:
            # stage_until is an orchestration field; keep original config bytes to preserve identity.
            self.extend(extensions,parents)
        self.state.update(state='completed',finished_at=now(),completion_scope='finite orchestration terminal; individual failed cases and science review retained');self.publish()

    def extend(self,configs,parents):
        dest=self.a.output/'extend';dest.mkdir();records=[]
        for c in configs:
            p=Path(parents[c['id']]);original=self.a.output/'main/configs'/(c['id']+'.json')
            records.append(dict(id=c['id'],config=str(original),output=str(self.root/'runs'/('v9-extend-'+c['id'])),
                max_rss_gib=48,timeout_seconds=self.deadline-time.time()-180,
                reuse_admission=str(p/'check/result.json'),reuse_recovery=str(p/'recovery.json'),
                train_arguments=['--until-tokens',str(c['training']['target_tokens']),'--resume',str(p/'train/latest.pt'),
                                 '--run-id','tl-v9-extend-'+c['id'],'--recording-group',c['study_id']]))
        phase=dict(state='running',cases=records,jobs=[],started_at=now(),meaning='conditional optimizer continuation; no new seed; LR remains at original horizon floor')
        self.state['phases']['extend']=phase;self.publish()
        for i,c in enumerate(records):
            p=dest/f'w{i}.json';atomic_json(p,dict(cases=[c]));p.chmod(0o444)
            phase['jobs'].append(self.submit(f'v9-extend-w{i}',['python','scripts/campaign_worker.py','--plan',str(p),
                '--data',str(self.a.data),'--output',str(self.root/'runs'/f'v9-extend-w{i}-worker'),
                '--recording-cli',self.a.lens_cli,'--deadline-epoch',str(self.deadline)],
                [p,Path(c['config']),Path(c['reuse_admission']),Path(c['reuse_recovery']),Path(c['train_arguments'][3])]))
        self.wait(phase['jobs']);phase.update(state='terminal',finished_at=now());self.publish()

    def stop_children(self):
        for job in self.state['jobs']:
            subprocess.run([sys.executable,str(self.source/'scripts/job.py'),'stop',str(self.root/'artifacts/jobs'/job)],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','data','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--lens-cli',required=True);a=p.parse_args();driver=Campaign(a)
    try:driver.run()
    except BaseException as e:
        driver.state.update(state='failed',error=repr(e),finished_at=now());driver.publish();driver.stop_children();raise
    finally:atomic_json(a.output/'submission-closed.json',dict(closed=True,state=driver.state['state'],at=now()))
