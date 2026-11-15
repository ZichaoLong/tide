#!/usr/bin/env python3
"""Finite fresh-confirmation controller; completion is not scientific support."""
import argparse,json,subprocess,sys,time,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.job import inspect
from tide_lab.records import atomic_json,now
from tide_lab.quota import account_quota
from tide_lab.supervisor import TERMINAL

def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}

class Controller:
    def __init__(self,a):
        self.a=a;self.root=a.root.resolve();self.source=Path(__file__).resolve().parents[1]
        self.spec=read(a.table);self.deadline=self.spec['deadline_epoch'];self.stopped=False
        a.output.mkdir(parents=True,exist_ok=False)
        self.state=dict(schema='tide-v11-confirmation-controller-v1',state='running',cases=[],started_at=now())
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
    def stop(self,*args):self.stopped=True
    def publish(self):atomic_json(self.a.output/'state.json',self.state)
    def submit(self,c):
        job='v11-fresh-'+c['id'];out=self.root/'runs'/job;jobout=self.root/'artifacts/jobs'/job
        policy=read(self.source/'configs/studies/language/capacity-v11/resource.json');policy['minimum_free_hbm_mib']=48000
        rp=self.a.output/(job+'-resource.json');atomic_json(rp,policy);rp.chmod(0o444)
        cap=min(7200,int(self.deadline-time.time()-60))
        argv=[sys.executable,str(self.source/'scripts/job.py'),'submit','--name',job,'--source',str(self.source),
            '--output',str(jobout),'--env-script',str(self.source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(self.root/'.venv-positive-floor-v1'),'--env','TIDE_LAB_RESOURCE_POLICY='+str(rp),
            '--device','npu','--threads','4','--memory-gib','64','--timeout',str(cap),'--wait-seconds',str(cap-300),
            '--queue-helper',str(self.source/'scripts/run_on_light_device.py')]
        for p in [self.a.table,self.root/c['config'],self.root/c['checkpoint'],rp,self.root/'runs/v11-confirmation-data/data-lock.json',self.root/'runs/v9-data-r3/data-lock.json']:
            argv+=['--input',str(p)]
        argv+=['--','python','scripts/evaluate_capacity_v11.py','--config',str(self.root/c['config']),
            '--checkpoint',str(self.root/c['checkpoint']),'--data',str(self.root/'runs/v11-confirmation-data'),
            '--training-data',str(self.root/'runs/v9-data-r3'),'--output',str(out),'--deadline-epoch',str(self.deadline),
            '--lens-cli',str(self.a.lens_cli),'--run-id','tl-'+job]
        r=subprocess.run(argv,capture_output=True,text=True)
        atomic_json(self.a.output/(job+'-submission.json'),dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
        return dict(id=c['id'],job=job,output=str(out),job_output=str(jobout),state='submitted',at=now())
    def run(self):
        self.publish()
        while not self.stopped and time.time()<self.deadline-300:
            if account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('quota headroom')
            data=read(self.root/'runs/v11-confirmation-data/state.json')
            if data.get('state')=='failed':raise RuntimeError('fresh data failed')
            active=0
            for row in self.state['cases']:
                if row['state']!='submitted':continue
                result=read(Path(row['output'])/'result.json')
                status=read(Path(row['job_output'])/'state.json')
                if status.get('state') in TERMINAL:
                    row.update(state=result.get('state','failed'),result=result.get('nll'),finished_at=now())
                else:active+=1
            known={r['id'] for r in self.state['cases']}
            if data.get('state')=='completed':
                for c in self.spec['cases']:
                    if c['id'] in known or active>=4:continue
                    if c.get('producer'):
                        status=read(self.root/c['producer']/'state.json')
                        if status.get('state') not in TERMINAL:continue
                    result=read(self.root/c['training_result'])
                    progress=result.get('progress',{})
                    if result.get('state')!='completed' or progress.get('tokens')!=c['tokens']:
                        self.state['cases'].append(dict(id=c['id'],state='ineligible',reason='required scientific endpoint missing or failed',at=now()));continue
                    self.state['cases'].append(self.submit(c));active+=1;self.publish()
            if len(self.state['cases'])==len(self.spec['cases']) and not active:
                self.state.update(state='completed',finished_at=now());self.publish();return
            self.publish();time.sleep(30)
        # Controller owns these descendants and must not leave them after deadline/stop.
        for row in self.state['cases']:
            if row['state']=='submitted':subprocess.run([sys.executable,str(self.source/'scripts/job.py'),'stop',str(Path(row['job_output'])/'job.json')],check=False)
        self.state.update(state='stopped' if self.stopped else 'budget_truncated',finished_at=now());self.publish()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','table','output','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();w=Controller(a)
    try:w.run()
    except BaseException as e:w.state.update(state='failed',error=repr(e),finished_at=now());w.publish();raise
