#!/usr/bin/env python3
"""Durable bounded Mem calibration/qualification, then independent main chains."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import verify_source
from tide_lab.quota import account_quota
from scripts.prepare_hb_mem_v13 import GROUPS

def main(a):
 root=a.root.resolve();source=a.source.resolve();sid=verify_source(source)
 base=source/'configs/studies/language/hb-mem-v13';budget=json.loads((base/'budget.json').read_text());deadline=budget['deadline_epoch']
 quota=account_quota();fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize/2**30
 if min(700,quota['hard_gib'])-quota['used_gib']<92 or free<92:raise RuntimeError('new writes plus simultaneous checkpoint reserve unavailable')
 out=root/f'runs/v13-submit-{a.phase}';out.mkdir(parents=True,exist_ok=False)
 state=dict(state='submitting',source=sid['commit'],started_at=now(),jobs=[]);atomic_json(out/'state.json',state)
 def launch(job,argv,inputs,device='npu',policy=None,threads=4,memory=72,cap=None):
  cap=min(int(deadline-time.time()-30),cap or 172800)
  command=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
   '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
   '--env','TIDE_LAB_VENV='+str(root/('.venv' if job=='v13-data' else '.venv-positive-floor-v1')),
   '--device',device,'--threads',str(threads),'--memory-gib',str(memory),'--timeout',str(cap)]
  if device=='npu':command+=['--env','TIDE_LAB_RESOURCE_POLICY='+str(policy),'--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(cap-180)]
  for p in inputs:command+=['--input',str(p)]
  command+=['--','python',*map(str,argv)];row=dict(job=job,state='submitting',argv=command);state['jobs'].append(row);atomic_json(out/'state.json',state)
  r=subprocess.run(command,capture_output=True,text=True);atomic_json(out/(job+'.json'),dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
  if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
  row.update(state='submitted',at=now());atomic_json(out/'state.json',state);print(job,flush=True)
 if a.phase=='cal':
  for name in GROUPS:
   cfg=base/'configs'/(name+'-sum-s11.json');job='v13-cal-'+name
   launch(job,['scripts/calibrate_hb_mem_v13.py','--config',cfg,'--output',root/'runs'/job,
    '--deadline-epoch',min(deadline,time.time()+5400),'--lens-cli',a.lens_cli],
    [cfg,base/'resource.json',base/'budget.json'],policy=base/'resource.json',cap=5400)
  parent=root/'.local/source-hb-v12-repair-v2'
  for device in ('cpu','npu'):
   launch('v13-check-'+device,['scripts/check_hb_mem_v13.py','--device',device+(':0' if device=='npu' else ''),
    '--output',root/'runs'/('v13-check-'+device),'--parent-source',parent,'--lens-cli',a.lens_cli],
    [base/'budget.json',parent/'.tide-source.json'],device=device,policy=base/'resource.json',threads=2,memory=16,cap=5400)
  locks=[root/'runs/v11-confirmation-data/data-lock.json',root/'runs/v12-data/data-lock.json']
  launch('v13-data',['scripts/prepare_capacity_v11_confirmation.py','--root',root,'--output',root/'runs/v13-data',
   *[arg for path in locks for arg in ('--exclude-lock',path)],'--salt','v13-mem-sumqry-confirmation','--deadline-epoch',time.time()+1800],
   [root/'runs/v9-data-r3/data-lock.json',*locks],device='cpu',threads=2,memory=24,cap=1800)
 else:
  raise NotImplementedError('main submission requires resolved calibration and reviewed qry reuse table')
 state.update(state='submitted',finished_at=now());atomic_json(out/'state.json',state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('cal','main'))
 for n in ('root','source','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
