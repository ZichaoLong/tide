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
 required=budget['planned_new_disk_gib']+budget['account_checkpoint_reserve_gib']
 if min(700,quota['hard_gib'])-quota['used_gib']<required or free<required:raise RuntimeError('new writes plus simultaneous checkpoint reserve unavailable')
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
 if a.phase=='triage':
  policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=8192
  rp=out/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
  launch('v13-numeric-triage',['scripts/diagnose_hb_mem_numerics.py','--output',root/'runs/v13-numeric-triage','--lens-cli',a.lens_cli],
   [base/'budget.json',rp],policy=rp,threads=2,memory=16,cap=1800)
 elif a.phase in ('cal','cal-r2'):
  for name in (GROUPS if a.phase=='cal' else ('q10-l','pool-l')):
   cfg=base/'configs'/(name+'-sum-s11.json');job='v13-'+a.phase+'-'+name
   launch(job,['scripts/calibrate_hb_mem_v13.py','--config',cfg,'--output',root/'runs'/job,
    '--deadline-epoch',min(deadline,time.time()+5400),'--lens-cli',a.lens_cli,*(['--qry-batches'] if a.phase=='cal-r2' else [])],
    [cfg,base/'resource.json',base/'budget.json'],policy=base/'resource.json',cap=5400)
  if a.phase=='cal-r2':
   state.update(state='submitted',finished_at=now());atomic_json(out/'state.json',state);return
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
  table=json.loads((base/'table.json').read_text())
  if not table.get('calibration_resolved') or len(table['cases'])!=16 or len(table['qry_reuse'])!=8:raise ValueError('bounded resolved table required')
  for name in ('v13-check-cpu','v13-check-npu','v13-numeric-triage'):
   r=json.loads((root/'runs'/name/'result.json').read_text());pr=json.loads((root/'runs'/name/'projection.json').read_text())
   if r['state']!='completed' or pr.get('status')!='verified' or not pr.get('readback_matches'):raise ValueError('qualification/Lens incomplete')
  if json.loads((root/'runs/v13-data/state.json').read_text())['state']!='completed':raise ValueError('data incomplete')
  if not (source/'docs/evidence/hb-mem-v13-admission.json').is_file():raise ValueError('reviewed finite numerical admission required')
  ordered=sorted(table['cases'],key=lambda c:json.loads((source/c['config']).read_text())['resource_admission']['targets_per_second'])
  for c in ordered:
   cfg=source/c['config'];config=json.loads(cfg.read_text());job='v13-main-'+c['id'];dest=out/c['id'];dest.mkdir()
   policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=int(config['resource_admission']['peak_mib']*1.2+8192)
   rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
   plan=dest/'worker.json';atomic_json(plan,dict(root=str(root),confirmation_data=str(root/'runs/v13-data'),
    cases=[dict(id=c['id'],config=str(cfg),output=str(root/'runs'/job),max_rss_gib=64)]));plan.chmod(0o444)
   launch(job,['scripts/run_hb_mem_v13_worker.py','--plan',plan,'--data',root/'runs/v9-data-r3',
    '--output',root/'runs'/(job+'-worker'),'--recording-cli',a.lens_cli,'--deadline-epoch',deadline],
    [cfg,plan,rp,base/'table.json',base/'budget.json',root/'runs/v9-data-r3/data-lock.json',root/'runs/v13-data/data-lock.json'],policy=rp)
  for c in table['qry_reuse']:
   job='v13-reuse-'+c['id'];dest=out/c['id'];dest.mkdir();cfg=source/c['config'];config=json.loads(cfg.read_text())
   policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=min(56000,int(config['resource_admission']['peak_mib']*1.2+8192))
   rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
   # These are inference-only new confirmation runs; original checkpoints stay read-only.
   launch(job,['scripts/evaluate_capacity_v11.py','--config',cfg,'--checkpoint',root/c['checkpoint'],
    '--data',root/'runs/v13-data','--training-data',root/'runs/v9-data-r3','--output',root/'runs'/job,
    '--deadline-epoch',deadline,'--lens-cli',a.lens_cli,'--run-id','tl-'+job,'--recording-group','TL-HB-MEM-CONFIRM'],
    [cfg,rp,root/c['checkpoint'],root/'runs/v13-data/data-lock.json',base/'table.json'],policy=rp)
  jobs=out/'hardware-jobs.json';atomic_json(jobs,dict(jobs=[r['job'] for r in state['jobs']]))
  launch('v13-hardware',['scripts/observe_hb_v12_hardware.py','--root',root,'--output',root/'runs/v13-hardware',
   '--job-list',jobs,'--deadline-epoch',deadline],[jobs,base/'budget.json'],device='cpu',threads=1,memory=2)
 state.update(state='submitted',finished_at=now());atomic_json(out/'state.json',state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('cal','cal-r2','triage','main'))
 for n in ('root','source','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
