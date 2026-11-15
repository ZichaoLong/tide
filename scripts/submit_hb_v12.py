#!/usr/bin/env python3
"""Submit fixed calibration/data or all38 independent durable comparison chains."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import verify_source
from tide_lab.quota import account_quota

CALIBRATIONS=['base-q3','base-q10','share8-q3','share8-d128-q3','share8-d192-q3','base128-q3','top2-q3','ports4-q3']

def main(a):
    root=a.root.resolve();source=a.source.resolve();sid=verify_source(source)
    base=source/'configs/studies/language/hb-v12';budget=json.loads((base/'budget.json').read_text());deadline=budget['deadline_epoch']
    if account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('quota headroom')
    out=root/f'runs/v12-submit-{a.phase}';out.mkdir(parents=True,exist_ok=False)
    state=dict(state='submitting',source=sid['commit'],started_at=now(),jobs=[]);atomic_json(out/'state.json',state)
    def launch(job,argv,inputs,device='npu',policy=None,threads=4,memory=72):
        cap=int(deadline-time.time()-30)
        command=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
            '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),'--device',device,
            '--threads',str(threads),'--memory-gib',str(memory),'--timeout',str(cap)]
        if device=='npu':command+=['--env','TIDE_LAB_RESOURCE_POLICY='+str(policy),'--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(cap-180)]
        for p in inputs:command+=['--input',str(p)]
        command+=['--','python',*map(str,argv)]
        row=dict(job=job,state='submitting',argv=command);state['jobs'].append(row);atomic_json(out/'state.json',state)
        r=subprocess.run(command,capture_output=True,text=True);atomic_json(out/(job+'.json'),dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
        row.update(state='submitted',at=now());atomic_json(out/'state.json',state);print(job,flush=True)
    if a.phase=='cal':
        for name in CALIBRATIONS:
            cfg=base/'configs'/(name+'-s11.json');job='v12-cal-'+name
            launch(job,['scripts/calibrate_hb_v12.py','--config',cfg,'--output',root/'runs'/job,
                '--deadline-epoch',min(deadline,time.time()+5400),'--lens-cli',a.lens_cli],
                [cfg,base/'resource.json',base/'budget.json'],policy=base/'resource.json')
        launch('v12-data',['scripts/prepare_capacity_v11_confirmation.py','--root',root,'--output',root/'runs/v12-data',
            '--exclude-lock',root/'runs/v11-confirmation-data/data-lock.json','--salt','v12-fresh-confirmation','--deadline-epoch',time.time()+1800],
            [root/'runs/v9-data-r3/data-lock.json',root/'runs/v11-confirmation-data/data-lock.json'],device='cpu',threads=2,memory=24)
    else:
        table=json.loads((base/'table.json').read_text())
        if len(table['cases'])!=38:raise ValueError('finite38 trajectories')
        if json.loads((root/'runs/v12-data/state.json').read_text())['state']!='completed':raise ValueError('confirmation preparation incomplete')
        for c in table['cases']:
            cfg=source/c['config'];config=json.loads(cfg.read_text());job='v12-main-'+c['id']
            if 'resource_admission' not in config:raise ValueError('calibration missing')
            dest=out/c['id'];dest.mkdir();policy=json.loads((base/'resource.json').read_text())
            policy['minimum_free_hbm_mib']=min(56000,max(16384,int(config['resource_admission']['peak_mib']*1.15+8192)))
            rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
            plan=dest/'worker.json';atomic_json(plan,dict(root=str(root),confirmation_data=str(root/'runs/v12-data'),
                cases=[dict(id=c['id'],config=str(cfg),output=str(root/'runs'/job),max_rss_gib=64)]));plan.chmod(0o444)
            launch(job,['scripts/run_hb_v12_worker.py','--plan',plan,'--data',root/'runs/v9-data-r3',
                '--output',root/'runs'/(job+'-worker'),'--recording-cli',a.lens_cli,'--deadline-epoch',deadline],
                [cfg,plan,rp,base/'table.json',base/'budget.json',root/'runs/v9-data-r3/data-lock.json',root/'runs/v12-data/data-lock.json'],policy=rp)
    state.update(state='submitted',finished_at=now());atomic_json(out/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('cal','main'))
    for n in ('root','source','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
