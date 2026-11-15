#!/usr/bin/env python3
"""Submit the diagnosis-selected four HB chains and finite confirmation controller."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.supervisor import verify_source
from tide_lab.quota import account_quota

def main(a):
    root=a.root.resolve();source=a.source.resolve();sid=verify_source(source)
    base=source/'configs/studies/language/capacity-v11';spec=json.loads((base/'hb-table.json').read_text())
    deadline=json.loads((base/'budget.json').read_text())['deadline_epoch']
    if len(spec['cases'])!=4:raise ValueError('only four selected HB chains')
    if account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('quota headroom')
    control=root/'runs/v11-hb-control';control.mkdir(parents=True,exist_ok=False)
    state=dict(state='submitting',source_commit=sid['commit'],started_at=now(),jobs=[]);atomic_json(control/'state.json',state)
    def launch(job,argv,inputs,device='npu',resource=None,threads=4):
        cap=int(deadline-time.time()-30)
        command=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
            '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
            '--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),'--device',device,'--threads',str(threads),
            '--memory-gib','72' if device=='npu' else '8','--timeout',str(cap)]
        if device=='npu':command+=['--env','TIDE_LAB_RESOURCE_POLICY='+str(resource),'--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(cap-300)]
        for p in inputs:command+=['--input',str(p)]
        command+=['--','python',*argv]
        row=dict(job=job,state='submitting',argv=command);state['jobs'].append(row);atomic_json(control/'state.json',state)
        r=subprocess.run(command,capture_output=True,text=True);atomic_json(control/(job+'-submission.json'),dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
        row.update(state='submitted',at=now());atomic_json(control/'state.json',state);print(job,flush=True)
    for c in spec['cases']:
        cid=c['id'];job='v11-hb-'+cid;dest=control/cid;dest.mkdir()
        cfg=source/c['config'];policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=48000
        rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
        plan=dest/'worker.json';atomic_json(plan,dict(root=str(root),cases=[dict(id=cid,config=str(cfg),output=str(root/'runs'/job),max_rss_gib=64)]));plan.chmod(0o444)
        launch(job,['scripts/run_capacity_v11_hb.py','--plan',str(plan),'--data',str(root/'runs/v9-data-r3'),
            '--output',str(root/'runs'/(job+'-worker')),'--recording-cli',str(a.lens_cli.resolve()),'--deadline-epoch',str(deadline)],
            [plan,cfg,rp,base/'budget.json',base/'hb-table.json',source/'docs/evidence/capacity-v11-diagnostics.json',root/'runs/v9-data-r3/data-lock.json'],resource=rp,threads=4)
    launch('v11-fresh-controller',['scripts/run_capacity_v11_confirmation.py','--root',str(root),
        '--table',str(base/'confirmation-table.json'),'--output',str(root/'runs/v11-fresh-controller'),'--lens-cli',str(a.lens_cli.resolve())],
        [base/'budget.json',base/'confirmation-table.json'],device='cpu',threads=1)
    state.update(state='submitted',finished_at=now());atomic_json(control/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
