#!/usr/bin/env python3
"""Submit six endpoint-only migrations, preserving the original no-progress attempts."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import verify_source
from tide_lab.quota import account_quota


def main(a):
    root=a.root.resolve();source=a.source.resolve();sid=verify_source(source)
    base=source/'configs/studies/language/capacity-v11'
    table=json.loads((base/'moe-table.json').read_text());budget=json.loads((base/'budget.json').read_text())
    deadline=budget['deadline_epoch'];cases=[c for c in table['cases'] if c['phase']=='moe-extend']
    if len(cases)!=6 or time.time()>deadline-3600:raise ValueError('invalid extension set/budget')
    if account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('quota headroom')
    for c in cases:
        if digest(root/c['config'])!=c['config_sha256']:raise ValueError('config drift')
        if digest(root/c['parent']/'train/latest.pt')!=c['parent_checkpoint_sha256']:raise ValueError('parent drift')
    out=root/'runs/v11-extension-r2-control';out.mkdir(parents=True,exist_ok=False)
    state=dict(schema='tide-v11-extension-submission-v1',state='submitting',source=sid['commit'],jobs=[],started_at=now())
    atomic_json(out/'state.json',state)
    try:
        for c in cases:
            cid=c['id'];job='v11-extend-r2-'+cid;dest=out/cid;dest.mkdir()
            config=json.loads((root/c['config']).read_text());config['training']['target_tokens']=256*2**20
            config['training']['milestones'].append(256*2**20)
            cp=dest/'config.json';atomic_json(cp,config);cp.chmod(0o444)
            policy=json.loads((base/'resource.json').read_text());policy['minimum_free_hbm_mib']=20000
            rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
            row=dict(id=cid,config=str(cp),output=str(root/'runs'/job),max_rss_gib=64,
                parent_config=str(root/c['config']),parent_source=str(root/c['execution_source']),
                parent_checkpoint=str(root/c['parent']/'train/latest.pt'),
                train_arguments=['--until-tokens',str(256*2**20),'--run-id','tl-'+job,
                                 '--recording-group','TL-CAPACITY-EXPOSURE-v11'])
            plan=dest/'worker.json';atomic_json(plan,dict(root=str(root),cases=[row]));plan.chmod(0o444)
            cap=int(deadline-time.time()-30)
            argv=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
                '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
                '--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),
                '--env','TIDE_LAB_RESOURCE_POLICY='+str(rp),'--device','npu','--threads','4',
                '--memory-gib','72','--timeout',str(cap),'--wait-seconds',str(cap-120),
                '--queue-helper',str(source/'scripts/run_on_light_device.py')]
            for p in (base/'budget.json',base/'moe-table.json',cp,rp,plan,a.data.resolve()/'data-lock.json',root/c['config'],root/c['parent']/'train/latest.pt'):
                argv+=['--input',str(p)]
            argv+=['--','python','scripts/run_v11_extension_worker.py','--plan',str(plan),'--data',str(a.data.resolve()),
                '--output',str(root/'runs'/(job+'-worker')),'--recording-cli',str(a.lens_cli.resolve()),'--deadline-epoch',str(deadline)]
            entry=dict(job=job,config=str(cp),plan=str(plan),argv=argv,state='submitting');state['jobs'].append(entry)
            atomic_json(out/'state.json',state)
            r=subprocess.run(argv,capture_output=True,text=True);atomic_json(dest/'submission.json',dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError(r.stderr+' '+r.stdout)
            entry.update(state='submitted',at=now());atomic_json(out/'state.json',state)
            print(job,flush=True)
        state['state']='submitted'
    except BaseException as e:state.update(state='failed',error=repr(e));raise
    finally:atomic_json(out/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','source','data','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
