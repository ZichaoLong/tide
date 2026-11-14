#!/usr/bin/env python3
"""Resolve the three authorized repair inputs and submit immutable durable chains."""
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
    root=a.root.resolve();source=a.source.resolve();source_id=verify_source(source)
    manifest=source/'configs/studies/language/direct-v10/repair.json'
    spec=json.loads(manifest.read_text());deadline=spec['deadline_epoch']
    parent_source=root/spec['parent_source'];parent_id=verify_source(parent_source)
    if parent_id['commit']!=spec['parent_commit']:raise ValueError('wrong historical source')
    if time.time()>deadline-3600:raise TimeoutError('less than one hour remains in original budget')
    if account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('account headroom below200GiB')
    # Validate all pre-existing parents before creating any new records.
    for case in spec['cases']:
        for item in case['inputs'].values():
            if digest(root/item['path'])!=item['sha256']:raise ValueError('fixed parent mismatch: '+item['path'])
        if (root/'runs'/('v10-repair-'+case['id'])).exists():raise FileExistsError('repair case already exists')
    control=root/'runs/v10-repair-control';control.mkdir(parents=True,exist_ok=False)
    state=dict(schema='tide-v10-repair-submission-v1',state='submitting',created_at=now(),
               source_commit=source_id['commit'],deadline_epoch=deadline,jobs=[])
    atomic_json(control/'state.json',state)
    try:
        for case in spec['cases']:
            cid=case['id'];job='v10-repair-'+cid;inputs=case['inputs']
            parent_config=root/inputs['config']['path']
            config=json.loads(parent_config.read_text());config.setdefault('implementation',{})['aggregate_vjp']='ordered-v1'
            dest=control/cid;dest.mkdir()
            resolved=dest/'config.json';atomic_json(resolved,config);resolved.chmod(0o444)
            policy=copy.deepcopy(json.loads((parent_source/'configs/studies/language/direct-v10/resource.json').read_text()))
            policy['minimum_free_hbm_mib']=min(56000,max(12288,int(config['resource_admission']['peak_mib']*1.15+8192)))
            resource=dest/'resource.json';atomic_json(resource,policy);resource.chmod(0o444)
            row=dict(id=cid,config=str(resolved),output=str(root/'runs'/job),max_rss_gib=64,
                original_run='tl-v10-main-'+cid,parent_config=str(parent_config),parent_source=str(parent_source),
                parent_checkpoint=str(root/inputs['checkpoint']['path']),
                parent_diagnosis=str(root/inputs['diagnosis']['path']),witness=str(root/inputs['witness']['path']),
                train_arguments=['--until-tokens',str(spec['target_tokens']),'--run-id','tl-'+job,
                                 '--recording-group','TL-CONTROL-v10'])
            plan=dest/'worker.json';atomic_json(plan,dict(root=str(root),cases=[row]));plan.chmod(0o444)
            cap=int(deadline-time.time()-30)
            argv=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),
                '--output',str(root/'artifacts/jobs'/job),'--env-script',str(source/'scripts/env.sh'),
                '--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),
                '--env','TIDE_LAB_RESOURCE_POLICY='+str(resource),'--device','npu','--threads','4',
                '--memory-gib','72','--timeout',str(cap),'--wait-seconds',str(cap-120),
                '--queue-helper',str(source/'scripts/run_on_light_device.py')]
            for p in [manifest,resolved,plan,resource,a.data.resolve()/'data-lock.json']:
                argv+=['--input',str(p)]
            for item in inputs.values():argv+=['--input',str(root/item['path'])]
            argv+=['--','python','scripts/run_v10_repair_worker.py','--plan',str(plan),
                '--data',str(a.data.resolve()),'--output',str(root/'runs'/(job+'-worker')),
                '--recording-cli',str(a.lens_cli.resolve()),'--deadline-epoch',str(deadline)]
            entry=dict(job=job,id=cid,config=str(resolved),config_sha256=digest(resolved),plan=str(plan),
                       argv=argv,state='submitting');state['jobs'].append(entry);atomic_json(control/'state.json',state)
            result=subprocess.run(argv,capture_output=True,text=True)
            atomic_json(dest/'submission.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:raise RuntimeError(result.stderr+' '+result.stdout)
            entry.update(state='submitted',submitted_at=now());atomic_json(control/'state.json',state)
            print(json.dumps(dict(submitted=job,source_commit=source_id['commit'])),flush=True)
        state.update(state='submitted',finished_at=now())
    except BaseException as error:
        state.update(state='failed',error=repr(error),finished_at=now());raise
    finally:atomic_json(control/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','data','lens-cli'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
