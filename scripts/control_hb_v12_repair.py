#!/usr/bin/env python3
"""At most one explicit repair per failed original v12 chain, same deadline."""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now
from tide_lab.supervisor import verify_source,TERMINAL
from tide_lab.quota import account_quota
from tide_lab.disk_usage import campaign_usage
from scripts.job import inspect


def read(path):return json.loads(path.read_text()) if path.is_file() else {}


def failure_class(root,name):
    out=root/'runs'/('v12-main-'+name)
    for stage in ('train','pilot','confirmation','diagnostic'):
        r=read(out/stage/'result.json')
        if 'allocation feasibility residual' in r.get('error',''):return 'quad_feasibility'
    p=root/'artifacts/jobs'/('v12-main-'+name)/'job.log'
    log=p.read_text() if p.exists() else ''
    if 'du: cannot access' in log and '.tmp' in log and 'No such file or directory' in log:return 'disk_scan_disappearance'
    return None


def main(a):
    root=a.root.resolve();source=Path(__file__).resolve().parents[1];sid=verify_source(source)
    base=source/'configs/studies/language/hb-v12';budget=read(base/'budget.json');deadline=budget['deadline_epoch']
    table=read(base/'table.json');a.output.mkdir(parents=True,exist_ok=False)
    if len(table['cases'])!=38:raise ValueError('original38 chains only')
    stopped=[];signal.signal(signal.SIGTERM,lambda *x:stopped.append(True));signal.signal(signal.SIGINT,lambda *x:stopped.append(True))
    state=dict(schema='tide-hb-v12-repair-controller-v1',state='waiting_qualification',source_commit=sid['commit'],
        started_at=now(),jobs={},unhandled={},budget=budget,scope='one repair per supported original failure; active originals untouched; no new candidates/seeds/exposure')
    publish=lambda:atomic_json(a.output/'state.json',state)
    publish();proof=root/'runs/v12-repair-qualification-npu/result.json'
    ready=False
    try:
        while time.time()<deadline-300 and not stopped:
            ready=True
            for dev in ('cpu','npu'):
                p=root/'runs'/('v12-repair-qualification-'+dev);r=read(p/'result.json');projection=read(p/'projection.json')
                if r.get('state')=='failed':raise RuntimeError(dev+' repair qualification failed')
                if r.get('state')!='completed' or projection.get('status')!='verified' or not projection.get('readback_matches'):
                    ready=False;continue
                for path,h in r['implementation_sha256'].items():
                    if digest(source/path)!=h:raise ValueError('qualified implementation changed: '+path)
            if ready:break
            time.sleep(5)
        if not ready or stopped:raise TimeoutError('qualification not available before stop/deadline')
        state['state']='running';state['qualification_sha256']=digest(proof);publish()
        while time.time()<deadline-180 and not stopped:
            quota=account_quota();disk=campaign_usage(root);state['resources']=dict(at=now(),quota=quota,disk=disk)
            if quota['hard_quota_headroom_gib']<200 or disk['gib']>96:raise RuntimeError('original disk/headroom limit')
            original_terminal=0
            for case in table['cases']:
                name=case['id'];original=root/'artifacts/jobs'/('v12-main-'+name)/'job.json';obs=inspect(original)
                lifecycle=obs['state']['state']
                if lifecycle in TERMINAL and obs.get('termination_verified_now'):original_terminal+=1
                if lifecycle!='failed' or not obs.get('termination_verified_now') or name in state['jobs']:continue
                cause=failure_class(root,name)
                if not cause:
                    state['unhandled'][name]='unsupported failure; no automatic retry';continue
                original_run=root/'runs'/('v12-main-'+name)
                parent=next((original_run/p/'latest.pt' for p in ('train','pilot') if (original_run/p/'latest.pt').exists()),None)
                if parent is None:
                    state['unhandled'][name]='no complete optimizer checkpoint; no guessed continuation';continue
                if time.time()>deadline-3600:
                    state['unhandled'][name]='less than1h remaining; no new repair submission';continue
                cfg=source/case['config'];config=read(cfg);old_cfg=root/'.local/source-hb-v12-main'/case['config']
                if digest(cfg)!=digest(old_cfg):raise ValueError('repair config changed')
                job='v12-repair-'+name;dest=a.output/name;dest.mkdir()
                policy=read(base/'resource.json')
                policy['minimum_free_hbm_mib']=min(56000,max(16384,int(config['resource_admission']['peak_mib']*1.15+8192)))
                policy['latest_authorization']='2026-11-16 user requested repair of original v12 failures'
                rp=dest/'resource.json';atomic_json(rp,policy);rp.chmod(0o444)
                plan=dict(root=str(root),confirmation_data=str(root/'runs/v12-data'),qualification=str(proof),cases=[dict(
                    id=name,config=str(cfg),output=str(root/'runs'/job),original_run=str(original_run),parent_checkpoint=str(parent),
                    parent_config=str(old_cfg),parent_source=str(root/'.local/source-hb-v12-main'),parent_sha256=digest(parent),
                    original_failure=cause,max_rss_gib=64)])
                pp=dest/'plan.json';atomic_json(pp,plan);pp.chmod(0o444);cap=int(deadline-time.time()-30)
                cmd=[sys.executable,str(source/'scripts/job.py'),'submit','--name',job,'--source',str(source),'--output',str(root/'artifacts/jobs'/job),
                    '--env-script',str(source/'scripts/env.sh'),'--env','TIDE_LAB_VENV='+str(root/'.venv-positive-floor-v1'),'--device','npu',
                    '--threads','4','--memory-gib','72','--timeout',str(cap),'--env','TIDE_LAB_RESOURCE_POLICY='+str(rp),
                    '--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(cap-300)]
                for p in [cfg,pp,rp,proof,parent,base/'budget.json',root/'runs/v9-data-r3/data-lock.json',root/'runs/v12-data/data-lock.json']:
                    cmd+=['--input',str(p)]
                cmd+=['--','python','scripts/run_hb_v12_repair_worker.py','--plan',str(pp),'--data',str(root/'runs/v9-data-r3'),
                    '--output',str(root/'runs'/(job+'-worker')),'--recording-cli',str(a.lens_cli),'--deadline-epoch',str(deadline)]
                state['jobs'][name]=dict(job=job,state='submitting',cause=cause,parent_checkpoint=str(parent),parent_sha256=plan['cases'][0]['parent_sha256'],at=now());publish()
                reply=subprocess.run(cmd,capture_output=True,text=True);atomic_json(dest/'submission.json',dict(argv=cmd,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr))
                if reply.returncode:raise RuntimeError('repair submission failed: '+name+' '+reply.stderr)
                state['jobs'][name].update(state='submitted',submitted_at=now());publish()
            terminal=0
            for name,row in state['jobs'].items():
                r=inspect(root/'artifacts/jobs'/row['job']/'job.json');row['state']=r['state']['state'];row['termination_verified']=r.get('termination_verified_now')
                if row['state'] in TERMINAL and row['termination_verified']:terminal+=1
            state.update(original_terminal=original_terminal,repair_terminal=terminal,last_observed_at=now());publish()
            if original_terminal==38 and terminal==len(state['jobs']):
                state['state']='completed';break
            time.sleep(30)
        if state['state']!='completed':state['state']='stopped' if stopped else 'budget_truncated'
    except BaseException as e:state.update(state='failed',error=repr(e));raise
    finally:state['finished_at']=now();publish()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output','lens-cli'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
