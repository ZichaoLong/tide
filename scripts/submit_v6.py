#!/usr/bin/env python3
"""Prepare immutable v6 worker plans and explicitly submit selected workers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest


def prepare(a):
    source=a.source.resolve();root=a.root.resolve();data=a.data.resolve()
    if json.loads((data/'state.json').read_text())['state']!='completed':raise ValueError('fixed final data is not complete')
    cfgroot=source/'configs/studies/language/v6/resolved';catalog=json.loads((cfgroot/'catalog.json').read_text())
    if not catalog['launchable']:raise ValueError('catalog not resolved')
    out=root/'.local/v6-submission';out.mkdir(parents=True,exist_ok=False)
    cover={t:[c for c in catalog['cases'] if c['id'].startswith('cover-'+t+'-')] for t in ('fc','hb')}
    ordered=[c for pair in zip(cover['fc'],cover['hb']) for c in pair]
    if len(ordered)!=182:raise ValueError('unexpected coverage count')
    groups=[(f'v6-cover-worker-{i:03d}',ordered[i::a.workers]) for i in range(a.workers)]
    groups += [('v6-'+c['id'],[c]) for c in catalog['cases'] if c['line']=='scale' or c['id'].startswith('cover-transformer-')]
    jobs=[]
    for name,cases in groups:
        if not cases:continue
        plan=dict(schema='tide-v6-worker-plan-v1',cases=[])
        device=cases[0]['device'];threads=cases[0]['threads']
        for row in cases:
            config=cfgroot/(row['id']+'.json');c=json.loads(config.read_text())
            plan['cases'].append(dict(id=c['id'],config=str(config),output=str(root/'runs'/('v6-'+c['id'])),
                max_rss_gib=c['resources']['max_rss_gib'],timeout_seconds=c['training']['timeout_seconds']))
        path=out/(name+'.json');atomic_json(path,plan);path.chmod(0o444)
        jobs.append(dict(name=name,plan=str(path),device=device,threads=threads,cases=[c['id'] for c in cases],
            case_caps=sum(c['timeout_seconds'] for c in plan['cases']),memory_gib=max(c['max_rss_gib'] for c in plan['cases'])+4))
    manifest=dict(schema='tide-v6-submission-v1',source=str(source),source_identity=json.loads((source/'.tide-source.json').read_text()),
        data=str(data),data_lock_sha256=digest(data/'data-lock.json'),cpu_coverage_workers=a.workers,jobs=jobs)
    atomic_json(out/'submission.json',manifest);(out/'submission.json').chmod(0o444)
    print(json.dumps(dict(prepared_jobs=len(jobs),cases=sum(len(j['cases']) for j in jobs))))


def submit(a):
    root=a.root.resolve();manifest=json.loads((root/'.local/v6-submission/submission.json').read_text())
    source=Path(manifest['source']);data=Path(manifest['data']);lib=root/'.local/native-build-v6'
    limits=source/'configs/studies/language/v6/budget.json';deadline=json.loads(limits.read_text())['deadline_epoch']
    if digest(data/'data-lock.json')!=manifest['data_lock_sha256']:raise ValueError('data lock changed')
    start,end=map(int,a.workers_range.split(':')) if a.workers_range else (0,0)
    for job in manifest['jobs']:
        name=job['name'];cpu_pool=name.startswith('v6-cover-worker-')
        selected=start<=int(name.rsplit('-',1)[1])<end if cpu_pool else a.individual
        if not selected:continue
        if (root/'artifacts/jobs'/name).exists():raise ValueError('submission exists; do not overwrite or silently retry '+name)
        npu=job['device'].startswith('npu');cap=min(job['case_caps']+600,int(deadline-time.time()))
        if cap<=120:raise ValueError('shared campaign deadline exhausted')
        cmd=[sys.executable,str(root/'scripts/job.py'),'submit','--name',name,'--source',str(source),
            '--output',str(root/'artifacts/jobs'/name),'--env-script',str(source/'scripts/env.sh'),
            '--env',f'TIDE_LAB_VENV={root}/.venv-positive-floor-v1','--env',f'TIDE_LAB_NATIVE_LIBRARY={lib}',
            '--device','npu' if npu else 'cpu','--timeout',str(cap),'--threads',str(job['threads']),
            '--memory-gib',str(job['memory_gib'])]
        for p in [Path(job['plan']),data/'data-lock.json',lib/'build-manifest.json',lib/'_tide_native.so',limits]:cmd+=['--input',str(p)]
        if npu:cmd+=['--queue-helper',str(source/'scripts/run_on_light_device.py'),'--wait-seconds',str(min(3600,cap-60)),
            '--env',f'TIDE_LAB_RESOURCE_POLICY={source}/configs/studies/language/v6/resource.json']
        cmd+=['--','python','scripts/campaign_worker.py','--plan',job['plan'],'--data',str(data),
            '--output',str(root/'runs'/(name+'-worker')),'--recording-cli',a.recording_cli,'--deadline-epoch',str(deadline)]
        reply=subprocess.run(cmd,text=True,capture_output=True)
        if reply.returncode:raise RuntimeError(name+': '+reply.stderr+' '+reply.stdout)
        print(json.dumps(dict(submitted=name,cases=job['cases'],device=job['device'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','submit'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source',type=Path);p.add_argument('--data',type=Path);p.add_argument('--workers',type=int,default=96)
    p.add_argument('--workers-range');p.add_argument('--individual',action='store_true')
    p.add_argument('--recording-cli',default='trackio-lens');a=p.parse_args()
    prepare(a) if a.action=='prepare' else submit(a)
