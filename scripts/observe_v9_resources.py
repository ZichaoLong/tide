#!/usr/bin/env python3
"""Sample existing v9 services and device counters without touching their inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import now
from tide_lab.supervisor import cgroup_processes

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text()) if p.exists() else {}


def active(campaign='v9'):
    result=[]
    for p in sorted((ROOT/'runs').glob(campaign+'-main-*-worker/state.json')):
        w=read(p)
        if w.get('state')!='running':continue
        job=p.parent.name.removesuffix('-worker')
        spec=read(ROOT/'artifacts/jobs'/job/'job.json')
        queue=read(ROOT/'artifacts/jobs'/job/'queue-state.json')
        c=w['cases'][-1];stages=c.get('stages',[]);stage=stages[-1] if stages else {}
        r=read(Path(c['output'])/'train/progress.json')
        info=subprocess.check_output(['systemctl','--user','show',spec['unit'],'-p','ControlGroup','-p','CPUQuotaPerSecUSec'],text=True)
        fields=dict(line.split('=',1) for line in info.splitlines() if '=' in line)
        processes=[]
        for child in cgroup_processes(fields['ControlGroup']):
            try:
                pid=child['pid'];raw=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
                cpu_group=[s for s in Path('/proc',str(pid),'cgroup').read_text().splitlines() if ':cpu,' in s or ',cpu:' in s]
                processes.append(dict(pid=pid,start_ticks=int(raw[19]),cpu_seconds=(int(raw[11])+int(raw[12]))/os.sysconf('SC_CLK_TCK'),cpu_controller=cpu_group))
            except (OSError,ValueError,IndexError):pass
        cfg=next((v for v in spec['inputs'] if v['path'].endswith('/'+c['id']+'.json')),None)
        result.append(dict(job=job,case=c['id'],physical_id=queue.get('physical_id'),stage=stage.get('name'),
            source_commit=spec['source']['commit'],config=cfg,training_tokens=r.get('tokens'),step=r.get('step'),
            last_update=r.get('last_update'),host_processes=processes,
            declared_cpu_quota=fields['CPUQuotaPerSecUSec']))
    return result


def usages(physical):
    argv=['npu-smi','info','-t','usages','-i',str(physical//2),'-c',str(physical%2)]
    raw=subprocess.check_output(argv,text=True,timeout=15);metrics={}
    for line in raw.splitlines():
        if ':' not in line:continue
        key,value=line.split(':',1)
        try:metrics[key.strip()]=float(value.strip())
        except ValueError:pass
    return dict(physical_id=physical,observed_at=now(),command=argv,metrics=metrics)


def main(a):
    mapping=subprocess.check_output(['npu-smi','info','-m'],text=True)
    for line in mapping.splitlines():
        fields=line.split()
        if len(fields)==5 and fields[3].isdigit():
            if int(fields[3])!=2*int(fields[0])+int(fields[1]):raise ValueError('physical mapping changed')
    rows=[];started=time.monotonic()
    for i in range(a.samples):
        jobs=active(a.campaign);ids=sorted({j['physical_id'] for j in jobs if j['physical_id'] is not None})
        with ThreadPoolExecutor(max_workers=8) as pool:devices=list(pool.map(usages,ids))
        rows.append(dict(index=i,at=now(),elapsed_seconds=time.monotonic()-started,jobs=jobs,devices=devices))
        print(json.dumps(dict(sample=i,devices=len(devices))),flush=True)
        if i+1<a.samples:time.sleep(a.interval)
    keys=('Aicore Usage Rate(%)','Aivector Usage Rate(%)','Aicpu Usage Rate(%)',
          'Ctrlcpu Usage Rate(%)','HBM Bandwidth Usage Rate(%)','NPU Utilization(%)')
    stats=[]
    for phy in sorted({d['physical_id'] for r in rows for d in r['devices']}):
        samples=[d['metrics'] for r in rows for d in r['devices'] if d['physical_id']==phy]
        jobs=[j for r in rows for j in r['jobs'] if j['physical_id']==phy]
        summary={k:dict(min=min(s[k] for s in samples),median=statistics.median(s[k] for s in samples),
                       max=max(s[k] for s in samples)) for k in keys if all(k in s for s in samples)}
        first,last=jobs[0],jobs[-1];duration=rows[-1]['elapsed_seconds']-rows[0]['elapsed_seconds']
        before={(p['pid'],p['start_ticks']):p['cpu_seconds'] for p in first['host_processes']}
        after={(p['pid'],p['start_ticks']):p['cpu_seconds'] for p in last['host_processes']}
        delta=sum(after[k]-before[k] for k in before.keys()&after.keys())
        stats.append(dict(physical_id=phy,cases=sorted({j['case'] for j in jobs}),metrics=summary,
            host_cpu=dict(declared_cpu_quota=last['declared_cpu_quota'],average_cores=delta/max(duration,1e-6),
                changed_process_set=before.keys()!=after.keys(),scope='CPU seconds of persistent service-owned PIDs; shared v1 CPU controller is not per-unit attribution')))
    write_review_json(a.output,dict(schema='tide-'+a.campaign+'-live-resource-observation-v1',observed_at=now(),
        elapsed_seconds=time.monotonic()-started,samples=rows,summary=stats,
        scope='Device-level sampled counters and existing service-owned process CPU deltas; no profiler, model execution, allocation or active-input mutation.',
        limits=['Shared-card counters are not process-attributed; other workloads may contribute.',
                'The percentages have different denominators and must not be summed or interpreted as kernel-time fractions.',
                'Source identity is recorded for each service; independent direct models do not certify foundation execution.',
                'Kernel/operator attribution and causal bottleneck diagnosis require a separate bounded trace.']))
    print(json.dumps(stats),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=8)
    p.add_argument('--campaign',choices=('v9','v10'),default='v9')
    p.add_argument('--interval',type=float,default=4)
    p.add_argument('--output',type=Path,default=Path('docs/evidence/direct-v9-resources-20261112.json'))
    main(p.parse_args())
