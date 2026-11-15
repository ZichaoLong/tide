#!/usr/bin/env python3
"""Read-only v11 status; lifecycle, requested exposure and scientific status differ."""
import argparse,json,subprocess,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_io import write_review_json
from tide_lab.records import now
from tide_lab.quota import account_quota

def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}

def main(a):
    root=Path(__file__).resolve().parents[1];base=root/'configs/studies/language/capacity-v11';rows=[]
    specs=[]
    for c in read(base/'moe-table.json')['cases']:
        job=('v11-extend-r2-' if c['phase']=='moe-extend' else 'v11-moe-balance-')+c['id']
        specs.append((job,c['until_tokens'],c['group']))
    for c in read(base/'hb-table.json')['cases']:specs.append(('v11-hb-'+c['id'],128*2**20,'TL-CAPACITY-HB-v11'))
    for job,target,group in specs:
        worker=read(root/'runs'/(job+'-worker')/'state.json');artifact=root/'artifacts/jobs'/job
        jobstate=read(artifact/'state.json');queue=read(artifact/'queue-state.json');unit=read(artifact/'job.json').get('unit')
        cases=worker.get('cases',[]);stages=cases[-1].get('stages',[]) if cases else []
        stage=stages[-1]['name'] if stages else 'waiting';path=root/'runs'/job/stage
        result=read(path/'result.json');progress=read(path/'progress.json') or result.get('progress',{})
        exposure=progress.get('tokens',result.get('exposure'))
        if exposure is None:exposure=read(root/'runs'/job/'train/progress.json').get('tokens')
        row=dict(job=job,study_id=group,lifecycle=jobstate.get('state','unsubmitted'),
            worker=worker.get('state'),stage=stage,stage_state=result.get('state'),exposure=exposure,requested_tokens=target,
            physical_device=worker.get('physical_visibility'),source_commit=read(artifact/'job.json').get('source',{}).get('commit'),
            endpoint='attained' if read(root/'runs'/job/'train/result.json').get('progress',{}).get('tokens')==target else 'pending_or_failed',
            science='unreviewed',error=worker.get('error'),run='tl-'+job,projection=str((root/'runs'/job/'train/projection.json').relative_to(root)))
        if unit:
            r=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','Slice','-p','ControlGroup','-p','MainPID'],capture_output=True,text=True)
            row['systemd']=dict(s.split('=',1) for s in r.stdout.splitlines() if '=' in s)
        rows.append(row)
    originals=[]
    for c in read(base/'moe-table.json')['cases']:
        if c['phase']!='moe-extend':continue
        job='v11-moe-extend-'+c['id'];r=read(root/'runs'/job/'train/result.json')
        originals.append(dict(job=job,inner_state=r.get('state'),tokens=r.get('progress',{}).get('tokens'),requested_tokens=c['until_tokens'],
            assessment='request_not_fulfilled; old config target128Mi; zero new updates'))
    paths=[p for b in ('runs','artifacts/jobs') for p in (root/b).glob('v11-*')]+list((root/'.local').glob('source-capacity-v11-*'))
    raw=subprocess.check_output(['du','-s','-B1',*map(str,paths)],text=True)
    evidence=dict(schema='tide-capacity-v11-status-v1',observed_at=now(),budget=read(base/'budget.json'),quota=account_quota(),
        allocated_new_gib=sum(int(line.split()[0]) for line in raw.splitlines())/2**30,training=rows,
        original_no_progress_requests=originals,diagnostics=read(root/'docs/evidence/capacity-v11-diagnostics.json'),
        fresh_data=read(root/'runs/v11-confirmation-data/state.json'),fresh_controller=read(root/'runs/v11-fresh-controller/state.json'),
        limitation='Live read-only snapshot; recorded progress is not completed independent quality review; original thresholds and failures retained.')
    if a.output:write_review_json(a.output,evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('diagnostics','fresh_controller')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);main(p.parse_args())
