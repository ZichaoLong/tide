#!/usr/bin/env python3
"""Re-evaluate fixed, already-trained controls on v12's unused confirmation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,now
from tide_lab.quota import account_quota

def main(a):
    root=a.root.resolve();cases=json.loads(a.table.read_text())['cases'];a.output.mkdir(parents=True,exist_ok=False)
    state=dict(state='running',started_at=now(),cases=[],scope='fixed historical128Mi controls, fresh confirmation only; no optimization/selection/test access')
    atomic_json(a.output/'state.json',state)
    try:
        for c in cases:
            if time.time()>a.deadline_epoch-300 or account_quota()['hard_quota_headroom_gib']<200:raise RuntimeError('control deadline/headroom')
            result=json.loads((root/c['training_result']).read_text())
            if result.get('state')!='completed' or result.get('progress',{}).get('tokens')!=128*2**20:raise ValueError('historical endpoint missing: '+c['id'])
            out=root/'runs'/('v12-control-'+c['id']);row=dict(id=c['id'],state='running',output=str(out));state['cases'].append(row)
            atomic_json(a.output/'state.json',state)
            cmd=[sys.executable,'scripts/evaluate_capacity_v11.py','--config',str(root/c['config']),'--checkpoint',str(root/c['checkpoint']),
                '--data',str(root/'runs/v12-data'),'--training-data',str(root/'runs/v9-data-r3'),'--output',str(out),
                '--deadline-epoch',str(min(a.deadline_epoch,time.time()+1800)),'--lens-cli',a.lens_cli,
                '--run-id','tl-v12-control-'+c['id'],'--recording-group','TL-HB-CONTROL-v12']
            with (a.output/(c['id']+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1900)
            value=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
            row.update(state=value.get('state','failed') if r.returncode==0 else 'failed',error=value.get('error'),finished_at=now())
            atomic_json(a.output/'state.json',state)
        state.update(state='completed',finished_at=now(),scope=state['scope']+'; individual failures retained, scientific conclusions require review')
    except BaseException as e:state.update(state='failed',error=repr(e),finished_at=now());raise
    finally:atomic_json(a.output/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','table','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--lens-cli',required=True);main(p.parse_args())
