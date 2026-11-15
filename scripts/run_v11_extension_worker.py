#!/usr/bin/env python3
"""One bounded endpoint extension, with explicit migration and exact recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.campaign_worker import Worker
from tide_lab.records import atomic_json,digest,now
from tide_lab.quota import account_quota


class ExtensionWorker(Worker):
    def __init__(self,a):
        super().__init__(a)
        self.last_guard=0

    def guard(self):
        if time.time()>=self.a.deadline_epoch-60:
            raise TimeoutError('v11 shared deadline')
        if time.monotonic()-self.last_guard<60:return
        self.last_guard=time.monotonic();root=Path(self.plan['root'])
        paths=[p for base in ('runs','artifacts/jobs') for p in (root/base).glob('v11-*')]
        raw=subprocess.check_output(['du','-s','-B1',*map(str,paths)],text=True)
        disk=sum(int(s.split()[0]) for s in raw.splitlines())/2**30
        quota=account_quota()
        self.state['resources']=dict(at=now(),new_disk_gib=disk,quota=quota);self.publish()
        if disk>96 or quota['hard_quota_headroom_gib']<200:raise RuntimeError('v11 disk/headroom guard')

    def run(self):
        if len(self.plan['cases'])!=1:raise ValueError('one extension per service')
        case=self.plan['cases'][0];out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
        row=dict(id=case['id'],output=str(out),state='running',stages=[],started_at=now())
        self.state['cases'].append(row);self.publish();self.guard()
        script='scripts/capacity_v11_extension_stage.py'
        def require(name,action,cap,extra=(),record=False,program=script):
            r=self.stage(case,name,action,cap,extra=extra,record=record,script=program)
            if not r or r['state']!='completed':raise RuntimeError('stage incomplete: '+name)
            return r
        checked=require('check','check',900,program='scripts/campaign_stage.py')
        if not checked.get('admitted_for_finite_exploration'):raise RuntimeError('admission failed')
        require('migration','migrate',900,['--resume',case['parent_checkpoint'],
            '--parent-config',case['parent_config'],'--parent-source',case['parent_source']])
        parent=out/'migration/checkpoint.pt'
        require('recovery-full','segment',900,['--resume',parent,'--steps','2'])
        require('recovery-one','segment',900,['--resume',parent,'--steps','1'])
        require('recovery-resumed','segment',900,['--resume',out/'recovery-one/checkpoint.pt','--steps','1'])
        require('recovery-compare','compare',900,['--checkpoint-a',out/'recovery-full/checkpoint.pt',
            '--checkpoint-b',out/'recovery-resumed/checkpoint.pt'])
        disposable=[out/name/'checkpoint.pt' for name in ('recovery-full','recovery-one','recovery-resumed')]
        atomic_json(out/'recovery-retirement.json',dict(at=now(),reason='passed exact temporary verification; formal parent retained',
            files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in disposable]))
        for p in disposable:p.unlink()
        row['gates_passed_at']=now();self.publish()
        r=require('train','train',self.a.deadline_epoch-time.time()-900,
            ['--resume',parent,'--admission',out/'check/result.json','--recovery',out/'recovery-compare/recovery.json'],
            record=True,program='scripts/campaign_stage.py')
        if r['progress']['tokens']!=256*2**20:raise RuntimeError('requested256Mi endpoint not reached')
        require('confirmation-seen',None,900,['--checkpoint',out/'train/latest.pt','--lens-cli',self.a.recording_cli,
            '--run-id','tl-v11-extend-confirm-seen-'+case['id'],'--recording-group','TL-CAPACITY-CONFIRM-v11'],
            program='scripts/evaluate_direct_v10.py')
        row.update(state='completed',finished_at=now())
        self.state.update(state='completed',finished_at=now());self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','data','output','recording-cli'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();w=ExtensionWorker(a)
    try:w.run()
    except BaseException as e:
        if w.state['cases']:w.state['cases'][-1].update(state='failed',error=repr(e),finished_at=now())
        w.state.update(state='failed',error=repr(e),finished_at=now());w.publish();raise
