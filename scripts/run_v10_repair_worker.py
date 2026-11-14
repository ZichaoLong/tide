#!/usr/bin/env python3
"""One authorized pool repair chain under the existing durable worker contract."""
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


class RepairWorker(Worker):
    def __init__(self,a):
        super().__init__(a);self.last_guard=0
        if len(self.plan['cases'])!=1:raise ValueError('one fixed repair case per service')

    def guard(self):
        if time.time()>=self.a.deadline_epoch-60:raise TimeoutError('original v10 shared deadline')
        if time.monotonic()-self.last_guard<120:return
        self.last_guard=time.monotonic();root=Path(self.plan['root'])
        def disk(pattern):
            paths=[p for base in ('runs','artifacts/jobs') for p in (root/base).glob(pattern)]
            if not paths:return 0.
            raw=subprocess.check_output(['du','-s','-B1',*map(str,paths)],text=True)
            return sum(int(x.split()[0]) for x in raw.splitlines())/2**30
        quota=account_quota();usage=dict(at=now(),v10_disk_gib=disk('v10-*'),
                                       repair_disk_gib=disk('v10-repair-*'),quota=quota)
        self.state['resources']=usage;self.publish()
        if usage['v10_disk_gib']>96 or usage['repair_disk_gib']>24 or quota['hard_quota_headroom_gib']<200:
            raise RuntimeError('shared campaign/repair disk or account headroom guard')

    def run(self):
        case=self.plan['cases'][0];out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
        row=dict(id=case['id'],output=str(out),state='running',stages=[],started_at=now(),
                 original_run=case['original_run'],migration='explicit_new_source_identity')
        self.state['cases'].append(row);self.publish();self.guard()
        script='scripts/v10_repair_stage.py'
        def require(name,action,cap,extra=(),record=False,program=script):
            if self.stopped:raise InterruptedError('repair stop requested')
            result=self.stage(case,name,action,cap,extra=extra,record=record,script=program)
            if not result or result['state']!='completed':raise RuntimeError('required stage not completed: '+name)
            return result
        require('witness','witness',300,['--witness',case['witness']])
        checked=require('check','check',600,program='scripts/campaign_stage.py')
        if not checked.get('admitted_for_finite_exploration'):raise RuntimeError('new source admission rejected')
        require('migration','migrate',600,extra=['--resume',case['parent_checkpoint'],
            '--parent-config',case['parent_config'],'--parent-source',case['parent_source'],
            '--parent-diagnosis',case['parent_diagnosis']])
        parent=out/'migration/checkpoint.pt'
        require('recovery-full','segment',300,['--resume',parent,'--steps','2'])
        require('recovery-one','segment',300,['--resume',parent,'--steps','1'])
        require('recovery-resumed','segment',300,['--resume',out/'recovery-one/checkpoint.pt','--steps','1'])
        require('recovery-compare','compare',300,['--checkpoint-a',out/'recovery-full/checkpoint.pt',
            '--checkpoint-b',out/'recovery-resumed/checkpoint.pt'])
        disposable=[out/p/'checkpoint.pt' for p in ('recovery-full','recovery-one','recovery-resumed')]
        atomic_json(out/'recovery-retirement.json',dict(at=now(),
            reason='Exact comparison passed. Verification-only payloads retired; hashes/results retained, no future byte availability promised.',
            files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in disposable]))
        for p in disposable:p.unlink()
        pilot=require('pilot','pilot',900,['--resume',parent,'--steps','64',
            '--run-id','tl-v10-repair-pilot-'+case['id']],record=True)
        if pilot['updates']!=64 or pilot['progress']['probe']:raise RuntimeError('incomplete learning prefix')
        projection=json.loads((out/'pilot/projection.json').read_text())
        if projection.get('status')!='verified' or not projection.get('readback_matches'):
            raise RuntimeError('pilot Lens destination/readback not verified before long continuation')
        row['gates_passed_at']=now();row['learning_prefix']=pilot['progress'];self.publish()
        remaining=self.a.deadline_epoch-time.time()-900
        trained=require('train','train',remaining,extra=['--resume',out/'pilot/latest.pt',
            '--admission',out/'check/result.json','--recovery',out/'recovery-compare/recovery.json'],
            record=True,program='scripts/campaign_stage.py')
        if trained['progress']['tokens']!=128*2**20:raise RuntimeError('original 128M endpoint not reached')
        row['training_state']='completed';self.publish()
        require('confirmation',None,900,extra=['--checkpoint',out/'train/latest.pt',
            '--lens-cli',self.a.recording_cli,'--run-id','tl-v10-repair-confirm-'+case['id']],
            program='scripts/evaluate_direct_v10.py')
        row.update(state='completed',finished_at=now())
        self.state.update(state='completed',finished_at=now());self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','data','output','recording-cli'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();w=RepairWorker(a)
    try:w.run()
    except BaseException as e:
        if w.state['cases']:w.state['cases'][-1].update(state='stopped' if w.stopped else 'failed',error=repr(e),finished_at=now())
        w.state.update(state='stopped' if w.stopped else 'failed',error=repr(e),finished_at=now());w.publish();raise
