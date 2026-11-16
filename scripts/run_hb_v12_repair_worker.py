#!/usr/bin/env python3
"""One explicit repair descendant; original input and failed records stay intact."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_hb_v12_worker import MechanismWorker
from tide_lab.records import atomic_json,digest,now


class RepairWorker(MechanismWorker):
    def run(self):
        case=self.plan['cases'][0];out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
        row=dict(id=case['id'],state='running',original_run=case['original_run'],stages=[],started_at=now())
        self.state['cases'].append(row);self.publish();self.guard();c=json.loads(Path(case['config']).read_text())
        def require(name,action,cap=1200,extra=(),record=False,script='scripts/hb_v12_repair_stage.py'):
            r=self.stage(case,name,action,cap,extra,record,script)
            if not r or r.get('state')!='completed':raise RuntimeError('repair stage failed: '+name)
            return r
        def retire(paths,label):
            atomic_json(out/(label+'-retirement.json'),dict(at=now(),reason='successful disposable verification or superseded repair state; original failed parent retained',
                files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths]))
            for p in paths:p.unlink()
        checked=require('check','check',script='scripts/campaign_stage.py')
        if not checked.get('admitted_for_finite_exploration'):raise ValueError('new source admission failed')
        migrated=require('migration','migrate',extra=['--resume',case['parent_checkpoint'],'--parent-config',case['parent_config'],
            '--parent-source',case['parent_source'],'--parent-sha256',case['parent_sha256'],'--qualification',self.plan['qualification']])
        row['recovered_parent_progress']=migrated['progress'];self.publish();parent=out/'migration/checkpoint.pt'
        require('recovery-full','segment',extra=['--resume',parent,'--steps','2'])
        require('recovery-one','segment',extra=['--resume',parent,'--steps','1'])
        require('recovery-resumed','segment',extra=['--resume',out/'recovery-one/checkpoint.pt','--steps','1'])
        require('recovery-compare','compare',extra=['--checkpoint-a',out/'recovery-full/checkpoint.pt','--checkpoint-b',out/'recovery-resumed/checkpoint.pt'])
        recovery_record=out/'recovery-compare/recovery.json'
        retire([out/k/'checkpoint.pt' for k in ('recovery-full','recovery-one','recovery-resumed')],'recovery')
        prefix=require('prefix','prefix',1800,['--resume',parent,'--steps','64','--run-id','tl-v12-repair-prefix-'+case['id']],True)
        projection=json.loads((out/'prefix/projection.json').read_text())
        if projection.get('status')!='verified' or not projection.get('readback_matches'):raise RuntimeError('prefix Lens readback failed')
        row['prefix_progress']=prefix['progress'];self.publish();retire([parent],'migration');parent=out/'prefix/latest.pt'
        if prefix['progress']['tokens']<8*2**20:
            case['train_arguments']=['--until-tokens',str(8*2**20),'--run-id','tl-v12-repair-pilot-'+case['id'],'--recording-group',c['study_id']]
            pilot=require('pilot','train',7200,['--resume',parent,'--admission',out/'check/result.json','--recovery',out/'recovery-compare/recovery.json'],True,'scripts/campaign_stage.py')
            initial=json.loads((Path(case['original_run'])/'pilot/validation-initial.json').read_text())['nll']
            rows=[json.loads(s)['metrics'] for s in (out/'pilot/metrics.jsonl').read_text().splitlines()]
            rates=[v['train/tokens_per_second'] for v in rows if 'train/tokens_per_second' in v]
            activity=[v['activity/full_all_fraction'] for v in rows if 'activity/full_all_fraction' in v]
            peak=max([v.get('resources/npu_peak_allocated_mib',0) for v in rows]+[0])
            gate=dict(state='passed',initial_nll=initial,pilot_nll=pilot['final_validation_nll'],peak_mib=peak,last_full_all=activity[-1] if activity else None,
                median_tail_rate=sorted(rates[-8:])[len(rates[-8:])//2] if rates else None,scope='original8Mi learning/resource gate after infrastructure interruption')
            if pilot['progress']['tokens']!=8*2**20 or initial-gate['pilot_nll']<.1 or peak>56000 or not activity or not .005<activity[-1]<.20 or not rates or gate['median_tail_rate']<1000:gate['state']='failed'
            atomic_json(out/'pilot-gate.json',gate)
            if gate['state']!='passed':raise RuntimeError('original pilot gate failed after repair')
            parent=out/'pilot/latest.pt'
            require('postpilot-full','segment',extra=['--resume',parent,'--steps','2'])
            require('postpilot-one','segment',extra=['--resume',parent,'--steps','1'])
            require('postpilot-resumed','segment',extra=['--resume',out/'postpilot-one/checkpoint.pt','--steps','1'])
            require('postpilot-compare','compare',extra=['--checkpoint-a',out/'postpilot-full/checkpoint.pt','--checkpoint-b',out/'postpilot-resumed/checkpoint.pt'])
            recovery_record=out/'postpilot-compare/recovery.json'
            retire([out/k/'checkpoint.pt' for k in ('postpilot-full','postpilot-one','postpilot-resumed')],'postpilot')
        case['train_arguments']=['--until-tokens',str(128*2**20),'--run-id','tl-v12-repair-main-'+case['id'],'--recording-group',c['study_id']]
        final=require('train','train',self.a.deadline_epoch-time.time()-1800,['--resume',parent,'--admission',out/'check/result.json',
            '--recovery',recovery_record],True,'scripts/campaign_stage.py')
        if final['progress']['tokens']!=128*2**20:raise RuntimeError('original128Mi endpoint not reached')
        import torch
        saved=torch.load(out/'train/latest.pt',map_location='cpu',weights_only=False)
        if saved['progress']['tokens']!=128*2**20 or 'optimizer' not in saved:raise ValueError('complete final recovery artifact missing')
        del saved
        retire([p for p in (out/'prefix/latest.pt',out/'pilot/latest.pt') if p.exists()],'superseded')
        original_data=self.a.data;self.a.data=Path(self.plan['confirmation_data'])
        try:
            require('confirmation',None,1200,['--training-data',original_data,'--checkpoint',out/'train/latest.pt','--lens-cli',self.a.recording_cli,
                '--run-id','tl-v12-repair-confirm-'+case['id'],'--recording-group','TL-HB-CONFIRM-v12'],script='scripts/evaluate_capacity_v11.py')
        finally:self.a.data=original_data
        table=out/'diagnostic-table.json';atomic_json(table,dict(cases=[dict(id=case['id'],config=case['config'],
            checkpoint=str(out/'train/latest.pt'),sha256=digest(out/'train/latest.pt'),history=dict(repair_parent=case['parent_checkpoint']))]))
        require('diagnostic',None,1800,['--table',table,'--root',self.plan['root'],'--lens-cli',self.a.recording_cli,
            '--run-id','tl-v12-repair-diag-'+case['id'],'--recording-group','TL-HB-DIAG-v12'],script='scripts/diagnose_hb_v12.py')
        row.update(state='completed',finished_at=now());self.state.update(state='completed',finished_at=now());self.publish()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','data','output','recording-cli'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();w=RepairWorker(a)
    try:w.run()
    except BaseException as e:
        if w.state['cases']:w.state['cases'][-1].update(state='failed',error=repr(e),finished_at=now())
        w.state.update(state='failed',error=repr(e),finished_at=now());w.publish();raise
