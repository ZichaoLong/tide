#!/usr/bin/env python3
"""One HB intervention trajectory: admission, recovery,8Mi pilot, recovery,128Mi."""
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_v11_extension_worker import ExtensionWorker
from tide_lab.records import atomic_json,digest,now

class HBWorker(ExtensionWorker):
    def run(self):
        case=self.plan['cases'][0];out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
        row=dict(id=case['id'],state='running',stages=[],started_at=now());self.state['cases'].append(row);self.publish();self.guard()
        def require(name,action,cap=900,extra=(),record=False,script='scripts/campaign_stage.py'):
            r=self.stage(case,name,action,cap,extra,record,script)
            if not r or r.get('state')!='completed':raise RuntimeError('stage failed: '+name)
            return r
        r=require('check','check')
        if not r.get('admitted_for_finite_exploration'):raise RuntimeError('admission failed')
        require('probe-full','probe',extra=['--steps','3'])
        require('probe-one','probe',extra=['--steps','1'])
        require('probe-resumed','probe',extra=['--steps','2','--resume',out/'probe-one/checkpoint.pt'])
        # The generic probe omits trained gradient signatures but compares every tensor/state.
        import torch
        from tide_lab.campaign_runtime import exact_tree
        left=torch.load(out/'probe-full/checkpoint.pt',map_location='cpu',weights_only=False)
        right=torch.load(out/'probe-resumed/checkpoint.pt',map_location='cpu',weights_only=False)
        exact_tree(left,right)
        recovery=dict(state='passed',identity=left['identity'],scope='three fresh-process B4/L32 initialization updates; trained-state recovery follows8Mi')
        atomic_json(out/'recovery.json',recovery);del left,right
        def retire(names):
            paths=[out/n/'checkpoint.pt' for n in names]
            atomic_json(out/(names[0]+'-retirement.json'),dict(reason='exact temporary verification passed; scientific parents retained',
                files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths]))
            for p in paths:p.unlink()
        retire(['probe-full','probe-one','probe-resumed'])
        case['train_arguments']=['--until-tokens',str(8*2**20),'--run-id','tl-v11-hb-pilot-'+case['id'],'--recording-group','TL-CAPACITY-HB-v11']
        pilot=require('pilot','train',5400,['--admission',out/'check/result.json','--recovery',out/'recovery.json'],True)
        initial=json.loads((out/'pilot/validation-initial.json').read_text());final=json.loads((out/'pilot/validation-final.json').read_text())
        rows=[json.loads(s)['metrics'] for s in (out/'pilot/metrics.jsonl').read_text().splitlines()]
        train=[r for r in rows if 'train/targets_per_second' in r or 'train/tokens_per_second' in r]
        # Resource observations are accepted independently of whether they use a particular throughput key.
        peak=max([r.get('resources/npu_peak_allocated_mib',0) for r in rows]+[0])
        activity=[r['activity/full_all_fraction'] for r in rows if 'activity/full_all_fraction' in r]
        bias=max([v for r in rows for k,v in r.items() if k.startswith('balance/') and k.endswith('/bias_abs_max')]+[0])
        speed_keys=['train/targets_per_second','train/tokens_per_second','train/targets_per_sec']
        speeds=[r[k] for r in rows for k in speed_keys if k in r]
        gate=dict(state='passed',initial_nll=initial['nll'],pilot_nll=final['nll'],peak_mib=peak,
            last_full_all=activity[-1] if activity else None,max_bias=bias,
            measured_speed_values=speeds[-8:],scope='8Mi existing milestone; pilot not independent confirmation')
        if pilot['progress']['tokens']!=8*2**20 or initial['nll']-final['nll']<.1 or peak>56000 or not activity or not .005<activity[-1]<.15 or bias<=0 or not speeds or sorted(speeds[-8:])[len(speeds[-8:])//2]<1000:
            gate['state']='failed'
        atomic_json(out/'pilot-gate.json',gate)
        if gate['state']!='passed':raise RuntimeError('bounded pilot gate failed')
        parent=out/'pilot/latest.pt';script='scripts/capacity_v11_extension_stage.py'
        require('trained-full','segment',extra=['--resume',parent,'--steps','2'],script=script)
        require('trained-one','segment',extra=['--resume',parent,'--steps','1'],script=script)
        require('trained-resumed','segment',extra=['--resume',out/'trained-one/checkpoint.pt','--steps','1'],script=script)
        require('trained-compare','compare',extra=['--checkpoint-a',out/'trained-full/checkpoint.pt','--checkpoint-b',out/'trained-resumed/checkpoint.pt'],script=script)
        retire(['trained-full','trained-one','trained-resumed'])
        case['train_arguments']=['--until-tokens',str(128*2**20),'--run-id','tl-v11-hb-'+case['id'],'--recording-group','TL-CAPACITY-HB-v11']
        result=require('train','train',self.a.deadline_epoch-time.time()-900,
            ['--resume',parent,'--admission',out/'check/result.json','--recovery',out/'trained-compare/recovery.json'],True)
        if result['progress']['tokens']!=128*2**20:raise RuntimeError('128Mi endpoint missing')
        row.update(state='completed',finished_at=now());self.state.update(state='completed',finished_at=now());self.publish()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','data','output','recording-cli'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();w=HBWorker(a)
    try:w.run()
    except BaseException as e:
        if w.state['cases']:w.state['cases'][-1].update(state='failed',error=repr(e),finished_at=now())
        w.state.update(state='failed',error=repr(e),finished_at=now());w.publish();raise
