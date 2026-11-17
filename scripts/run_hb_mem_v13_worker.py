#!/usr/bin/env python3
"""One fixed HB comparison: checks, pilot, trained recovery,128Mi,confirm,diagnose."""
import argparse,json,subprocess,time,os,signal
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.campaign_worker import Worker
from tide_lab.records import atomic_json,digest,now
from tide_lab.quota import account_quota
from tide_lab.disk_usage import campaign_usage

M=2**20

class MemWorker(Worker):
    def __init__(self,a):super().__init__(a);self.last_guard=0
    def guard(self):
        if self.stopped:return
        if time.time()>self.a.deadline_epoch-120:
            self.state['resource_stop']='shared48h deadline';self.publish();self.stop(signal.SIGTERM,None)
            if self.child is None:raise TimeoutError('v13 deadline')
            return
        if time.monotonic()-self.last_guard<60:return
        self.last_guard=time.monotonic();root=Path(self.plan['root'])
        quota=account_quota();fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize/2**30
        cap=min(700.,quota['hard_gib']);disk=campaign_usage(root,pattern='v13-*')
        self.state['resources']=dict(at=now(),campaign_gib=disk['gib'],quota=quota,filesystem_free_gib=free,
            effective_cap_gib=cap,checkpoint_reserve_gib=32);self.publish()
        if min(cap-quota['used_gib'],free)<32:
            self.state['resource_stop']='less than32GiB simultaneous checkpoint reserve';self.publish()
            self.stop(signal.SIGTERM,None)
            if self.child is None:raise RuntimeError(self.state['resource_stop'])

    def run(self):
        case=self.plan['cases'][0];out=Path(case['output']);out.mkdir(parents=True,exist_ok=False)
        row=dict(id=case['id'],state='running',stages=[],started_at=now());self.state['cases'].append(row);self.publish();self.guard()
        def require(name,action,cap=1200,extra=(),record=False,script='scripts/campaign_stage.py'):
            r=self.stage(case,name,action,cap,extra,record,script)
            if not r or r.get('state')!='completed':raise RuntimeError('stage failed: '+name)
            return r
        def retire(paths,name):
            atomic_json(out/(name+'-retirement.json'),dict(at=now(),reason='temporary verification passed or pilot superseded by complete final optimizer plus retained8Mi weights',
                files=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths]))
            for p in paths:p.unlink()
        checked=require('check','check')
        if not checked.get('admitted_for_finite_exploration'):raise RuntimeError('admission failed')
        require('probe-full','probe',extra=['--steps','3']);require('probe-one','probe',extra=['--steps','1'])
        require('probe-resumed','probe',extra=['--steps','2','--resume',out/'probe-one/checkpoint.pt'])
        import torch
        from tide_lab.campaign_runtime import exact_tree
        left=torch.load(out/'probe-full/checkpoint.pt',map_location='cpu',weights_only=False)
        right=torch.load(out/'probe-resumed/checkpoint.pt',map_location='cpu',weights_only=False);exact_tree(left,right)
        atomic_json(out/'recovery.json',dict(state='passed',identity=left['identity'],scope='three fresh-process B4/L32 updates; model/Adam/gradients/RNG/control state'))
        del left,right
        retire([out/n/'checkpoint.pt' for n in ('probe-full','probe-one','probe-resumed')],'probe')
        c=json.loads(Path(case['config']).read_text());group=c['study_id']
        case['train_arguments']=['--until-tokens',str(8*M),'--run-id','tl-v13-pilot-'+case['id'],'--recording-group',group]
        pilot=require('pilot','train',7200,['--admission',out/'check/result.json','--recovery',out/'recovery.json'],True)
        initial=json.loads((out/'pilot/validation-initial.json').read_text());final=json.loads((out/'pilot/validation-final.json').read_text())
        rows=[json.loads(s)['metrics'] for s in (out/'pilot/metrics.jsonl').read_text().splitlines()]
        speeds=[r['train/tokens_per_second'] for r in rows if 'train/tokens_per_second' in r]
        activity=[r['activity/full_all_fraction'] for r in rows if 'activity/full_all_fraction' in r]
        peak=max([r.get('resources/npu_peak_allocated_mib',0) for r in rows]+[0])
        gate=dict(state='passed',initial_nll=initial['nll'],pilot_nll=final['nll'],peak_mib=peak,last_full_all=activity[-1] if activity else None,
            median_tail_rate=sorted(speeds[-8:])[len(speeds[-8:])//2] if speeds else None,
            scope='8Mi finite-learning/resource gate; not candidate ranking or independent confirmation')
        activity_ok=bool(activity) and ((.99<=activity[-1]<=1.) if c['mem_comparison']['route']=='pool' else (.005<activity[-1]<.20))
        if pilot['progress']['tokens']!=8*M or initial['nll']-final['nll']<.1 or peak>56000 or not activity_ok or not speeds or gate['median_tail_rate']<1000:gate['state']='failed'
        atomic_json(out/'pilot-gate.json',gate)
        if gate['state']!='passed':raise RuntimeError('pilot gate failed')
        parent=out/'pilot/latest.pt';script='scripts/capacity_v11_extension_stage.py'
        require('trained-full','segment',extra=['--resume',parent,'--steps','2'],script=script)
        require('trained-one','segment',extra=['--resume',parent,'--steps','1'],script=script)
        require('trained-resumed','segment',extra=['--resume',out/'trained-one/checkpoint.pt','--steps','1'],script=script)
        require('trained-compare','compare',extra=['--checkpoint-a',out/'trained-full/checkpoint.pt','--checkpoint-b',out/'trained-resumed/checkpoint.pt'],script=script)
        retire([out/n/'checkpoint.pt' for n in ('trained-full','trained-one','trained-resumed')],'trained')
        case['train_arguments']=['--until-tokens',str(128*M),'--run-id','tl-v13-main-'+case['id'],'--recording-group',group]
        main=require('train','train',self.a.deadline_epoch-time.time()-1800,
            ['--resume',parent,'--admission',out/'check/result.json','--recovery',out/'trained-compare/recovery.json'],True)
        if main['progress']['tokens']!=128*M:raise RuntimeError('128Mi endpoint missing')
        # Decode the final checkpoint before retiring its superseded pilot optimizer.
        saved=torch.load(out/'train/latest.pt',map_location='cpu',weights_only=False)
        if saved['progress']['tokens']!=128*M or 'optimizer' not in saved:raise ValueError('final recovery artifact missing')
        if not (out/f'pilot/tokens-{8*M}.pt').is_file():raise ValueError('8Mi scientific weight missing')
        del saved;retire([parent],'pilot-optimizer')
        train_data=self.a.data;self.a.data=Path(self.plan['confirmation_data'])
        try:
            require('confirmation',None,1200,['--training-data',train_data,'--checkpoint',out/'train/latest.pt',
                '--lens-cli',self.a.recording_cli,'--run-id','tl-v13-confirm-'+case['id'],'--recording-group','TL-HB-MEM-CONFIRM'],script='scripts/evaluate_capacity_v11.py')
        finally:self.a.data=train_data
        table=out/'diagnostic-table.json';atomic_json(table,dict(cases=[dict(id=case['id'],config=case['config'],
            checkpoint=str(out/'train/latest.pt'),sha256=digest(out/'train/latest.pt'),history={})]))
        # The observer accepts model configs through its table, not --config.
        # A dedicated adapter below retains the common stage lifecycle.
        require('diagnostic',None,1800,['--table',table,'--root',self.plan['root'],'--lens-cli',self.a.recording_cli,
            '--run-id','tl-v13-diag-'+case['id'],'--recording-group','TL-HB-MEM-DIAG'],script='scripts/diagnose_hb_v12.py')
        row.update(state='completed',finished_at=now());self.state.update(state='completed',finished_at=now());self.publish()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','data','output','recording-cli'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);a=p.parse_args();w=MemWorker(a)
    try:w.run()
    except BaseException as e:
        if w.state['cases']:w.state['cases'][-1].update(state='failed',error=repr(e),finished_at=now())
        w.state.update(state='failed',error=repr(e),finished_at=now());w.publish();raise
