#!/usr/bin/env python3
"""Bounded v4: longer A learning, stronger reference, one B bottleneck control.

Scientific support requires later review. Scale is conditional on validation
learning, relevant checks, exact tiny recovery and a complete-pair budget.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from tide_lab.language_driver import StageRunner
from tide_lab.records import atomic_json,digest


def scale_forecast(calibrations,steps=128):
    return sum(60+1.35*(steps*r['mean_step_seconds']+
        3*r['evaluation_preview']['seconds']*2048/max(1,r['evaluation_preview']['tokens']))
        for r in calibrations.values())+900


def run(a):
    import torch
    from tide_lab.language_training import load_tokens,indices
    from tide_lab.train import installed_identity
    from scripts.run_capability_fourth import exact_tree
    torch.set_num_threads(1)
    plan=json.loads(a.plan.read_text());budget=plan['budget'];v4=plan['confirmation']
    if not 0<a.compute_seconds<=budget[a.route+'_allocated_seconds']:
        raise ValueError('requested compute exceeds route allocation')
    if a.deadline_epoch!=v4['deadline_epoch']:
        raise ValueError('shared deadline must match the frozen new batch')
    limits={key:budget[a.route+'_'+key+'_seconds'] for key in ('admission','training','final')}
    if a.route=='a':limits['scale']=budget['a_scale_seconds']
    runner=StageRunner(a,plan,limits,a.compute_seconds-60)
    data,lock=load_tokens(a.data,a.route,32)
    if lock['confirmation']['selection_seed']!=v4['selection_seed']:
        raise ValueError('wrong confirmation selection')
    counts=(data['train']['labels']!=-100).sum(1)
    def exposure(steps):return sum(int(counts[indices(len(counts),8,11,s)].sum()) for s in range(steps))
    runner.state.update(package_id=plan['package_id'],training=[],evaluations=[],admission={},
        installation=installed_identity(),
        exposure={str(n):exposure(n) for n in (128,512,2048)},
        axes=dict(execution='running',strict_numerical='original thresholds; finite failures retained',
            discrete_semantics='pending affected checks',recovery='pending fresh-process tiny check',
            learning='unchecked',performance='engineering observations; speedup is not a science gate'))
    runner.publish();cache_path=None

    def flags(kind,length=32,batch=8,depth=1):
        result=['--kind',kind,'--length',str(length),'--batch',str(batch),'--depth',str(depth),'--seed','11']
        if a.route=='b':result+=['--cache',str(cache_path)]
        return result

    def skip(name,reason):
        runner.state['skipped'].append(dict(name=name,reason=reason));runner.publish()

    def admit(kind,depth=1,category='admission'):
        label=f'd{depth}-{kind}'
        _,checks=runner.stage('check-'+label,'checks',category,180,flags(kind,3,2,depth))
        record=dict(checks=checks,recovery=dict(state='unchecked'),admitted=False)
        runner.state['admission'][label]=record;runner.publish()
        if checks['state']!='completed' or not checks.get('admitted_for_finite_exploration'):return False
        common=flags(kind,3,2,depth)+['--steps','3','--probe','--retain-step','1']
        if a.route=='a':common+=['--schedule-steps','512']
        outputs=[]
        for name,extra in [('full',[]),('split',['--max-updates','1']),('continued',None)]:
            if extra is None:extra=['--resume',str(outputs[1]/'checkpoint.pt')]
            path,result=runner.stage('resume-'+label+'-'+name,'train',category,100,common+extra,record=False)
            if result['state']!='completed':
                record['recovery']=dict(state='failed',stage=str(path));runner.publish();return False
            outputs.append(path)
        try:
            full=torch.load(outputs[0]/'checkpoint.pt',map_location='cpu',weights_only=False)
            resumed=torch.load(outputs[2]/'checkpoint.pt',map_location='cpu',weights_only=False)
            exact_tree(full,resumed)
            retained=torch.load(outputs[0]/'checkpoint-step1.pt',map_location='cpu',weights_only=False)
            split=torch.load(outputs[1]/'checkpoint.pt',map_location='cpu',weights_only=False)
            exact_tree(retained,split)
            record['recovery']=dict(state='passed',full=str(outputs[0]),resumed=str(outputs[2]),
                scope='B2/L3 three updates, fresh processes, exact model/optimizer/RNG/cursor/identity; retained prefix exact',
                schedule_steps=512 if a.route=='a' else 3)
            record['admitted']=True
        except AssertionError as error:record['recovery']=dict(state='failed',error=str(error))
        runner.publish();return record['admitted']

    def evaluate(kind,checkpoint=None,label=None,category='final',depth=1,inference=True,baseline=False):
        label=label or kind
        common=flags(kind,depth=depth)+(['--baseline-only'] if baseline else ['--checkpoint',str(checkpoint)])
        path,result=runner.stage('test-'+label,'evaluate',category,240,common)
        row=dict(label=label,kind=kind,depth=depth,state=result['state'],output=str(path),checkpoint=str(checkpoint))
        runner.state['evaluations'].append(row);runner.publish()
        if inference and result['state']=='completed':
            _,inf=runner.stage('inference-'+label,'inference',category,180,common)
            row['inference_state']=inf['state'];runner.publish()
        return result

    def train(kind,steps,label,category='training',depth=1,horizon=None,retain=False,cap=10800):
        extra=flags(kind,depth=depth)+['--steps',str(steps)]
        if horizon:extra+=['--schedule-steps',str(horizon)]
        if retain:extra+=['--retain-step','128']
        path,result=runner.stage('train-'+label,'train',category,cap,extra)
        row=dict(label=label,kind=kind,depth=depth,seed=11,state=result['state'],output=str(path),
            steps=result.get('steps'),valid_tokens=result.get('valid_tokens'),schedule_steps=horizon or steps,
            learning_signal=result.get('learning_signal'))
        runner.state['training'].append(row);runner.publish()
        if result['state']=='completed' and (result['steps']!=steps or result['valid_tokens']!=exposure(steps)):
            raise AssertionError('observed exposure differs from fixed sampler workload')
        return path,result

    try:
        if a.route=='b':
            path,result=runner.stage('cache-confirmation','cache','admission',300,
                ['--length','32','--reuse-cache',str(a.parent_cache)])
            if result['state']!='completed':raise RuntimeError('confirmation cache failed')
            cache_path=path/'features.pt'
            if admit('bottleneck'):
                path,result=train('bottleneck',128,'bottleneck-128',cap=300)
                if result['state']=='completed':evaluate('bottleneck',path/'checkpoint.pt')
            else:skip('bottleneck-training','new branch mechanism or exact micro recovery failed')
            evaluate('original',label='frozen',baseline=True,inference=False)
            for kind in ('original','mlp','tide-dense','tide-sparse'):
                prior=a.parent_b/('train-seed11-'+kind)/'checkpoint.pt'
                evaluate(kind,prior,label='v3-'+kind,inference=False)
            runner.state['prior_inference_scope']='v3 fixed checkpoints retain prior B1/L16 inference, strict finite failures and exact micro recovery; only confirmation documents are new'
        else:
            runner.stage('unigram','unigram','final',90,['--length','32'])
            completed={}
            for kind in ('transformer','tide-dense','tide-sparse'):
                if not admit(kind):
                    skip('train-'+kind,'affected mechanism or exact micro recovery failed');continue
                path,result=train(kind,512,kind+'-512',horizon=512,retain=kind.startswith('tide'),
                    cap=300 if kind=='transformer' else 10800)
                if result['state']=='completed':
                    completed[kind]=(path,result);evaluate(kind,path/'checkpoint.pt',label=kind+'-512')
                    runner.stage('context-reset-'+kind,'evaluate','final',180,
                        flags(kind)+['--checkpoint',str(path/'checkpoint.pt'),'--intervention','context_reset'])
                    if kind.startswith('tide'):
                        runner.stage('zero-core-'+kind,'evaluate','final',180,
                            flags(kind)+['--checkpoint',str(path/'checkpoint.pt'),'--intervention','zero_core'])
                    if kind=='transformer':
                        extra_path,extra_result=train('transformer',2048,'transformer-extra2048',horizon=2048,cap=300)
                        if extra_result['state']=='completed':
                            evaluate('transformer',extra_path/'checkpoint.pt',label='transformer-extra2048')
                            runner.stage('context-reset-extra2048','evaluate','final',180,
                                flags('transformer')+['--checkpoint',str(extra_path/'checkpoint.pt'),'--intervention','context_reset'])
            pair=('tide-dense','tide-sparse')
            # Finish every main confirmation before spending remaining route time on scale.
            if all(k in completed and completed[k][1]['learning_signal'] for k in pair) and runner.remaining('scale')>2400:
                cals={}
                for kind in pair:
                    if not admit(kind,2,'scale'):break
                    _,result=runner.stage('cal-d2-'+kind,'calibrate','scale',360,flags(kind,depth=2))
                    if result['state']!='completed':break
                    if result.get('npu_peak_allocated_mib',0)>.9*result.get('npu_capacity_mib',float('inf')):break
                    cals[kind]=result
                needed=scale_forecast(cals) if len(cals)==2 else float('inf')
                if needed<runner.remaining('scale')-60:
                    runner.state['decisions'].append(dict(name='FC8-matched128',forecast_seconds=needed,
                        remaining_seconds=runner.remaining('scale'),schedule_steps=512,updates=128,
                        reference='FC4 checkpoint-step128 from the 512-update trajectory; same sampler and LR prefix'))
                    runner.publish()
                    for i,kind in enumerate(pair):
                        other_reserve=(scale_forecast({pair[1]:cals[pair[1]]}) if i==0 else 480)
                        cap=min(7200,60+1.5*128*cals[kind]['mean_step_seconds'],runner.remaining('scale')-other_reserve)
                        path,result=train(kind,128,'fc8-'+kind,category='scale',depth=2,horizon=512,cap=cap)
                        if result['state']=='completed':
                            evaluate(kind,path/'checkpoint.pt',label='fc8-'+kind,category='scale',depth=2)
                            evaluate(kind,completed[kind][0]/'checkpoint-step128.pt',label='fc4-prefix128-'+kind,
                                     category='scale',inference=False)
                else:skip('FC8-training','complete matched pair plus independent evaluation does not fit remaining route budget or relevant admission failed')
            else:skip('FC8','requires both FC4 validation learning signals and remaining complete-pair budget')
        skip('seed22','not selected in this finite batch; one seed cannot establish training-randomness robustness')
        required=1 if a.route=='b' else 4
        complete=sum(r['state']=='completed' for r in runner.state['training'])>=required
        main_eval=[r for r in runner.state['evaluations'] if not r['label'].startswith(('fc8-','fc4-prefix'))]
        complete=complete and len(main_eval)>=(6 if a.route=='b' else 4) and all(r['state']=='completed' and r.get('inference_state','completed')=='completed' for r in main_eval)
        runner.state['axes']['execution']='completed' if complete else 'partial'
        runner.state['axes']['recovery']='inspect per-candidate affected checks; no long uninterrupted-vs-resumed reference'
        runner.state['axes']['learning']='independent confirmation records; human review pending'
        runner.state['science_review']='pending; no support assigned from process exit'
        runner.finish('completed' if complete else 'failed');return 0 if complete else 1
    except BaseException as error:
        runner.state['axes']['execution']='failed';runner.finish('failed',repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','data','output','asset-path','recording-cli'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-cache',type=Path);p.add_argument('--parent-b',type=Path)
    p.add_argument('--route',choices=['a','b'],required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--compute-seconds',type=float,required=True)
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
