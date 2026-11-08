#!/usr/bin/env python3
"""One fixed, budgeted language route in a cooperative single-NPU allocation.

Stage exit, numerical checks, recovery and learning remain separate. The driver
may choose only predeclared batch/length/steps, a depth2 pair, or a seed22 pair.
Scientific support is assigned later by review, never by this coordinator.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
from tide_lab.language_driver import StageRunner
from tide_lab.records import atomic_json,digest


def forecast(calibration,steps,validation_tokens=2048):
    preview=calibration['evaluation_preview']
    validation=preview['seconds']*validation_tokens/max(1,preview['tokens'])
    # Startup/checkpoint/recorder allowance and conservative timing headroom.
    return 60+1.35*(steps*calibration['mean_step_seconds']+(2+(steps-1)//64)*validation)


def choose_case(cases,plan,available):
    """Maximize calibrated token throughput subject to full paired pilot budget."""
    training=plan['training'];eligible=[]
    for case in cases:
        budget_steps=min(training['maximum_updates'],training['maximum_valid_tokens']//(case['batch']*case['length']))
        feasible=[n for n in range(training['minimum_updates'],budget_steps+1)
                  if sum(forecast(r,n) for r in case['calibrations'].values())<=available
                  and all(forecast(r,n)<=training['per_attempt_seconds'] for r in case['calibrations'].values())]
        if not feasible:continue
        steps=max(feasible)
        # Token counts are locked exactly by the deterministic sampler below.
        expected=(case['exact_exposures'][steps-1] if 'exact_exposures' in case else
                  min(r['valid_tokens_per_update'] for r in case['calibrations'].values())*steps)
        if expected<training['minimum_valid_tokens']:continue
        seconds=sum(r['mean_step_seconds'] for r in case['calibrations'].values())
        eligible.append(dict(case,steps=steps,throughput=case['batch']*case['length']/seconds,
                             forecast_seconds=sum(forecast(r,steps) for r in case['calibrations'].values())))
    if not eligible:return None
    best=max(c['throughput'] for c in eligible)
    return min((c for c in eligible if c['throughput']>=best/1.1),key=lambda c:c['batch'])


def run(a):
    import torch
    torch.set_num_threads(1)
    from scripts.run_capability_fourth import exact_tree
    from tide_lab.language_training import load_tokens,indices
    plan=json.loads(a.plan.read_text());prefix=a.route+'_'
    limits={key:plan['budget'][prefix+key+'_seconds'] for key in ('admission','training','final')}
    if a.route=='a':limits['scale']=plan['budget']['a_scale_seconds']
    configured=plan['budget'][prefix+'allocated_seconds']
    if not 0<a.compute_seconds<=configured:raise ValueError('route compute request exceeds fixed allocation cap')
    runner=StageRunner(a,plan,limits,a.compute_seconds-30)
    runner.state.update(candidates={},calibration=[],training=[],evaluations=[],
        axes=dict(execution='running',strict_numerical='inspect individual records',
                  recovery='unchecked',learning='unchecked',performance='calibration only'),
        source_cpu_admission=dict(path=str(a.cpu_evidence),sha256=digest(a.cpu_evidence)))
    candidates=plan[a.route+'_candidates'];cache_paths={};seed=plan['initial_seed'];base_plan=a.plan
    def skip(name,reason):
        runner.state['skipped'].append(dict(name=name,reason=reason));runner.publish()
    def flags(kind,length,batch=2,depth=1,seed_value=seed):
        result=['--kind',kind,'--length',str(length),'--batch',str(batch),'--depth',str(depth),'--seed',str(seed_value)]
        if a.route=='b':result+=['--cache',str(cache_paths[32 if length==3 else length])]
        return result
    def cache(length):
        if a.route!='b':return True
        if length in cache_paths:return True
        path,result=runner.stage(f'cache-L{length}','cache','admission',420,['--length',str(length)])
        if result['state']!='completed':return False
        cache_paths[length]=path/'features.pt';return True
    def recover(kind,depth=1):
        label=f'resume-d{depth}-{kind}';common=flags(kind,3,depth=depth)+['--steps','3','--probe']
        outputs=[]
        for suffix,extra in [('full',[]),('split',['--max-updates','1']),('continued',None)]:
            if extra is None:extra=['--resume',str(outputs[1]/'checkpoint.pt')]
            path,result=runner.stage(label+'-'+suffix,'train','admission' if depth==1 else 'scale',100,common+extra,record=False)
            if result['state']!='completed':return dict(state='failed',reason='fresh-process trajectory failed',attempt=str(path))
            outputs.append(path)
        try:
            exact_tree(torch.load(outputs[0]/'checkpoint.pt',map_location='cpu',weights_only=False),
                       torch.load(outputs[2]/'checkpoint.pt',map_location='cpu',weights_only=False))
            result=dict(state='passed',scope='B2/L3 three updates, fresh processes, full model/optimizer/RNG/cursor identity',
                        full_sha256=digest(outputs[0]/'checkpoint.pt'),resumed_sha256=digest(outputs[2]/'checkpoint.pt'))
        except AssertionError as error:result=dict(state='failed',error=str(error))
        atomic_json(a.output/(label+'.json'),result);return result
    def calibration(kind,length,batch,category='admission',depth=1):
        name=f'cal-d{depth}-{kind}-L{length}-B{batch}'
        _,result=runner.stage(name,'calibrate',category,plan['batch_calibration']['case_seconds'],flags(kind,length,batch,depth))
        row=dict(kind=kind,length=length,batch=batch,depth=depth,**result)
        runner.state['calibration'].append(row);runner.publish()
        if result['state']!='completed':return None
        capacity=result.get('npu_capacity_mib');peak=result.get('npu_peak_allocated_mib')
        if capacity and peak>plan['batch_calibration']['peak_fraction']*capacity:
            skip(name+'-admission','HBM headroom below declared 10%');return None
        return result
    def lock_selection(selection,label,seed_value=seed,depth=1):
        data,_=load_tokens(a.data,a.route,selection['length']);counts=(data['train']['labels']!=-100).sum(1)
        exposure=sum(int(counts[indices(len(counts),selection['batch'],seed_value,i)].sum()) for i in range(selection['steps']))
        if not plan['training']['minimum_valid_tokens']<=exposure<=plan['training']['maximum_valid_tokens']:
            raise ValueError('calibrated case does not meet exact deterministic token bounds')
        resolved=copy.deepcopy(plan)
        resolved['resolved']=dict(route=a.route,kinds=list(selection['calibrations']),length=selection['length'],
            batch=selection['batch'],microbatch=selection['batch'],accumulation=1,global_batch=selection['batch'],
            steps=selection['steps'],seed=seed_value,depth=depth,exact_valid_token_exposure=exposure,
            forecast_seconds=selection['forecast_seconds'],selection_source='stage.json calibration records',
            selection_reason='smallest feasible batch within 10% of best measured token throughput; paired updates/token exposure locked',
            checkpoint_rule='final declared update; no test selection',strict_mode='finite exploration; original numerical failures retained')
        path=a.output/(label+'.json');atomic_json(path,resolved);path.chmod(0o444)
        runner.state['decisions'].append(dict(config=str(path),sha256=digest(path),**resolved['resolved']));runner.publish()
        return path
    def train_group(selection,label,category,seed_value=seed,depth=1):
        a.plan=lock_selection(selection,label,seed_value,depth);completed={}
        for kind in selection['calibrations']:
            # Reserve the measured cost of the other candidates, rather than
            # dividing equally and prematurely timing out the slow graph.
            names=list(selection['calibrations']);later=names[names.index(kind)+1:]
            reserve=sum(forecast(selection['calibrations'][k],selection['steps']) for k in later)
            predicted=forecast(selection['calibrations'][kind],selection['steps'])
            cap=min(plan['training']['per_attempt_seconds'],predicted*1.15+30,runner.remaining(category)-reserve-15)
            name=f'train-{label}-{kind}'
            path,result=runner.stage(name,'train',category,cap,
                                    flags(kind,selection['length'],selection['batch'],depth,seed_value)+['--steps',str(selection['steps'])])
            entry=dict(kind=kind,name=name,output=str(path),state=result['state'],config=str(a.plan),
                       depth=depth,seed=seed_value,length=selection['length'],batch=selection['batch'],steps=selection['steps'],
                       learning_signal=result.get('learning_signal'),valid_tokens=result.get('valid_tokens'))
            runner.state['training'].append(entry);runner.publish()
            if result['state']=='completed':completed[kind]=(path,result,entry)
        return completed
    def evaluate_group(completed,category='final'):
        for kind,(path,_,entry) in completed.items():
            a.plan=Path(entry['config'])
            common=flags(kind,entry['length'],entry['batch'],entry['depth'],entry['seed'])+['--checkpoint',str(path/'checkpoint.pt')]
            name=entry['name'].replace('train-','test-')
            destination,result=runner.stage(name,'evaluate',category,240,common)
            runner.state['evaluations'].append(dict(name=name,kind=kind,state=result['state'],output=str(destination),source=entry['name']))
            if result['state']=='completed' and runner.remaining(category)>180:
                runner.stage(name+'-inference','inference',category,180,common)
            runner.publish()
    try:
        cpu=json.loads(a.cpu_evidence.read_text())
        if not cpu.get('independent_cpu_admission'):raise ValueError('new sequence CPU admission is not established')
        if not cache(32):raise RuntimeError('B exact-context L32 feature cache failed')
        admitted=[]
        for kind in candidates:
            _,result=runner.stage('check-'+kind,'checks','admission',150,flags(kind,3))
            record=dict(mechanisms=result['state'],admitted=False)
            if result['state']=='completed' and result.get('admitted_for_finite_exploration'):
                record['recovery']=recover(kind)
                if record['recovery']['state']=='passed':record['admitted']=True;admitted.append(kind)
            runner.state['candidates'][kind]=record;runner.publish()
        runner.state['axes']['recovery']='inspect per-candidate fresh-process results'
        if not admitted:raise RuntimeError('no candidate passed mechanisms and exact fresh-process recovery')
        if not any(k.startswith('tide') for k in admitted):
            skip('TIDE language comparison','TIDE mechanism/recovery admission failed; controls alone cannot answer architecture question')
        cases=[];previous=None;long_attempted=False;long_cases=[]
        for batch in plan['batch_calibration']['batches']+[plan['batch_calibration']['extra_batch_if_growing']]:
            if batch>=256 and (previous is None or len(cases)<2 or previous<cases[-2]['throughput']*1.1):
                skip(f'cal-B{batch}','throughput flattened or no two lower-batch cases');break
            rows={}
            for kind in admitted:
                row=calibration(kind,32,batch)
                if row is not None:rows[kind]=row
            if len(rows)!=len(admitted):
                skip(f'cal-larger-than-B{batch}','incomplete case/capacity/time; no repeated OOM search');break
            previous=batch*32/sum(r['mean_step_seconds'] for r in rows.values())
            cases.append(dict(length=32,batch=batch,calibrations=rows,throughput=previous))
            # Reserve one real L128 attempt before growing every short batch.
            # The largest step-time candidate is the conservative sentinel.
            if not long_attempted:
                long_attempted=True
                if cache(128):
                    bottleneck=max(rows,key=lambda k:rows[k]['mean_step_seconds'])
                    sentinel=calibration(bottleneck,128,batch)
                    if sentinel and forecast(sentinel,plan['training']['minimum_updates'])<min(plan['training']['per_attempt_seconds'],runner.remaining('training')-60):
                        longer={bottleneck:sentinel}
                        for other in admitted:
                            if other!=bottleneck:longer[other]=calibration(other,128,batch)
                        if all(longer.values()):long_cases.append(dict(length=128,batch=batch,calibrations=longer))
                    else:skip('L128-minimum-pilot','actual longest-step candidate failed or forecast cannot fit minimum updates; preserve exact attempt')
            if len(cases)>=2 and previous<=cases[-2]['throughput']*1.1:
                skip('larger-batches','marginal throughput gain no greater than 10%');break
        for case in cases+long_cases:
            rows,_=load_tokens(a.data,a.route,case['length']);counts=(rows['train']['labels']!=-100).sum(1)
            total=0;exposures=[]
            for step in range(plan['training']['maximum_updates']):
                total+=int(counts[indices(len(counts),case['batch'],seed,step)].sum());exposures.append(total)
            case['exact_exposures']=exposures
        selection=choose_case(long_cases,plan,runner.remaining('training')-60)
        if selection is None:selection=choose_case(cases,plan,runner.remaining('training')-60)
        if selection is None:raise RuntimeError('no calibrated common case fits minimum updates/tokens; retain execution bottleneck')
        if selection['length']==32:skip('L128','actual preferred-context attempt could not establish a complete affordable paired pilot; all subsequent candidates use L32')
        a.plan=base_plan
        completed=train_group(selection,'seed11','training')
        evaluate_group(completed)
        a.plan=Path(next(iter(completed.values()))[2]['config']) if completed else base_plan
        if a.route=='b':
            common=flags('original',selection['length'],selection['batch'])+['--baseline-only']
            runner.stage('frozen-baseline-test','evaluate','final',180,common)
            runner.stage('frozen-baseline-inference','inference','final',150,common)
        else:
            runner.stage('unigram','unigram','final',90,['--length',str(selection['length'])])
            for kind,(path,_,entry) in completed.items():
                a.plan=Path(entry['config']);common=flags(kind,entry['length'],entry['batch'])+['--checkpoint',str(path/'checkpoint.pt')]
                if kind.startswith('tide'):
                    runner.stage('ablate-'+kind,'evaluate','final',180,common+['--intervention','zero_core'])
                runner.stage('context-reset-'+kind,'evaluate','final',180,common+['--intervention','context_reset'])
            pair=['tide-dense','tide-sparse']
            if all(k in completed and completed[k][1].get('learning_signal') for k in pair) and runner.remaining('scale')>1000:
                a.plan=base_plan;scale_rows={}
                for kind in pair:
                    _,result=runner.stage('check-d2-'+kind,'checks','scale',180,flags(kind,3,depth=2))
                    if result['state']=='completed' and recover(kind,2)['state']=='passed':
                        scale_rows[kind]=calibration(kind,selection['length'],selection['batch'],'scale',2)
                scaled=choose_case([dict(length=selection['length'],batch=selection['batch'],calibrations=scale_rows)],plan,runner.remaining('scale')-480) if len(scale_rows)==2 and all(scale_rows.values()) else None
                if scaled:
                    # For a capacity comparison both depths need the same updates.
                    if scaled['steps']>=selection['steps']:
                        scaled['steps']=selection['steps'];scaled['forecast_seconds']=sum(forecast(r,scaled['steps']) for r in scale_rows.values())
                        scale_done=train_group(scaled,'depth2-seed11','scale',depth=2);evaluate_group(scale_done,'scale')
                    else:skip('depth2-training','remaining scale allocation cannot match the depth1 update/token exposure')
                else:skip('depth2-training','no admitted complete calibrated pair fits remaining scale budget')
            else:skip('depth2','requires both TIDE candidates heldout learning signal and sufficient complete-pair budget')
        # Evaluation has priority. A second seed is conditional, never a new budget.
        confirmation_category='final'
        pair={k:selection['calibrations'][k] for k in ('tide-dense','tide-sparse') if k in completed}
        need=sum(forecast(r,selection['steps']) for r in pair.values())+480
        if len(pair)==2 and runner.remaining(confirmation_category)>need:
            extra=dict(selection,calibrations=pair,forecast_seconds=need-480)
            repeated=train_group(extra,'seed22',confirmation_category,seed_value=plan['confirmation_seed']);evaluate_group(repeated)
        else:skip('seed22','complete paired repeat plus heldout evaluation does not fit remaining final allocation')
        runner.state['axes']['execution']='completed' if len(completed)==len(admitted) else 'partial'
        runner.state['axes']['learning']='raw heldout signals only; independent quality/structure review pending'
        runner.state['science_review']='pending; no automatic support or scale/quality claim'
        runner.finish('completed' if completed else 'failed')
        return 0 if completed else 1
    except BaseException as error:
        runner.state['axes']['execution']='failed';runner.finish('failed',repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','data','output','asset-path','recording-cli','cpu-evidence'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--route',choices=['a','b'],required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--compute-seconds',type=float,required=True)
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
