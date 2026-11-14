#!/usr/bin/env python3
"""Explicit source migration, full-batch recovery, and bounded learning prefix."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,digest,now,Metrics
from tide_lab.train import runtime,cpu_tree
from tide_lab.campaign_runtime import (make_model,optimizer_for,restore,checkpoint,logical_update,
                                      exact_tree,evaluate,memory)
from tide_lab.campaign_data import Corpus
from tide_lab.language_training import tensor_save
from tide_lab.supervisor import verify_source
from tide_lab.v10_repair import (validate_config_change,assert_finite_tree,training_batches,
                                advance,gradient_signatures)
from scripts.campaign_stage import identity


def migrate(a,c,ident):
    old=json.loads(a.parent_config.read_text());validate_config_change(old,c)
    source=verify_source(a.parent_source)
    proof=json.loads(a.parent_diagnosis.read_text())
    if (proof['state']!='failure_reproduced' or proof['mode']!='replay' or
        proof['config_id']!=c['id'] or proof['training_source']['commit']!=source['commit'] or
        not proof['common_step_comparisons'] or
        any(x['loss_delta']!=0 or x['gradient_norm_delta']!=0 or not x['tokens_equal']
            for x in proof['common_step_comparisons'])):
        raise ValueError('parent reconstruction has not passed the declared observed replay checks')
    state=torch.load(a.resume,map_location='cpu',weights_only=False);parent=state['identity']
    expected=dict(ident,config_sha256=digest(a.parent_config),source_commit=source['commit'],
                  source_sha256=source['content']['tree_sha256'])
    if state['schema']!='tide-language-campaign-checkpoint-v1' or parent!=expected:
        raise ValueError('parent source/config/data/installation/backend identity mismatch')
    if state['progress']!=proof['failure_progress'] or state['progress'].get('probe'):
        raise ValueError('not the declared reconstructed pre-failure training checkpoint')
    if proof['failure_step']!=state['progress']['step']+1:raise ValueError('wrong failure boundary')
    model=make_model(c,torch.device('cpu'));model.load_state_dict(state['model'],strict=True)
    opt=optimizer_for(model,c);opt.load_state_dict(state['optimizer'])
    if model.embedding.weight is not model.head.weight:raise AssertionError('shared head alias lost')
    exact_tree(state['model'],cpu_tree(model.state_dict()))
    exact_tree(state['optimizer'],cpu_tree(opt.state_dict()))
    assert_finite_tree(state['model']);assert_finite_tree(state['optimizer'])
    ancestry=dict(parent_checkpoint=str(a.resume),parent_sha256=digest(a.resume),parent_identity=parent,
        parent_diagnosis_sha256=digest(a.parent_diagnosis),
        kind='explicit_source_migration_from_reconstructed_prefailure_state',
        change='aggregate_vjp native -> ordered-v1; weights/optimizer/RNG/cursor preserved',
        limit='Logged observables and failure matched; original historical parameter bytes unavailable.')
    migrated=dict(state,identity=ident,source_migration=ancestry)
    tensor_save(a.output/'checkpoint.pt',migrated)
    reread=torch.load(a.output/'checkpoint.pt',map_location='cpu',weights_only=False)
    for key in state:
        if key!='identity':exact_tree(state[key],reread[key],key)
    return dict(state='completed',ancestry=ancestry,progress=state['progress'],
        checkpoint_sha256=digest(a.output/'checkpoint.pt'),
        axes=dict(execution='completed',migration='payload_byte_values_preserved',
                  recovery='requires_new_source_fresh_process_check',learning='unchecked'))


def segment(a,c,ident,device,metrics):
    model=make_model(c,device);opt=optimizer_for(model,c)
    progress=restore(a.resume,model,opt,ident,device);start_progress=copy.deepcopy(progress)
    verification=a.action=='segment'
    if not verification and progress.get('probe'):raise ValueError('verification checkpoint is not learning ancestry')
    if verification:progress['probe']=True
    corpus=Corpus(a.data,c['training']['length']);rows=[]
    initial=None;final=None
    if not verification:
        model.set_progress(progress['tokens'])
        initial=evaluate(model,corpus,c,device,token_limit=c['training']['eval_tokens'],deadline=a.deadline_epoch-90)
        if initial['state']!='completed':raise TimeoutError('pilot initial validation incomplete')
        atomic_json(a.output/'validation-initial.json',initial)
        metrics.log(progress['step'],{'validation/nll':initial['nll'],'train/tokens':progress['tokens']})
    for i in range(a.steps):
        if time.time()>=a.deadline_epoch-90:raise TimeoutError('bounded prefix deadline reached')
        batches,cursor,raw=training_batches(corpus,c,progress,device,128*2**20)
        row=logical_update(model,opt,batches,c,progress['tokens'],device)
        advance(progress,row,cursor,raw,c);rows.append(row)
        if verification:
            progress['rng_probe']=[random.random(),float(torch.rand(())),float(torch.rand((),device=device))]
        if metrics and (i<2 or (i+1)%16==0):
            metrics.log(progress['step'],{'train/loss':row['loss'],'train/tokens':progress['tokens'],
                'train/gradient_norm':row['gradient_norm'],'train/lr':row['lr'],
                'train/step_seconds':row['seconds'],'train/tokens_per_second':row['tokens']/row['seconds'],
                **row['work'],**memory(device)})
        atomic_json(a.output/'progress.json',dict(progress,updated_at=now(),last_loss=row['loss'],
            last_gradient_norm=row['gradient_norm'],prefix_updates=i+1))
        print(json.dumps(dict(step=progress['step'],tokens=progress['tokens'],loss=row['loss'],
                              gradient_norm=row['gradient_norm'])),flush=True)
    state=checkpoint(model,opt,progress,ident,device)
    assert_finite_tree(state['model']);assert_finite_tree(state['optimizer'])
    if verification:state['last_gradient_signatures']=gradient_signatures(model)
    if not verification:
        model.set_progress(progress['tokens'])
        final=evaluate(model,corpus,c,device,token_limit=c['training']['eval_tokens'],deadline=a.deadline_epoch-30)
        if final['state']!='completed':raise TimeoutError('pilot final validation incomplete')
        atomic_json(a.output/'validation-final.json',final)
        metrics.log(progress['step'],{'validation/nll':final['nll'],'train/tokens':progress['tokens']})
    filename='checkpoint.pt' if verification else 'latest.pt'
    tensor_save(a.output/filename,state)
    return dict(state='completed',progress=progress,initial_progress=start_progress,updates=len(rows),
        checkpoint_sha256=digest(a.output/filename),
        loss_first=rows[0]['loss'],loss_last=rows[-1]['loss'],
        max_gradient_norm=max(r['gradient_norm'] for r in rows),
        initial_validation_nll=initial['nll'] if initial else None,final_validation_nll=final['nll'] if final else None,
        axes=dict(execution='completed',finite_parameters_and_optimizer='passed',
            discrete_semantics='per_forward_support_and_feasibility_checks_passed',
            recovery='comparison_pending' if verification else 'required_separately',
            learning='verification_only' if verification else 'short_prefix_not_long_stability_or_quality_qualification'))


def compare(a,ident):
    left=torch.load(a.checkpoint_a,map_location='cpu',weights_only=False)
    right=torch.load(a.checkpoint_b,map_location='cpu',weights_only=False)
    if left['identity']!=ident or right['identity']!=ident:raise ValueError('wrong recovery identity')
    if not left['progress']['probe'] or not right['progress']['probe']:raise ValueError('only verification checkpoints may compare')
    exact_tree(left,right)
    value=dict(state='passed',identity=ident,at=now(),
        scope='Two full L256/global64 updates on trained parameters; continuous vs 1+1 in fresh processes, including model/Adam/progress/gradient hashes/CPU+NPU+Python RNG.',
        sources=[dict(path=str(p),sha256=digest(p)) for p in (a.checkpoint_a,a.checkpoint_b)])
    atomic_json(a.output/'recovery.json',value)
    return dict(state='completed',recovery=value,axes=dict(execution='completed',recovery='passed',learning='unchecked'))


def main(a):
    c=json.loads(a.config.read_text())
    if a.action=='witness':
        from scripts.probe_v10_division import main as probe
        try:
            probe(argparse.Namespace(device=c['device'],output=a.output,witness=a.witness))
            value=json.loads((a.output/'result.json').read_text())
            if not value['broadcast_fp64_gradcheck']:raise AssertionError('broadcast VJP gradcheck failed')
            for field in ('forward','grad_a','grad_b'):
                if value['witness']['variants']['ordered'][field]['check']['state']!='passed':
                    raise AssertionError('production quotient witness strict check failed: '+field)
            for row in value['cases']:
                if row['method']=='ordered' and row['case']!='genuine_overflow':
                    if any(x['state']!='passed' for x in row['checks'].values()):raise AssertionError('scalar witness failed')
            if not value['witness']['variants']['ordered']['forward_bitwise_equal_to_ordinary']:
                raise AssertionError('forward changed on the actual witness')
            atomic_json(a.output/'completed.json',dict(producer_finished=True,state='completed'))
        except BaseException as e:
            value=json.loads((a.output/'result.json').read_text()) if (a.output/'result.json').exists() else {}
            atomic_json(a.output/'result.json',dict(value,state='failed',error=repr(e)))
            raise
        return
    device=runtime(c['device'] if a.action in ('segment','pilot') else 'cpu',c['threads'],c['seed'])
    ident=identity(a.config,c,a.data,'train');started=time.monotonic();metrics=None
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='tide-v10-repair-stage-v1',state='running',action=a.action,config_id=c['id'],
                identity=ident,started_at=now())
    if a.resume:result['ancestry']=dict(checkpoint=str(a.resume),sha256=digest(a.resume))
    atomic_json(a.output/'result.json',result)
    try:
        if a.action=='migrate':result.update(migrate(a,c,ident))
        elif a.action=='compare':result.update(compare(a,ident))
        else:
            if a.action=='pilot':
                atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
                    run_id=a.run_id,name=a.run_id,group='TL-CONTROL-v10',study_id='TL-CONTROL-v10',
                    stage='repaired_learning_prefix',created_at=now(),config=c,identity=ident,ancestry=result['ancestry']))
                metrics=Metrics(a.output,a.run_id)
            result.update(segment(a,c,ident,device,metrics))
    except BaseException as e:
        result.update(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if metrics:metrics.close()
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output/'result.json',result)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=result['state']))
        print(json.dumps(dict(state=result['state'],action=a.action,seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('witness','migrate','segment','compare','pilot'))
    for name in ('config','data','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('parent-config','parent-source','parent-diagnosis','resume','checkpoint-a','checkpoint-b','witness'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--run-id')
    p.add_argument('--deadline-epoch',type=float,required=True)
    main(p.parse_args())
