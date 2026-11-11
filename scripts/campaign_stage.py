#!/usr/bin/env python3
"""Separate admission, fresh-process recovery and learning for finite v6 cases."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tide_lab.records import atomic_json,digest,now,Metrics
from tide_lab.train import runtime,source_identity,installed_identity,cpu_tree
from tide_lab.campaign_runtime import (make_model,core_options,optimizer_for,update,logical_update,
    checkpoint,restore,memory,evaluate,synthetic_batch)
from tide_lab.campaign_data import Corpus
from tide_lab.language import trainable_inventory
from tide_lab.language_training import tensor_save
from tide_lab.language_checks import core_axes,ce_formula,finite_delta
import torch
from torch.nn import functional as F


def identity(config_path,c,data,mode):
    source=source_identity()
    value=dict(config_sha256=digest(config_path),source_commit=source.get('commit'),
        source_sha256=source.get('content',{}).get('tree_sha256',source.get('source_sha256')),
        installation=installed_identity(),device=c['device'],threads=c['threads'],dtype=c['dtype'],
        data_lock_sha256=digest(data/'data-lock.json') if data else None,mode=mode)
    if c['core'].get('eager_implementation')=='native':
        path=Path(core_options(c)['native_library'])
        value['native_manifest_sha256']=digest(path/'build-manifest.json')
        value['native_binary_sha256']=digest(path/'_tide_native.so')
    return value


def admission(model,c,device,out):
    b=synthetic_batch(2,3,991,device);model.eval();logits=model(b['input_ids'])
    checks=dict(analytic_ce_vjp=ce_formula(logits,b['labels']))
    opt=optimizer_for(model,c)
    if model.embedding.weight is not model.head.weight:raise AssertionError('tied embedding/head alias lost')
    before=model.embedding.weight.detach().clone()
    update(model,opt,b,c,0,device)
    if torch.equal(before,model.embedding.weight):raise AssertionError('optimizer did not update tied head')
    if c['kind']=='tide':
        checks['core']=core_axes(model.encoder,core_options(c),c['seed'],device,require_all_regions=False)
    else:
        reference=make_model(c,torch.device('cpu'))
        reference.load_state_dict(cpu_tree(model.state_dict()))
        model.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
        x=b['input_ids'];y=model(x);yr=reference(x.cpu())
        y.sum().backward();yr.sum().backward()
        if not torch.isfinite(y).all() or not torch.isfinite(yr).all():
            raise FloatingPointError('nonfinite independent forward check')
        for owner in (model,reference):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in owner.parameters()):
                raise FloatingPointError('nonfinite independent VJP check')
        checks['forward']=finite_delta(y,yr,1e-6,1e-5)
        checks['parameter_vjp']={n:finite_delta(p.grad,dict(reference.named_parameters())[n].grad,1e-6,1e-5)
            for n,p in model.named_parameters() if p.grad is not None}
        with torch.no_grad():
            changed=x.clone();changed[:,2]=(changed[:,2]+17)%4096
            checks['causal']=finite_delta(model(changed)[:,:2],model(x)[:,:2],0,0)
        if checks['causal']['state']!='passed':raise AssertionError('causal intervention failed')
        if c['kind']=='direct_semantic':
            work=model.last_metrics
            if not 0<work['work/full']<=work['work/candidates']<=work['work/physical_full_rows']:
                raise AssertionError('static fixture support/work invariants failed')
            checks['direct_support']=dict(state='passed',work=work,
                scope='Actual unit-input static FC/HB graph; physical Full rows include inactive rows. Independent formula/VJP suite is a separate source qualification. General continuation/absent-zero fibre equivalence unchecked.')
    atomic_json(out/'checks.json',checks)
    if checks['analytic_ce_vjp']['state']!='passed':raise AssertionError('CE VJP formula failed')
    return dict(checks=checks,admitted_for_finite_exploration=True,
        scope='actual topology/modules/width/backend, independent B2 L3; inactive routed regions recorded, not required to all fire',
        axes=dict(execution='passed',strict_numerical='see checks; original 1e-6/1e-5 and zero causal thresholds retained',
            discrete_semantics='passed' if c['kind']=='tide' else ('passed_static_unit_input_fixture; general_continuation_unchecked' if c['kind']=='direct_semantic' else 'static_dense_graph; no learned routing' if c['kind']=='direct_graph' else 'not_applicable'),mechanisms='passed',
            recovery='separate_fresh_process_probe_required',learning='unchecked'))


def probe(a,c,model,opt,device,ident):
    corpus=Corpus(a.data,3);progress=dict(step=0,tokens=0,cursor=0,probe=True)
    if a.resume:progress=restore(a.resume,model,opt,ident,device)
    if not progress.get('probe'):raise ValueError('probe cannot restore a training trajectory')
    losses=[]
    for _ in range(a.steps):
        ids,cursor=corpus.sample_ids(progress['cursor'],2,c['seed'])
        b=corpus.batch('train',ids,device)
        row=update(model,opt,b,c,progress['tokens'],device)
        progress.update(step=progress['step']+1,tokens=progress['tokens']+row['tokens'],cursor=cursor)
        # Explicitly test both RNG streams without changing the model objective.
        progress['rng_probe']=[random.random(),float(torch.rand(()))]
        losses.append(row['loss'])
    state=checkpoint(model,opt,progress,ident,device)
    state['last_gradients']={n:cpu_tree(p.grad) for n,p in model.named_parameters()}
    tensor_save(a.output/'checkpoint.pt',state)
    return dict(progress=progress,losses=losses,checkpoint_sha256=digest(a.output/'checkpoint.pt'),
                axes=dict(execution='passed',recovery='pending_comparison',learning='unchecked'))


def train(a,c,model,opt,device,ident,metrics,result):
    if not a.admission or not a.recovery:raise ValueError('formal training needs separate admission and recovery records')
    checked=json.loads(a.admission.read_text());recovered=json.loads(a.recovery.read_text())
    same=lambda value:{k:v for k,v in value.items() if k!='mode'}
    if not checked.get('admitted_for_finite_exploration') or same(checked['identity'])!=same(ident):
        raise ValueError('missing or wrong admission')
    if recovered.get('state')!='passed' or same(recovered['identity'])!=same(ident):
        raise ValueError('exact recovery did not pass')
    corpus=Corpus(a.data,c['training']['length']);t=c['training'];started=time.monotonic()
    deadline=min(a.deadline_epoch,time.time()+t['timeout_seconds']);stopped=[]
    def stop(sig,frame):stopped.append(sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    progress=dict(step=0,tokens=0,cursor=0,probe=False,milestones=[],dropped_tail_tokens=0)
    if a.resume:progress=restore(a.resume,model,opt,ident,device)
    if progress.get('probe'):raise ValueError('micro recovery checkpoint is not formal training ancestry')
    global_batch=t.get('global_batch',t['batch']);micro=t['batch']
    if micro>global_batch:raise ValueError('physical batch exceeds declared independent-sample global batch')
    result.update(axes=dict(execution='running',strict_numerical='see independent admission; finite exploration',
        discrete_semantics=checked['axes']['discrete_semantics'],mechanisms=checked['axes']['mechanisms'],
        recovery='passed_same_stack_three_micro_updates; long_trajectory_replay_unchecked',learning='running'),
        global_batch=global_batch,microbatch=micro,accumulation=(global_batch+micro-1)//micro)
    atomic_json(a.output/'result.json',result)
    def save(label='latest'):
        path=a.output/(label+'.pt')
        if label=='latest' and path.exists():os.replace(path,a.output/'previous.pt')
        tensor_save(path,checkpoint(model,opt,progress,ident,device))
        atomic_json(a.output/'checkpoint.json',dict(file=path.name,sha256=digest(path),progress=progress,at=now()))
    def validate(tag,limit):
        if hasattr(model,'encoder'):model.encoder.observe=False
        val=evaluate(model,corpus,c,device,token_limit=limit,deadline=deadline-45)
        atomic_json(a.output/(tag+'.json'),dict(val,progress=copy.deepcopy(progress)))
        if val['nll'] is not None:
            metrics.log(progress['step'],{'validation/nll':val['nll'],'validation/tokens':val['tokens'],
                'train/tokens':progress['tokens'],'validation/complete':val['state']=='completed'})
        return val
    initial=validate('validation-initial',min(t['eval_tokens'],t.get('initial_eval_tokens',2048)))
    last_save=time.monotonic();next_eval=((progress['tokens']//t['validation_every_tokens'])+1)*t['validation_every_tokens']
    training_seconds=0.;row=None
    stop_target=min(t['target_tokens'],a.until_tokens or t['target_tokens'])
    if progress['tokens']>stop_target:raise ValueError('resume exceeds requested stage exposure')
    while progress['tokens']<stop_target and not stopped and time.time()<deadline-120:
        next_milestone=min([m for m in t['milestones'] if m>progress['tokens']]+[stop_target])
        ids,cursor=corpus.sample_ids(progress['cursor'],global_batch,c['seed'])
        remain=next_milestone-progress['tokens'];batches=[];raw_tokens=0
        for i in range(0,len(ids),micro):
            subset=ids[i:i+micro]
            raw_tokens+=sum(int(corpus.blocks['train'][j,1]) for j in subset)
            if remain<=0:continue
            b=corpus.batch('train',subset,device,token_limit=remain)
            remain-=int((b['labels']!=-100).sum());batches.append(b)
        if hasattr(model,'encoder'):model.encoder.observe=progress['step']%t['log_every']==0
        row=logical_update(model,opt,batches,c,progress['tokens'],device)
        training_seconds+=row['seconds'];progress.update(step=progress['step']+1,
            tokens=progress['tokens']+row['tokens'],cursor=cursor,
            dropped_tail_tokens=progress['dropped_tail_tokens']+max(0,raw_tokens-row['tokens']))
        resources=memory(device)
        if resources['resources/peak_rss_mib']>c['resources']['max_rss_gib']*1024:
            raise MemoryError('campaign per-case RSS ceiling exceeded')
        values={'train/loss':row['loss'],'train/tokens':progress['tokens'],'train/lr':row['lr'],
            'train/gradient_norm':row['gradient_norm'],'train/step_seconds':row['seconds'],
            'train/tokens_per_second':row['tokens']/row['seconds'],**row['work'],**resources}
        if progress['step']<=2 or progress['step']%t['log_every']==0:metrics.log(progress['step'],values)
        atomic_json(a.output/'progress.json',dict(progress,updated_at=now(),last_update=values,
            training_seconds=training_seconds,elapsed_seconds=time.monotonic()-started))
        milestone=progress['tokens'] in t['milestones']
        if milestone:
            progress['milestones'].append(progress['tokens']);save('tokens-'+str(progress['tokens']))
        if milestone or time.monotonic()-last_save>=t['checkpoint_seconds']:
            save();last_save=time.monotonic()
        if milestone or progress['tokens']>=next_eval:
            validate('validation-'+str(progress['tokens']),t['eval_tokens'])
            next_eval=((progress['tokens']//t['validation_every_tokens'])+1)*t['validation_every_tokens']
    save()
    final=None
    if time.time()<deadline-60 and not stopped:final=validate('validation-final',t['eval_tokens'])
    inference=dict(state='unchecked_budget')
    if time.time()<deadline-60 and not stopped:
        with torch.no_grad():
            generated=torch.ones((1,1),device=device,dtype=torch.long);inference_start=time.monotonic()
            for _ in range(c['evaluation']['inference_length']):
                if time.time()>=deadline-30:break
                logits=model(generated)
                if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite generation')
                generated=torch.cat([generated,logits[:,-1].argmax(-1,keepdim=True)],1)
            inference=dict(state='completed' if generated.shape[1]==c['evaluation']['inference_length']+1 else 'budget_truncated',
                token_ids=generated.cpu().tolist(),seconds=time.monotonic()-inference_start,
                method='greedy full-prefix recompute; execution/finite-output acceptance, not language-quality proof')
        atomic_json(a.output/'inference.json',inference)
    return dict(state='completed' if progress['tokens']==stop_target else 'budget_truncated',
        progress=progress,target_tokens=t['target_tokens'],stage_target_tokens=stop_target,training_seconds=training_seconds,
        initial_validation_nll=initial['nll'],final_validation_nll=final['nll'] if final else None,
        inference=inference,stop_signal=stopped or None,
        axes=dict(result['axes'],execution='completed' if progress['tokens']==stop_target else 'budget_truncated',
            learning='awaiting_scientific_review'),
        comparison_note='same global sample batch and token schedule; physical accumulation and implementation recorded; milestone tail drops explicit and identical for matched global batch')


def main(a):
    c=json.loads(a.config.read_text());device=runtime(c['device'],c['threads'],c['seed'])
    ident=identity(a.config,c,a.data,'probe' if a.action=='probe' else a.action)
    a.output.mkdir(parents=True,exist_ok=False);metrics=None;started=time.monotonic()
    result=dict(schema='tide-language-campaign-stage-v1',state='running',action=a.action,
        started_at=now(),identity=ident,config_id=c['id'],physical_visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    if a.resume:
        result['ancestry']=dict(checkpoint=str(a.resume.resolve()),sha256=digest(a.resume),
            meaning='exact application continuation; not scientific support')
    atomic_json(a.output/'result.json',result)
    try:
        model=make_model(c,device);opt=optimizer_for(model,c);result['inventory']=trainable_inventory(model)
        if a.action=='check':result.update(admission(model,c,device,a.output),state='completed')
        elif a.action=='probe':result.update(probe(a,c,model,opt,device,ident),state='completed')
        else:
            name=a.run_id or 'tl-v6-'+c['id']
            group=a.recording_group or c['study_id']
            atomic_json(a.output/'manifest.json',dict(schema='tide-lab-run-v1',project='tide-learning-lab',
                run_id=name,name=name,study_id=group,group=group,stage='train',
                created_at=now(),config=c,identity=ident,ancestry=result.get('ancestry'),resolved=dict(seed=c['seed'],device=c['device'],
                    batch=c['training']['batch'],global_batch=c['training'].get('global_batch'),length=c['training']['length'])))
            metrics=Metrics(a.output,name)
            result.update(train(a,c,model,opt,device,ident,metrics,result))
    except BaseException as error:
        result.update(state='failed',error=repr(error));raise
    finally:
        if metrics:metrics.close()
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-started)
        atomic_json(a.output/'result.json',result)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=result['state']))
        print(json.dumps({k:v for k,v in result.items() if k not in ('identity','checks','inventory')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','probe','train'])
    for n in ('config','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('data','resume','admission','recovery'):p.add_argument('--'+n,type=Path)
    p.add_argument('--steps',type=int,default=3);p.add_argument('--deadline-epoch',type=float,default=float('inf'))
    p.add_argument('--until-tokens',type=int);p.add_argument('--run-id')
    p.add_argument('--recording-group')
    main(p.parse_args())
