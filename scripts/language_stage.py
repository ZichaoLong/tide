#!/usr/bin/env python3
"""One bounded stage of the language/structure package; supervisor owns time caps."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
from tide_lab.records import Metrics,atomic_json,digest,now,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity,synchronize,cpu_tree
from tide_lab.language import load_qwen,FrozenPrefixTail,IntegratedMLP,trainable_inventory
from tide_lab.language_training import (load_tokens,load_cache,core_options,make_a,make_b,batch_from,indices,
    optimizer_for,update,evaluate,checkpoint_state,restore,tensor_save,prefix_cache,feature_statistics)


def data_and_model(a,plan,device):
    data,lock=load_tokens(a.data,a.route,a.length)
    if a.route=='a':return data,make_a(plan,a.kind,a.seed,a.depth).to(device),None
    if a.cache is None:raise ValueError('B stages require a context-matched immutable feature cache')
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if a.length!=cache['length'] and a.length!=3:raise ValueError('B context/cache mismatch')
    data=cache['splits']
    if a.length==3:
        data={s:{k:(v[:,:3] if isinstance(v,torch.Tensor) else v) for k,v in r.items()} for s,r in data.items()}
    model=make_b(plan,a.kind,a.seed,cache,a.asset_path,device,a.depth)
    return data,model,cache


def cache(a,plan,device,metrics):
    data,lock=load_tokens(a.data,'b',a.length);lm=load_qwen(a.asset_path,device)
    prepared={};start=time.monotonic();offset=0
    for split in ('train','validation','test'):
        def progress(count):
            if count%32==0 or count==len(data[split]['labels']):
                metrics.log(offset+count,{'cache/blocks':offset+count,'cache/elapsed_seconds':time.monotonic()-start})
        prepared[split]=prefix_cache(lm,data[split],device,8,progress)
        offset+=len(data[split]['labels'])
    statistics=feature_statistics(prepared['train'])
    value=dict(schema='tide-frozen-prefix-cache-v1',length=a.length,splits=prepared,statistics=statistics,
        data_lock_sha256=digest(a.data/'data-lock.json'),source=source_identity(),
        qwen_revision=plan['revisions']['qwen'],scope='teacher-forced final-MLP inputs/residual/targets; original frozen prefix recomputed at this exact context')
    tensor_save(a.output/'features.pt',value);(a.output/'features.pt').chmod(0o444)
    return dict(state='completed',features_sha256=digest(a.output/'features.pt'),length=a.length,
        splits={s:dict(blocks=len(r['labels']),tokens=int((r['labels']!=-100).sum())) for s,r in prepared.items()},
        cache_seconds=time.monotonic()-start,training_statistics='train only',
        size_bytes=(a.output/'features.pt').stat().st_size)


def checks(a,plan,device,metrics):
    from tide_lab.language_checks import a_checks,b_checks
    data,model,features=data_and_model(a,plan,device)
    if a.route=='a':result=a_checks(model,data['train'],plan,a.kind,a.seed,device,a.depth)
    else:
        lm=load_qwen(a.asset_path,device)
        tokens,_=load_tokens(a.data,'b',a.length)
        result=b_checks(model,lm,tokens['validation'],plan,a.kind,device,a.seed,a.depth)
    metrics.log(0,{'checks/exploration_admitted':int(result['admitted_for_finite_exploration'])})
    return result


def calibrate(a,plan,device,metrics):
    data,model,_=data_and_model(a,plan,device);optimizer=optimizer_for(model,plan,a.route,a.kind)
    rows=[];total=128
    for index in range(3):
        ids=indices(len(data['train']['labels']),a.batch,a.seed,index)
        b=batch_from(data['train'],ids,device)
        row=update(model,optimizer,b,plan,a.route,a.kind,total-1,total,device)
        rows.append(row)
        metrics.log(index,{'calibration/update_seconds':row['seconds'],
            'calibration/tokens_per_second':row['valid_tokens']/row['seconds'],'calibration/batch':a.batch})
    mean=sum(r['seconds'] for r in rows[1:])/2
    preview=evaluate(model,data['validation'],a.route,device,min(a.batch,8),limit=8,alpha=1.)
    return dict(state='completed',batch=a.batch,length=a.length,rows=rows,warmup_updates=1,timed_updates=2,
        mean_step_seconds=mean,valid_tokens_per_update=sum(r['valid_tokens'] for r in rows[1:])/2,
        independent_sample_batch=a.batch,microbatch=a.batch,accumulation=1,global_batch=a.batch,
        phase='full language CE at alpha1 including backward and optimizer; excludes process/model startup',
        inventory=trainable_inventory(model),evaluation_preview={k:preview[k] for k in ('seconds','tokens','blocks')},
        npu_peak_allocated_mib=torch.npu.max_memory_allocated()/2**20 if device.type=='npu' else None,
        npu_capacity_mib=torch.npu.get_device_properties(device).total_memory/2**20 if device.type=='npu' else None)


def identity(a,plan,device):
    return dict(source=source_identity(),installation=installed_identity(),config_sha256=digest(a.plan),
        data_lock_sha256=digest(a.data/'data-lock.json'),cache_sha256=digest(a.cache) if a.cache else None,
        route=a.route,kind=a.kind,seed=a.seed,batch=a.batch,length=a.length,steps=a.steps,depth=a.depth,
        device=str(device),dtype=plan['dtype'],threads=plan['threads'],physical_visible=os.getenv('ASCEND_RT_VISIBLE_DEVICES'))


def train(a,plan,device,metrics):
    data,model,_=data_and_model(a,plan,device);optimizer=optimizer_for(model,plan,a.route,a.kind)
    ident=identity(a,plan,device);step,tokens=0,0
    if a.resume:step,tokens=restore(a.resume,model,optimizer,ident,device,a.route)
    limit=min(a.steps,a.max_updates if a.max_updates is not None else a.steps)
    if limit<step:raise ValueError('resume target precedes checkpoint')
    if not a.probe and (a.steps<plan['training']['minimum_updates'] or a.steps>plan['training']['maximum_updates']):
        raise ValueError('formal pilot update budget outside frozen bounds')
    initial=None;validations=[];started=time.monotonic();history=[]
    if not a.probe:
        initial=evaluate(model,data['validation'],a.route,device,min(a.batch,8),alpha=1.)
        atomic_json(a.output/'initial-validation.json',initial)
        metrics.log(step,{'validation/nll':initial['nll'],'validation/ppl':initial['ppl'],'train/valid_tokens':tokens})
    try:
        while step<limit:
            ids=indices(len(data['train']['labels']),a.batch,a.seed,step)
            b=batch_from(data['train'],ids,device)
            next_tokens=int((b['labels']!=-100).sum())
            if not a.probe and tokens+next_tokens>plan['training']['maximum_valid_tokens']:
                raise ValueError('resolved logical steps exceed declared token budget')
            row=update(model,optimizer,b,plan,a.route,a.kind,step,a.steps,device)
            step+=1;tokens+=row['valid_tokens'];history.append(row['loss'])
            if step%plan['training']['log_every']==0 or step==limit or a.probe:
                values={'train/loss':row['loss'],'train/gradient_norm':row['gradient_norm'],'train/lr':row['lr'],
                    'train/alpha':row['alpha'],'train/teacher_mse_phase':int(row['teacher_mse']),
                    'train/valid_tokens':tokens,'train/update_seconds':row['seconds'],
                    'train/tokens_per_second':row['valid_tokens']/row['seconds'],**row['work']}
                if device.type=='npu':values['resource/npu_peak_allocated_mib']=torch.npu.max_memory_allocated()/2**20
                metrics.log(step,values)
            if step%plan['training']['checkpoint_every']==0 or step==limit:
                tensor_save(a.output/'checkpoint.pt',checkpoint_state(model,optimizer,step,tokens,ident,device,a.route))
                atomic_json(a.output/'checkpoint.json',dict(step=step,valid_tokens=tokens,sha256=digest(a.output/'checkpoint.pt')))
            if not a.probe and step%plan['training']['eval_every']==0 and step<limit:
                val=evaluate(model,data['validation'],a.route,device,min(a.batch,8),alpha=1.)
                validations.append(dict(step=step,nll=val['nll'],ppl=val['ppl']))
                metrics.log(step,{'validation/nll':val['nll'],'validation/ppl':val['ppl']})
        final=None
        if not a.probe:
            final=evaluate(model,data['validation'],a.route,device,min(a.batch,8),alpha=1.)
            atomic_json(a.output/'final-validation.json',final)
            metrics.log(step,{'validation/nll':final['nll'],'validation/ppl':final['ppl']})
        return dict(state='completed',steps=step,valid_tokens=tokens,probe=a.probe,
            inventory=trainable_inventory(model),initial_validation_nll=None if initial is None else initial['nll'],
            final_validation_nll=None if final is None else final['nll'],validation_curve=validations,
            learning_signal=None if initial is None else final['nll']<initial['nll']-.1,
            learning_scope='heldout loss signal only; no parity, convergence or architecture advantage claim',
            checkpoint_sha256=digest(a.output/'checkpoint.pt'),update_loss_first=history[0] if history else None,
            update_loss_last=history[-1] if history else None,training_wall_seconds=time.monotonic()-started,
            exact_resume='fresh-process check is a separate stage; current checkpoint decodes on CPU')
    finally:
        # Published periodic/final checkpoints survive timeouts; never label a
        # partial trajectory complete or synthesize an unobserved checkpoint.
        atomic_json(a.output/'progress.json',dict(step=step,valid_tokens=tokens,target=limit))


def evaluate_stage(a,plan,device,metrics):
    data,model,_=data_and_model(a,plan,device)
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        saved=state['identity']
        for key,value in dict(route=a.route,kind=a.kind,seed=a.seed,length=a.length,depth=a.depth).items():
            if saved[key]!=value:raise ValueError('evaluation checkpoint architecture/task mismatch: '+key)
        (model if a.route=='a' else model.selected).load_state_dict(state['model'],strict=True)
    elif not a.baseline_only:raise ValueError('heldout trained evaluation needs a checkpoint')
    alpha=0. if a.baseline_only and a.route=='b' else 1.
    result=evaluate(model,data[a.split],a.route,device,min(a.batch,8),alpha=alpha,intervention=a.intervention)
    metrics.log(0,{'evaluation/nll':result['nll'],'evaluation/ppl':result['ppl'],'evaluation/tokens':result['tokens'],**result['work_per_token']})
    return dict(state='completed',split=a.split,evaluation=result,
        checkpoint_sha256=digest(a.checkpoint) if a.checkpoint else None,quality_judgment='pending paired review')


def inference(a,plan,device,metrics):
    from tide_lab.language_inference import inference as run_inference
    _,model,_=data_and_model(a,plan,device)
    if a.checkpoint:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        for key,value in dict(route=a.route,kind=a.kind,seed=a.seed,length=a.length,depth=a.depth).items():
            if state['identity'][key]!=value:raise ValueError('inference checkpoint identity mismatch: '+key)
        (model if a.route=='a' else model.selected).load_state_dict(state['model'],strict=True)
    elif not a.baseline_only:raise ValueError('inference needs trained checkpoint or explicit baseline')
    result=run_inference(model,a,plan,device)
    metrics.log(0,{'inference/greedy_seconds':result['greedy_seconds'],'inference/generated_tokens':8})
    return result


def unigram(a,plan,device,metrics):
    from tide_lab.language_inference import unigram as run_unigram
    result=run_unigram(a)
    metrics.log(0,{'unigram/test_nll':result['evaluations']['test']['nll']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['cache','checks','calibrate','train','evaluate','inference','unigram'])
    for key in ('plan','data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--asset-path',type=Path)
    p.add_argument('--route',choices=['a','b'],required=True)
    p.add_argument('--kind',choices=['tide-dense','tide-sparse','transformer','original','mlp'],default='tide-dense')
    p.add_argument('--device',choices=['cpu','npu:0']);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--length',type=int,default=32)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--depth',type=int,choices=[1,2],default=1)
    p.add_argument('--max-updates',type=int);p.add_argument('--probe',action='store_true')
    group=p.add_mutually_exclusive_group();group.add_argument('--resume',type=Path);group.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--intervention',choices=['zero_core','context_reset'])
    p.add_argument('--run-id',required=True);p.add_argument('--recording-cli')
    a=p.parse_args();plan=json.loads(a.plan.read_text());start=time.monotonic()
    if a.length not in [3,32,128] or not 1<=a.batch<=512:raise ValueError('shape outside bounded package')
    device=runtime(a.device or plan['device'],plan['threads'],a.seed)
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved.get('identity')!=identity(a,plan,device):raise ValueError('resume rejected before output reservation')
    a.output.mkdir(parents=True,exist_ok=False)
    study='TL-LM-EXEC' if a.stage in ('cache','checks','calibrate') or a.probe else ('TL-A-LM' if a.route=='a' else 'TL-B-LM')
    if a.depth==2 and study=='TL-A-LM':study='TL-LM-SCALE'
    manifest=dict(schema='tide-lab-run-v1',project='tide-learning-lab',study_id=study,group=study,
        run_id=a.run_id,name=a.run_id,created_at=now(),config=plan,
        resolved=dict(route=a.route,kind=a.kind,seed=a.seed,length=a.length,batch=a.batch,steps=a.steps,depth=a.depth,probe=a.probe),
        identity=identity(a,plan,device),stage=a.stage)
    atomic_json(a.output/'manifest.json',manifest);metrics=Metrics(a.output,a.run_id)
    result=dict(state='running',started_at=now());atomic_json(a.output/'state.json',result);code=0
    try:
        result.update((evaluate_stage if a.stage=='evaluate' else globals()[a.stage])(a,plan,device,metrics))
    except BaseException as error:
        result.update(state='failed',error=repr(error),traceback=traceback.format_exc());traceback.print_exc();code=1
    finally:
        result.update(finished_at=now(),elapsed_seconds=time.monotonic()-start,run_id=a.run_id,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(a.output/'result.json',result);atomic_json(a.output/'state.json',result)
        metrics.close();atomic_json(a.output/'completed.json',dict(producer_finished=True,state=result['state']))
        if a.recording_cli:
            result['lens_projection_completed']=project_to_lens(a.output,a.recording_cli)
            atomic_json(a.output/'result.json',result)
    return code


if __name__=='__main__':raise SystemExit(main())
