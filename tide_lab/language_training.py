"""Finite all-token language training helpers and complete application state."""
import copy
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .language import (TideLanguageModel,SmallTransformer,FrozenPrefixTail,
                       load_qwen,trainable_inventory)
from .language_data import shifted_block
from .records import atomic_json,digest
from .train import cpu_tree,synchronize


def tensor_save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(cpu_tree(value),f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_tokens(directory,route,length):
    directory=Path(directory);lock=json.loads((directory/'data-lock.json').read_text())
    info=lock['routes'][route];path=directory/info['file']
    if digest(path)!=info['sha256']:raise ValueError('language token artifact changed')
    data=torch.load(path,map_location='cpu',weights_only=False)
    if length==lock['sequence_length']:return data,lock
    if length not in (3,32):raise ValueError('only explicit micro-check or L32 rechunk is allowed')
    result={}
    for split,rows in data.items():
        xx,yy,docs,ids=[],[],[],[]
        for i,targets in enumerate(rows['labels'].tolist()):
            valid=[t for t in targets if t!=-100]
            offsets=[0] if length==3 else range(0,len(valid),length)
            for offset in offsets:
                chunk=valid[offset:offset+length]
                if not chunk:continue
                x,y=shifted_block(chunk,length=length,bos=info['bos'],pad=info['pad'])
                xx.append(x);yy.append(y);docs.append(rows['document_ids'][i]);ids.append(rows['block_ids'][i]+f'+{offset}/L{length}')
        result[split]=dict(input_ids=torch.tensor(xx),labels=torch.tensor(yy),document_ids=docs,block_ids=ids)
    return result,lock


def core_options(plan,kind,depth=1):
    options=copy.deepcopy(plan['core']);options['depth']=depth
    options['send']='sparse' if kind=='tide-sparse' else 'dense'
    return options


def make_a(plan,kind,seed,depth=1):
    if kind=='transformer':return SmallTransformer(seed=seed)
    if kind not in ('tide-dense','tide-sparse'):raise ValueError('unknown A candidate')
    return TideLanguageModel(core_options(plan,kind,depth),seed=seed)


def load_cache(path,length):
    cache=torch.load(path,map_location='cpu',weights_only=False)
    if cache['length']!=length:raise ValueError('frozen prefix cache context mismatch')
    return cache


def make_b(plan,kind,seed,cache,asset,device,depth=1):
    lm=load_qwen(asset,device)
    model=FrozenPrefixTail(lm,kind,core_options(plan,kind,depth),cache['statistics'],seed).to(device)
    del lm
    return model


def batch_from(rows,ids,device):
    return {k:v[ids].to(device) for k,v in rows.items() if isinstance(v,torch.Tensor)}


def indices(count,batch,seed,step):
    generator=torch.Generator(device='cpu');generator.manual_seed(seed+104729*step)
    return torch.randint(count,(batch,),generator=generator)


def schedule(plan,route,kind,step,total):
    training=plan['training'];teacher=max(1,round(total*plan['b_schedule']['teacher_fraction']))
    shadow=route=='b' and kind!='original' and step<teacher
    if route=='a' or kind=='original':alpha=1.
    elif shadow:alpha=0.
    else:
        ramp=max(1,round((total-teacher)*plan['b_schedule']['ce_alpha_ramp_fraction']))
        alpha=min(1.,plan['b_schedule']['ce_alpha_start']+(1-plan['b_schedule']['ce_alpha_start'])*(step-teacher)/ramp)
    base=(training['lr_a'] if route=='a' else training['lr_b_original'] if kind=='original' else training['lr_b_replacement'])
    warm=max(1,round(total*training['warmup_fraction']))
    fraction=(step+1)/warm if step<warm else training['lr_floor_fraction']+(1-training['lr_floor_fraction'])*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))
    return shadow,alpha,base*fraction


def optimizer_for(model,plan,route,kind):
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params)!=len({id(p) for p in params}):raise AssertionError('duplicated optimizer parameter')
    cfg=plan['training']
    return torch.optim.AdamW(params,lr=cfg['lr_a'] if route=='a' else cfg['lr_b_replacement'],
        eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False)


def update(model,optimizer,batch,plan,route,kind,step,total,device):
    model.train();shadow,alpha,lr=schedule(plan,route,kind,step,total)
    for group in optimizer.param_groups:group['lr']=lr
    synchronize(device);started=time.monotonic();optimizer.zero_grad(set_to_none=True)
    if route=='a':
        logits=model(batch['input_ids'])
        loss=F.cross_entropy(logits.flatten(0,1),batch['labels'].flatten(),ignore_index=-100)
    elif shadow:
        prediction=model.predicted_mlp(batch['x']);mask=batch['labels']!=-100
        loss=((prediction-batch['teacher'])/model.selected.y_scale).square()[mask].mean()
    else:loss=model.ce(batch,alpha,plan['b_schedule']['vocabulary_loss_chunk'])
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite language loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],plan['training']['clip_norm'],error_if_nonfinite=True)
    optimizer.step();synchronize(device)
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started,
                lr=lr,alpha=alpha,teacher_mse=shadow,valid_tokens=int((batch['labels']!=-100).sum()),
                work=dict(getattr(model,'last_metrics',{})))


@torch.no_grad()
def evaluate(model,rows,route,device,batch_size,*,limit=None,alpha=1.,intervention=None):
    model.eval();total=0.;tokens=0;records=[];work={};started=time.monotonic()
    count=len(rows['labels']) if limit is None else min(limit,len(rows['labels']))
    for start in range(0,count,batch_size):
        ids=torch.arange(start,min(start+batch_size,count));batch=batch_from(rows,ids,device)
        if route=='a':
            x=batch['input_ids']
            if intervention=='context_reset':
                # Keep the immediate input token but reset earlier history at
                # every target. Never introduce future targets as context.
                logits=model(x.reshape(-1,1)).reshape(*x.shape,-1)
            else:logits=model(x,intervention='zero_core' if intervention=='zero_core' else None)
            loss=F.cross_entropy(logits.flatten(0,1),batch['labels'].flatten(),ignore_index=-100,reduction='none').reshape(batch['labels'].shape)
        else:loss=model.token_nll(batch,alpha=alpha)
        if not torch.isfinite(loss).all():raise FloatingPointError('nonfinite heldout NLL')
        sums=loss.sum(1).cpu().tolist();counts=(batch['labels']!=-100).sum(1).cpu().tolist()
        for i,nll,n in zip(ids.tolist(),sums,counts):
            records.append(dict(block_id=rows['block_ids'][i],document_id=rows['document_ids'][i],nll_sum=nll,tokens=n))
            total+=nll;tokens+=n
        for name,value in getattr(model,'last_metrics',{}).items():
            if name.startswith('work/') and not name.startswith('work/max_'):work[name]=work.get(name,0)+value
    synchronize(device)
    if not tokens:raise ValueError('empty independent evaluation')
    return dict(nll=total/tokens,ppl=math.exp(min(80,total/tokens)),tokens=tokens,blocks=count,
        nll_sum=total,seconds=time.monotonic()-started,rows=records,work=work,
        work_per_token={k:v/tokens for k,v in work.items()},alpha=alpha,intervention=intervention)


def checkpoint_state(model,optimizer,step,valid_tokens,identity,device,route):
    return dict(schema='tide-language-checkpoint-v1',model=cpu_tree(model.state_dict() if route=='a' else model.selected.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),step=step,valid_tokens=valid_tokens,identity=identity,
        rng=torch.get_rng_state(),python_rng=random.getstate(),
        accelerator_rng=cpu_tree(torch.npu.get_rng_state_all()) if device.type=='npu' else None,
        data_cursor=dict(next_update=step,sampling='counter-seeded replacement'),
        sequence_state='fresh per chunk and optimizer update; frozen prefix cache keyed by exact context')


def restore(path,model,optimizer,identity,device,route):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='tide-language-checkpoint-v1' or state['identity']!=identity:
        raise ValueError('exact resume identity mismatch; use a distinct initialization protocol for changes')
    (model if route=='a' else model.selected).load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    for slots in optimizer.state.values():
        for k,v in slots.items():
            if isinstance(v,torch.Tensor) and k!='step':slots[k]=v.to(device)
    torch.set_rng_state(state['rng']);random.setstate(state['python_rng'])
    if device.type=='npu':torch.npu.set_rng_state_all(state['accelerator_rng'])
    return state['step'],state['valid_tokens']


@torch.no_grad()
def prefix_cache(lm,rows,device,batch_size,progress=None):
    storage={k:[] for k in ('x','teacher','residual')};captured={}
    layer=lm.model.layers[-1]
    handles=[layer.post_attention_layernorm.register_forward_pre_hook(lambda module,args:captured.update(residual=args[0].detach())),
             layer.mlp.register_forward_pre_hook(lambda module,args:captured.update(x=args[0].detach())),
             layer.mlp.register_forward_hook(lambda module,args,out:captured.update(teacher=out.detach()))]
    try:
        for start in range(0,len(rows['labels']),batch_size):
            ids=torch.arange(start,min(start+batch_size,len(rows['labels'])));b=batch_from(rows,ids,device)
            mask=(b['labels']!=-100).long();positions=(mask.cumsum(-1)-1).clamp_min(0)
            lm.model(input_ids=b['input_ids'],attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
            for key in storage:storage[key].append(captured[key].cpu().clone())
            if progress:progress(min(start+batch_size,len(rows['labels'])))
    finally:
        for handle in handles:handle.remove()
    return dict(rows,**{k:torch.cat(v) for k,v in storage.items()})


def feature_statistics(rows):
    mask=rows['labels']!=-100;x=rows['x'][mask];y=rows['teacher'][mask]
    xm=x.mean(0);ym=y.mean(0);xs=(x-xm).square().mean().sqrt();ys=(y-ym).square().mean().sqrt()
    if not torch.isfinite(xs+ys) or min(float(xs),float(ys))<=1e-8:raise ValueError('degenerate training normalization')
    return dict(x_mean=xm,y_mean=ym,x_scale=xs,y_scale=ys)
