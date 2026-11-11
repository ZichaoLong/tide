"""Application-owned v6 training state; execution stays in public tidegraph."""
import copy
import math
import os
from pathlib import Path
import random
import resource
import time
import torch
from torch.nn import functional as F
from .language import TideLanguageModel, SmallTransformer, trainable_inventory
from .language_training import tensor_save
from .train import cpu_tree, synchronize


def core_options(config):
    options=copy.deepcopy(config['core'])
    if options.get('eager_implementation')=='native' and not options.get('native_library'):
        options['native_library']=os.environ['TIDE_LAB_NATIVE_LIBRARY']
    return options


def make_model(config, device):
    if config['kind']=='tide':
        model=TideLanguageModel(core_options(config),seed=config['seed'])
    elif config['kind']=='transformer':
        model=SmallTransformer(seed=config['seed'],**config['transformer'])
    elif config['kind']=='direct_graph':
        from .direct_graph_cost import DirectGraphLM
        model=DirectGraphLM(seed=config['seed'],**config['direct'])
    else:raise ValueError('unsupported campaign model')
    model=model.to(device)
    inventory=trainable_inventory(model)
    planned=config.get('planned_inventory',{}).get('lm_trainable')
    if planned is not None and inventory['trainable_parameters']!=planned:
        raise ValueError(f'parameter inventory mismatch: {planned} vs {inventory}')
    return model


def optimizer_for(model,config):
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params)!=len({id(p) for p in params}):raise AssertionError('optimizer aliases duplicated')
    c=config['training']
    return torch.optim.AdamW(params,lr=c['lr'],eps=c['eps'],weight_decay=c['weight_decay'],foreach=False)


def learning_rate(config,tokens):
    c=config['training'];warm=c['warmup_tokens'];total=c['target_tokens']
    if tokens<warm:factor=max(1,tokens)/warm
    else:factor=c['lr_floor_fraction']+(1-c['lr_floor_fraction'])*.5*(1+math.cos(math.pi*min(1,(tokens-warm)/max(1,total-warm))))
    return c['lr']*factor


def update(model,opt,batch,config,tokens,device):
    """Full device transfer is owned by caller; timing includes model/loss/VJP/AdamW."""
    model.train();synchronize(device);start=time.monotonic()
    count=int((batch['labels']!=-100).sum())
    if count<=0:raise ValueError('empty logical update')
    lr=learning_rate(config,tokens+count)
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True)
    logits=model(batch['input_ids'])
    loss=F.cross_entropy(logits.flatten(0,1),batch['labels'].flatten(),ignore_index=-100)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite next-token loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      config['training']['clip_norm'],error_if_nonfinite=True)
    opt.step();synchronize(device)
    return dict(loss=float(loss.detach()),tokens=count,gradient_norm=float(norm),lr=lr,
        seconds=time.monotonic()-start,work=dict(model.last_metrics))


def logical_update(model,opt,batches,config,tokens,device):
    """Same independent-sample global batch; physical microbatches accumulate VJP."""
    count=sum(int((b['labels']!=-100).sum()) for b in batches)
    if count<=0:raise ValueError('empty logical batch')
    model.train();synchronize(device);start=time.monotonic()
    lr=learning_rate(config,tokens+count)
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True);total=0.;work={}
    for b in batches:
        logits=model(b['input_ids'])
        loss=F.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum')/count
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite logical-batch loss')
        loss.backward();total+=float(loss.detach())
        for key,value in model.last_metrics.items():
            if '/max_' in key:work[key]=max(work.get(key,0),value)
            elif key!='activity/full_fraction':work[key]=work.get(key,0)+value
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      config['training']['clip_norm'],error_if_nonfinite=True)
    opt.step();synchronize(device)
    if work.get('work/candidates'):
        work['activity/full_fraction']=work['work/full']/work['work/candidates']
    return dict(loss=total,tokens=count,gradient_norm=float(norm),lr=lr,
        seconds=time.monotonic()-start,work=work,microbatches=len(batches))


def checkpoint(model,opt,progress,identity,device):
    return dict(schema='tide-language-campaign-checkpoint-v1',model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(opt.state_dict()),progress=copy.deepcopy(progress),identity=identity,
        torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        npu_rng=cpu_tree(torch.npu.get_rng_state_all()) if device.type=='npu' else None,
        private_state='empty at update boundaries; independent document blocks',
        scheduler='deterministic function of effective target tokens and frozen configuration')


def restore(path,model,opt,identity,device):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema']!='tide-language-campaign-checkpoint-v1' or state['identity']!=identity:
        raise ValueError('exact resume requires the same source/config/data/installation/backend identity')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    if device.type=='npu':torch.npu.set_rng_state_all(state['npu_rng'])
    return state['progress']


def memory(device):
    result={'resources/peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    if device.type=='npu':
        result.update({'resources/npu_allocated_mib':torch.npu.memory_allocated()/2**20,
            'resources/npu_reserved_mib':torch.npu.memory_reserved()/2**20,
            'resources/npu_peak_allocated_mib':torch.npu.max_memory_allocated()/2**20})
    return result


@torch.no_grad()
def evaluate(model,corpus,config,device,*,token_limit=None,deadline=None):
    model.eval();c=config['training'];limit=token_limit or c['eval_tokens']
    ids=corpus.evaluation_ids('validation',limit);records=[];total=0.;count=0
    started=time.monotonic();work={};batch_size=c.get('eval_batch',c['batch'])
    for offset in range(0,len(ids),batch_size):
        if deadline is not None and time.time()>=deadline:break
        subset=ids[offset:offset+batch_size]
        b=corpus.batch('validation',subset,device,token_limit=limit-count)
        logits=model(b['input_ids'])
        losses=F.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100,
                              reduction='none').reshape(b['labels'].shape)
        if not torch.isfinite(losses).all():raise FloatingPointError('nonfinite validation')
        sums=losses.sum(1).cpu().tolist();counts=(b['labels']!=-100).sum(1).cpu().tolist()
        for i,nll,n in zip(subset,sums,counts):
            if n:
                records.append(dict(corpus.metadata('validation',i),nll_sum=nll,tokens=n))
                total+=nll;count+=n
        for key,value in model.last_metrics.items():
            if key.startswith('work/'):
                work[key]=max(work.get(key,0),value) if '/max_' in key else work.get(key,0)+value
    synchronize(device)
    return dict(state='completed' if count==limit else 'budget_truncated',nll=total/count if count else None,
        nll_sum=total,tokens=count,target_tokens=limit,seconds=time.monotonic()-started,records=records,
        work=work,split='validation',test_access=False)


def exact_tree(a,b,path='root'):
    """Fresh-owner recovery compares values, shapes, dtypes and disconnected state."""
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b):
            raise AssertionError('exact recovery tensor mismatch: '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('exact recovery keys: '+path)
        for key in a:exact_tree(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError('exact recovery structure: '+path)
        for i,(x,y) in enumerate(zip(a,b)):exact_tree(x,y,path+f'.{i}')
    elif a!=b:raise AssertionError('exact recovery value: '+path)


def synthetic_batch(batch,length,seed,device):
    generator=torch.Generator().manual_seed(seed)
    ids=torch.randint(4,4096,(batch,length+1),generator=generator)
    return dict(input_ids=ids[:,:-1].clone().to(device),labels=ids[:,1:].clone().to(device))
