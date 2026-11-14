"""Application-owned, explicitly limited v10 source migration helpers."""
import copy
import hashlib
import torch


def validate_config_change(old,new):
    expected=copy.deepcopy(old)
    expected.setdefault('implementation',{})['aggregate_vjp']='ordered-v1'
    if new!=expected:
        raise ValueError('migration permits only implementation.aggregate_vjp=ordered-v1')
    if new['kind']!='direct_specialized' or new['implementation'].get('aggregation')!='neighbors':
        raise ValueError('only the diagnosed direct neighbors implementation may migrate')


def assert_finite_tree(value,path='state'):
    if isinstance(value,torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite checkpoint tensor: '+path)
    elif isinstance(value,dict):
        for k,v in value.items():assert_finite_tree(v,path+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):assert_finite_tree(v,path+'.'+str(i))


def training_batches(corpus,config,progress,device,stop_target):
    """Match campaign_stage's original milestone/cursor/tail convention.

    A short prefix stops at an UPDATE boundary, never at a new token milestone.
    """
    t=config['training'];global_batch=t.get('global_batch',t['batch']);micro=t['batch']
    target=min([m for m in t['milestones'] if m>progress['tokens']]+[stop_target])
    ids,cursor=corpus.sample_ids(progress['cursor'],global_batch,config['seed'])
    remain=target-progress['tokens'];batches=[];raw=0
    for i in range(0,len(ids),micro):
        subset=ids[i:i+micro];raw+=sum(int(corpus.blocks['train'][j,1]) for j in subset)
        if remain<=0:continue
        b=corpus.batch('train',subset,device,token_limit=remain)
        remain-=int((b['labels']!=-100).sum());batches.append(b)
    return batches,cursor,raw


def advance(progress,row,cursor,raw,config):
    progress.update(step=progress['step']+1,tokens=progress['tokens']+row['tokens'],cursor=cursor,
        dropped_tail_tokens=progress['dropped_tail_tokens']+max(0,raw-row['tokens']))
    if progress['tokens'] in config['training']['milestones']:
        progress['milestones'].append(progress['tokens'])


def gradient_signatures(model):
    values={}
    for name,p in model.named_parameters():
        if p.grad is None:values[name]=None;continue
        x=p.grad.detach().cpu().contiguous()
        values[name]=dict(shape=list(x.shape),dtype=str(x.dtype),
            sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    return values
