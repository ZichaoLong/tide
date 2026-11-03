"""Finite one-factor schedule/attention diagnostics, separate from throughput."""
from copy import deepcopy
from pathlib import Path
from .catalog import local_path, model
from .checkpoint import cache_state
from .config import resolve
from .inputs import create
from .inference import generate
from .runtime import setup
from .verification import compare
from ..experiment.records import now, write_json


def full_prefix(owner, ids, mask, count, device, *, last_only=False):
    import torch
    ids=ids.to(device); mask=mask.to(device); tokens=[]; logits=[]
    with torch.inference_mode():
        for _ in range(count):
            positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
            args={'logits_to_keep':1} if last_only else {}
            value=owner(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,**args)
            last=value.logits[:,-1,:];logits.append(last.cpu())
            token=last.argmax(-1,keepdim=True);tokens.append(token.cpu())
            ids=torch.cat((ids,token),dim=1)
            mask=torch.cat((mask,mask.new_ones((mask.shape[0],1))),dim=1)
    return dict(tokens=torch.cat(tokens,dim=1),logits=logits,cache=value.past_key_values)


def comparison(a,b):
    import torch
    left=cache_state(a['cache']);right=cache_state(b['cache'])
    if len(a['logits'])!=len(b['logits']) or len(left)!=len(right):
        raise RuntimeError('diagnostic: incomplete logits/KV layer count')
    fields=[(f'logits.{i}',x,y) for i,(x,y) in enumerate(zip(a['logits'],b['logits']))]
    fields += [(f'cache.{i}.{key}',x,y) for i,(xs,ys) in enumerate(zip(left,right))
               for key,x,y in zip(('key','value'),xs,ys)]
    checks={};first=None
    for key,x,y in fields:
        checks[key]=compare(x,y)
        if first is None and not checks[key]['passed']:
            x=x.detach().float().cpu();y=y.detach().float().cpu()
            if x.shape!=y.shape:raise RuntimeError('diagnostic: shape mismatch')
            index=tuple(((x-y).abs()>1e-5+1e-4*x.abs()).nonzero()[0].tolist())
            first=dict(field=key,index=list(index),reference=float(x[index]),candidate=float(y[index]))
    margins=[]
    for i,(x,y) in enumerate(zip(a['logits'],b['logits'])):
        av,ai=x.float().topk(2,dim=-1);bv,bi=y.float().topk(2,dim=-1)
        margins.append(dict(step=i,reference_top2=ai.tolist(),candidate_top2=bi.tolist(),
                            reference_margin=(av[:,0]-av[:,1]).tolist(),candidate_margin=(bv[:,0]-bv[:,1]).tolist()))
    return dict(numeric='passed' if first is None else 'failed',
                discrete='passed' if torch.equal(a['tokens'],b['tokens']) else 'failed',
                token_mismatches=int((a['tokens']!=b['tokens']).sum()),first_numeric_difference=first,
                checks=checks,top2=margins,
                schedule='each greedy trajectory is independent; later differences may include changed tokens')


def diagnose(config, assets, output):
    import torch
    from transformers import AutoModelForCausalLM
    c=resolve(config)['request'];t=c['task']
    if t['mode']!='inference' or t['batch']>2 or t['input_tokens']>32 or t['output_tokens']>8:
        raise ValueError('diagnose: inference B<=2, L<=32, output<=8 required')
    if c['execution']['placement']!='single':raise ValueError('diagnose: single-device factor isolation')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tide-lm-diagnostic-v1',state='running',configuration=c,started=now(),
                tolerance=dict(atol=1e-5,rtol=1e-4),pairs={},scope='diagnostics only; no large-point qualification')
    def save():write_json(out/'check.json',record)
    save()
    try:
        candidate_config=deepcopy(c);candidate_config['execution']['attention']='eager'
        owner,tokenizer,device,backend,runtime=setup(resolve(candidate_config),assets)
        ids,mask,input_id=create(t,model(c['model'])['config'],tokenizer)
        eager=generate(owner,ids,mask,candidate_config,device,backend,diagnostics=True)
        prefix=full_prefix(owner,ids,mask,t['output_tokens'],device)
        record['pairs']['same_dtype_device_eager_prefix_vs_cached']=comparison(prefix,eager);save()
        last=full_prefix(owner,ids,mask,t['output_tokens'],device,last_only=True)
        record['pairs']['same_eager_prefix_all_vs_last_logits']=comparison(prefix,last)
        record['pairs']['same_eager_last_logits_prefix_vs_cached']=comparison(last,eager);save()
        owner.set_attn_implementation('sdpa')
        sdpa=generate(owner,ids,mask,c,device,backend,diagnostics=True)
        record['pairs']['same_dtype_device_eager_vs_sdpa_cached']=comparison(eager,sdpa);save()
        reference=AutoModelForCausalLM.from_pretrained(local_path(assets,c['model']),local_files_only=True,
                    dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).eval()
        anchor=full_prefix(reference,ids,mask,t['output_tokens'],torch.device('cpu'))
        record['pairs']['cpu_fp32_prefix_vs_candidate_eager_prefix']=comparison(anchor,prefix)
        record.update(state='completed',finished=now(),runtime=runtime,input=input_id,
                      numeric='see_pairs',discrete='see_pairs',candidate_reference_schedule_input='none')
    except Exception as error:
        record.update(state='failed',finished=now(),error_kind=type(error).__name__,error=str(error)[:2000]);save();raise
    save();return record
