"""Small independent CPU-FP32 prefix-recompute anchor; never a large-point pass."""
from copy import deepcopy
from pathlib import Path
import time
from .config import resolve
from .catalog import model, local_path
from .inputs import create
from .runtime import setup
from .inference import generate
from .checkpoint import cache_state
from ..experiment.records import write_json, now


def compare(reference, candidate, *, atol=1e-5, rtol=1e-4):
    import torch
    a = reference.detach().float().cpu(); b = candidate.detach().float().cpu()
    if a.shape != b.shape:
        return dict(passed=False, reference_shape=list(a.shape), candidate_shape=list(b.shape))
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise FloatingPointError('nonfinite value in independent comparison')
    delta = (a-b).abs(); allowed = atol+rtol*a.abs()
    return dict(passed=bool((delta<=allowed).all()), max_absolute=float(delta.max()),
                max_relative=float((delta/a.abs().clamp_min(1e-12)).max()),
                outside_tolerance=int((delta>allowed).sum()), elements=a.numel(), atol=atol, rtol=rtol)


def independent_prefix(owner, ids, mask, count):
    import torch
    tokens=[]; logits=[]; prefix=ids
    with torch.inference_mode():
        for _ in range(count):
            positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,0)
            result=owner(input_ids=prefix, attention_mask=mask, position_ids=positions, use_cache=True)
            last=result.logits[:,-1,:]
            logits.append(last.cpu());token=last.argmax(-1,keepdim=True);tokens.append(token.cpu())
            prefix=torch.cat((prefix,token),dim=1)
            mask=torch.cat((mask,torch.ones((mask.shape[0],1),dtype=mask.dtype)),dim=1)
    return dict(tokens=torch.cat(tokens,dim=1),logits=logits,cache=cache_state(result.past_key_values))


def check(config, asset_root, output):
    import torch
    from transformers import AutoModelForCausalLM
    resolved=resolve(config);c=resolved['request']
    if c['task']['mode']!='inference' or c['task']['batch']>2 or c['task']['input_tokens']>32 or c['task']['output_tokens']>8:
        raise ValueError('check uses explicit inference configurations with B<=2, input<=32, output<=8')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tide-lm-check-v1',started=now(),state='running',configuration=c,
                scope='small same-weight CPU FP32 full-prefix schedule vs independent online cached candidate',
                tolerance=dict(atol=1e-5,rtol=1e-4),numeric='not_checked',discrete='not_checked')
    write_json(out/'check.json',record);started=time.perf_counter()
    try:
        candidate,tokenizer,device,backend,runtime=setup(resolved,asset_root)
        ids,mask,input_identity=create(c['task'],model(c['model'])['config'],tokenizer)
        actual=generate(candidate,ids,mask,c,device,backend,diagnostics=True)
        reference=AutoModelForCausalLM.from_pretrained(local_path(asset_root,c['model']),
            local_files_only=True,dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).eval()
        expected=independent_prefix(reference,ids,mask,c['task']['output_tokens'])
        if len(expected['logits'])!=len(actual['logits']) or len(expected['cache'])!=len(cache_state(actual['cache'])):
            raise RuntimeError('independent check: incomplete logits or layer cache count')
        checks={f'logits.{i}':compare(a,b) for i,(a,b) in enumerate(zip(expected['logits'],actual['logits']))}
        for i, (a,b) in enumerate(zip(expected['cache'],cache_state(actual['cache']))):
            for name,x,y in zip(('key','value'),a,b):
                checks[f'cache.{i}.{name}']=compare(x,y)
        equal=torch.equal(expected['tokens'],actual['tokens'])
        record.update(state='completed',numeric='passed' if all(x['passed'] for x in checks.values()) else 'failed',
            discrete='passed' if equal else 'failed', checks=checks,input=input_identity,runtime=runtime,
            generated_token_mismatches=int((expected['tokens']!=actual['tokens']).sum()),
            reference_precision='float32',candidate_precision=c['execution']['dtype'],
            routing='not_checked; token or cache differences prevent equivalent-work claims',
            seconds=time.perf_counter()-started,finished=now())
    except Exception as error:
        record.update(state='failed',error_kind=type(error).__name__,error=str(error)[:2000],finished=now())
        write_json(out/'check.json',record);raise
    write_json(out/'check.json',record)
    return record
