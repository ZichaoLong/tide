"""Small full-parameter anchor: manual CPU CE and first-step AdamW equations."""
from pathlib import Path
import time
from .config import resolve
from .catalog import model, local_path
from .runtime import setup
from .inputs import create
from .training import optimizer_for, update
from .verification import compare
from ..experiment.records import now, write_json


def check(config, asset_root, output):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    c=resolve(config)['request']
    if c['model']!='qwen05' or c['task']['mode']!='full' or c['task']['input_tokens']>32 or c['task']['global_batch']!=1:
        raise ValueError('training-check: fixed qwen05 full-parameter B1 and L<=32 anchor')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='tide-lm-check-v1',state='running',started=now(),configuration=c,
                scope='CPU FP32 explicit shifted CE/first AdamW equations vs candidate HF loss/autocast/optimizer',
                numeric='not_checked',optimizer='not_checked')
    write_json(out/'check.json',record);begin=time.perf_counter()
    try:
        owner,tokenizer,device,backend,runtime=setup(resolve(c),asset_root)
        ids,mask,input_identity=create(c['task'],model(c['model'])['config'],tokenizer)
        reference=AutoModelForCausalLM.from_pretrained(local_path(asset_root,c['model']),local_files_only=True,
                        dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).train()
        logits=reference(input_ids=ids,attention_mask=mask,use_cache=False).logits
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
        loss.backward()
        expected=dict(reference.named_parameters());checks={};missing=[]
        def observe(phase,candidate,opt):
            for name,p in candidate.named_parameters():
                ref=expected[name];grad=ref.grad
                if phase=='gradient':
                    if (grad is None)!=(p.grad is None):missing.append(name)
                    elif grad is not None:checks['gradient.'+name]=compare(grad,p.grad)
                elif grad is not None:
                    # AdamW step 1, no weight decay; epsilon is outside sqrt.
                    projected=ref.detach()-1e-4*grad/(grad.abs()+1e-8)
                    checks['parameter.'+name]=compare(projected,p)
                    checks['momentum.'+name]=compare(grad*.1,opt.state[p]['exp_avg'])
                    checks['variance.'+name]=compare(grad.square()*.001,opt.state[p]['exp_avg_sq'])
        observation=update(owner,optimizer_for(owner),ids,mask,c,device,backend,observer=observe)
        record.update(state='completed',finished=now(),seconds=time.perf_counter()-begin,runtime=runtime,
            input=input_identity,reference_loss=float(loss.detach()),candidate=observation,checks=checks,
            disconnected_mismatches=missing,numeric='passed' if not missing and all(v['passed'] for v in checks.values()) else 'failed',
            optimizer='passed' if all(v['passed'] for k,v in checks.items() if not k.startswith('gradient.')) else 'failed')
    except Exception as error:
        record.update(state='failed',error_kind=type(error).__name__,error=str(error)[:2000],finished=now())
        write_json(out/'check.json',record);raise
    write_json(out/'check.json',record)
    return record
