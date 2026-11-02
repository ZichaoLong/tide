"""Explicit online cached generation; all operator preparation and returns timed."""
import time
from .runtime import synchronize


def generate(owner,host_ids,host_mask,c,device,backend,*,output_tokens=None,diagnostics=False):
    import torch
    e=c['execution'];t=c['task'];count=e['devices'];out=output_tokens or t['output_tokens']
    synchronize(backend,count);begin=time.perf_counter()
    ids=host_ids.to(device);mask=host_mask.to(device)
    cache=None;last=None;finite=torch.ones((),dtype=torch.bool,device=device)
    chunk=e['prefill_chunk'];length=ids.shape[1];calls=[];logits=[]
    with torch.inference_mode():
        for start in range(0,length,chunk):
            end=min(length,start+chunk)
            positions=mask[:,:end].cumsum(-1)-1
            positions.masked_fill_(mask[:,:end]==0,0)
            result=owner(input_ids=ids[:,start:end],attention_mask=mask[:,:end],
                position_ids=positions[:,start:end],past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values;last=result.logits[:,-1,:]
            finite &= torch.isfinite(last).all()
            calls.append(dict(phase='prefill',sequences=ids.shape[0],tokens_per_sequence=end-start,rows=ids.shape[0]*(end-start)))
        if diagnostics:logits.append(last.detach().cpu())
        token=last.argmax(-1,keepdim=True);generated=[token];done=torch.zeros(ids.shape[0],dtype=torch.bool,device=device)
        synchronize(backend,count);prefill_end=time.perf_counter()
        active=[];eos=owner.config.eos_token_id
        eos_ids=eos if isinstance(eos,list) else [eos]
        natural=t['input_kind']=='varied_text'
        if natural:
            for value in eos_ids:
                if value is not None:done |= token[:,0]==value
        active.append((~done).sum())
        for _ in range(out-1):
            if natural and bool(done.all()):break
            mask=torch.cat((mask,(~done if natural else torch.ones_like(done)).to(mask.dtype)[:,None]),dim=1)
            position=mask.sum(-1,keepdim=True)-1
            result=owner(input_ids=token,attention_mask=mask,position_ids=position,
                         past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values;last=result.logits[:,-1,:]
            finite &= torch.isfinite(last).all()
            token=last.argmax(-1,keepdim=True)
            generated.append(token)
            if diagnostics:logits.append(last.detach().cpu())
            active.append((~done).sum())
            if natural:
                for value in eos_ids:
                    if value is not None:done |= token[:,0]==value
        tokens=torch.cat(generated,dim=1).cpu()
        prefill_active=[int((host_mask[:,x:x+chunk].sum(-1)>0).sum()) for x in range(0,length,chunk)]
        counts=torch.stack(active).cpu().tolist()
        all_finite=bool(finite.cpu())
        synchronize(backend,count);finish=time.perf_counter()
    if not all_finite:raise FloatingPointError('nonfinite generation logits')
    # First token counts even if it is EOS; later iterations count live rows.
    produced=ids.shape[0]+sum(counts[1:]) if natural else tokens.numel()
    return dict(tokens=tokens,cache=cache,mask=mask,last_token=token,logits=logits,
        metrics=dict(total_seconds=finish-begin,prefill_seconds=prefill_end-begin,
                     decode_seconds=finish-prefill_end,output_tokens=produced,physical_output_rows=tokens.numel(),
                     output_tokens_per_s=produced/(finish-begin),input_tokens=int(host_mask.sum()),
                     prefill_tokens_per_s=int(host_mask.sum())/(prefill_end-begin),
                     decode_tokens_per_s=max(0,produced-ids.shape[0])/max(1e-12,finish-prefill_end),
                     mean_tpot_seconds=(finish-prefill_end)/max(1,tokens.shape[1]-1)),
        work=dict(logical_batch=ids.shape[0],prefill_calls=calls,prefill_active_sequences=prefill_active,decode_calls=tokens.shape[1]-1,
                  decode_rows_per_call=ids.shape[0],active_sequences=counts,
                  inactive_rows_policy='retained padded rows; no physical batch compaction' if natural else 'all active',
                  finite_logits_check='included; device reductions, one final host scalar',
                  return_contract='generated token IDs plus scalar statistics; KV remains on owner device'))
