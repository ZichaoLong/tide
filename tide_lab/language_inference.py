"""Bounded full-model inference and a train-only unigram language control."""
import time
import torch
from .language import IntegratedMLP, load_qwen
from .language_checks import finite_delta
from .language_training import load_tokens
from .train import synchronize


@torch.no_grad()
def inference(model, args, plan, device):
    data,_=load_tokens(args.data,args.route,args.length)
    # Fixed validation prefix, no heldout-dependent recipe selection.
    tokens=data['validation']['input_ids'][:1,:min(16,args.length)].to(device)
    if args.route=='a':
        full=model(tokens)
        other=tokens.clone();other[:,8:]=(other[:,8:]+1)%model.embedding.num_embeddings
        causal=finite_delta(model(other)[:,:8],full[:,:8],0,0)
        causal_mechanisms=None
        if args.kind.startswith('tide'):
            from .language_checks import core_axes
            from .language_training import core_options
            with torch.enable_grad():
                causal_mechanisms=core_axes(model.encoder,core_options(plan,args.kind,args.depth),args.seed,device)
        elif causal['state']!='passed':raise AssertionError('Transformer future-token intervention changed prefix')
        paths={}
        if args.kind.startswith('tide'):
            for name,kwargs in [('chunked',{'chunk_size':4}),('streaming',{'implementation':'streaming'})]:
                alternate=model(tokens,**kwargs)
                paths[name]=finite_delta(alternate,full,plan['checks']['core_atol'],plan['checks']['core_rtol'])
                paths[name]['argmax_equal']=bool(torch.equal(alternate.argmax(-1),full.argmax(-1)))
        generated=tokens[:,:8].clone();start=time.monotonic()
        for _ in range(8):generated=torch.cat([generated,model(generated)[:,-1:].argmax(-1)],dim=1)
        synchronize(device)
        return dict(state='completed',causal=causal,causal_mechanisms=causal_mechanisms,paths=paths,
            prefix_token_ids=tokens[:,:8].cpu().tolist(),generated_token_ids=generated[:,8:].cpu().tolist(),
            greedy_seconds=time.monotonic()-start,scope='B1, 16 positions; greedy uses full-prefix replay',
            quality='generation is an inspectable diagnostic, not a language-quality pass')
    lm=load_qwen(args.asset_path,device)
    original=lm.model.layers[-1].mlp
    wrapper=None
    if not args.baseline_only:
        wrapper=IntegratedMLP(original,model.selected,alpha=1)
        lm.model.layers[-1].mlp=wrapper
    def fresh():
        if wrapper is not None:wrapper.history=None;wrapper.replay=False
    def forward(x,**kw):return lm(input_ids=x,return_dict=True,**kw)
    try:
        fresh();synchronize(device);start=time.monotonic()
        full=forward(tokens,use_cache=False).logits;synchronize(device);full_seconds=time.monotonic()-start
        changed=tokens.clone();changed[:,8:]=(changed[:,8:]+1)%lm.config.vocab_size
        fresh();causal=finite_delta(forward(changed,use_cache=False).logits[:,:8],full[:,:8],0,0)
        causal_mechanisms=None
        if args.kind.startswith('tide') and not args.baseline_only:
            from .language_checks import core_axes
            from .language_training import core_options
            with torch.enable_grad():
                causal_mechanisms=core_axes(model.selected.core,core_options(plan,args.kind,args.depth),args.seed,device)
        elif causal['state']!='passed':raise AssertionError('pointwise/native future-token intervention changed prefix')
        fresh()
        if wrapper is not None:wrapper.replay=True
        past=None;parts=[];start=time.monotonic()
        for t in range(tokens.shape[1]):
            out=forward(tokens[:,t:t+1],past_key_values=past,use_cache=True,
                        attention_mask=torch.ones(1,t+1,dtype=torch.long,device=device))
            parts.append(out.logits);past=out.past_key_values
        cached=torch.cat(parts,1);synchronize(device)
        comparison=finite_delta(cached,full,plan['checks']['qwen_atol'],plan['checks']['qwen_rtol'])
        comparison['argmax_equal']=bool(torch.equal(cached.argmax(-1),full.argmax(-1)))
        cache_seconds=time.monotonic()-start
        fresh()
        if wrapper is not None:wrapper.replay=True
        prefix=tokens[:,:8].clone();out=forward(prefix,use_cache=True);past=out.past_key_values;generated=[]
        start=time.monotonic()
        for i in range(8):
            nxt=out.logits[:,-1:].argmax(-1);generated.append(int(nxt.item()))
            out=forward(nxt,past_key_values=past,use_cache=True,
                        attention_mask=torch.ones(1,9+i,dtype=torch.long,device=device))
            past=out.past_key_values
        synchronize(device)
        if wrapper is not None and wrapper.original_calls:raise AssertionError('alpha1 invoked original MLP')
        return dict(state='completed',causal=causal,causal_mechanisms=causal_mechanisms,full_cache=comparison,full_seconds=full_seconds,
            cache_seconds=cache_seconds,greedy_seconds=time.monotonic()-start,
            prefix_token_ids=prefix.cpu().tolist(),generated_token_ids=[generated],
            original_mlp_calls=None if wrapper is None else wrapper.original_calls,
            scope='B1/L16 full Qwen, one last-MLP replacement, 8 greedy tokens; TIDE hidden-prefix replay with Qwen KV cache',
            implementation='replay recomputes TIDE history; not an optimized streaming or speedup claim',
            strict_note='finite failures retain original thresholds and unknown causes; argmax is separate')
    finally:lm.model.layers[-1].mlp=original


def unigram(args):
    data,lock=load_tokens(args.data,'a',args.length)
    train=data['train']['labels'].flatten();train=train[train!=-100]
    counts=torch.bincount(train,minlength=lock['routes']['a']['vocab_size']).double()+1.
    nll=-(counts/counts.sum()).log();evaluations={}
    for split in ('validation','test'):
        rows=[];total=0.;tokens=0
        for i,targets in enumerate(data[split]['labels']):
            valid=targets[targets!=-100];value=float(nll[valid].sum());total+=value;tokens+=len(valid)
            rows.append(dict(block_id=data[split]['block_ids'][i],document_id=data[split]['document_ids'][i],nll_sum=value,tokens=len(valid)))
        evaluations[split]=dict(nll=total/tokens,tokens=tokens,rows=rows)
    return dict(state='completed',evaluations=evaluations,smoothing=1.,source='A train token pool only',
        scope='token-frequency control, no learned context')
