#!/usr/bin/env python3
"""Compare the original failed update with one local VJP ordering candidate.

This is a new diagnostic identity, not an implementation switch in a main job.
Only the last aggregate quotient backward changes; all forward operations,
parameters, input microbatches, loss scaling and the optimizer stay fixed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types


def main(a):
    diagnostic_source=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(a.training_source.resolve()),str(a.training_source.resolve()/'scripts')]
    import torch
    from torch.nn import functional as F
    from tide_lab.records import atomic_json,digest,now
    from tide_lab.train import runtime
    from tide_lab.campaign_runtime import make_model,optimizer_for,restore,learning_rate
    from tide_lab.direct_semantic_ops import norm
    from campaign_stage import identity
    # Import the candidate from this diagnostic source, never the original tree.
    import importlib.util
    spec=importlib.util.spec_from_file_location('division_probe',diagnostic_source/'scripts/probe_v10_division.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    OrderedDivide=module.OrderedDivide

    c=json.loads(a.config.read_text());device=runtime(c['device'],c['threads'],c['seed'])
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    ident=identity(a.config,c,a.data,'train')
    model=make_model(c,device);opt=optimizer_for(model,c)
    progress=restore(a.checkpoint,model,opt,ident,device)
    batches=torch.load(a.batches,map_location='cpu',weights_only=False)
    batches=[{k:v.to(device) for k,v in b.items()} for b in batches]
    count=sum(int((b['labels']!=-100).sum()) for b in batches)

    def repaired(self,j,z,w,incidence):
        if self.aggregation!='neighbors':raise ValueError('only original neighbors path covered')
        agg=self.aggregators[j];n,b,l,d=z.shape
        e=agg.e.index_select(0,getattr(self,'_classes_'+str(j)))[:,None,None,:]
        f=F.linear(F.silu(F.linear(norm(z),agg.a)+e),agg.b)
        weighted=w.unsqueeze(-1)*(z+self.plan['alpha_agg']*f)
        idx=getattr(self,'_neighbor_'+str(j));rnodes=len(self.members[j]);degree=self.slot_degrees[j]
        received=w.index_select(0,idx).reshape(rnodes,degree,b,l).sum(1)
        value=weighted.index_select(0,idx).reshape(rnodes,degree,b,l,d).sum(1)
        return OrderedDivide.apply(value,torch.where(received>0,received,torch.ones_like(received)).unsqueeze(-1)),received

    result=dict(schema='tide-v10-aggregate-vjp-probe-v1',state='running',created_at=now(),
        config_id=c['id'],original_identity=ident,
        diagnostic_source=json.loads((diagnostic_source/'.tide-source.json').read_text())['commit'],
        checkpoint_sha256=digest(a.checkpoint),batches_sha256=digest(a.batches),progress=progress,
        scope='One failed logical batch, first-order VJP, finite optimizer update only; no convergence claim.',variants={})
    atomic_json(a.output/'result.json',result)
    hashes={}
    for variant in ('original','ordered_aggregate_quotient'):
        if variant!='original':model.aggregate=types.MethodType(repaired,model)
        opt.zero_grad(set_to_none=True);model.train();model.set_progress(progress['tokens'])
        loss_sum=0.;fingerprints=[];work=[]
        for b in batches:
            logits=model(b['input_ids']);work.append(dict(model.last_metrics))
            fingerprints.append(hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            loss=F.cross_entropy(logits.flatten(0,1),b['labels'].flatten(),ignore_index=-100,reduction='sum')/count
            if not bool(torch.isfinite(loss)):raise FloatingPointError('probe nonfinite forward loss')
            loss.backward();loss_sum+=float(loss.detach())
        hashes[variant]=fingerprints
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        row=dict(loss=loss_sum,nonfinite_parameters=bad,forward_sha256=fingerprints,
                 support_errors=sum(w['discrete/positive_support_errors'] for w in work),
                 floor_events=sum(w['numerical/positive_floor_events'] for w in work))
        try:
            norm_value=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['clip_norm'],error_if_nonfinite=True)
            row['gradient_norm']=float(norm_value);row['gradient_acceptance']='passed'
        except RuntimeError as exc:
            row.update(gradient_acceptance='failed',error=repr(exc))
        if variant!='original' and not bad and row['gradient_acceptance']=='passed':
            before=model.embedding.weight.detach().clone()
            for group in opt.param_groups:group['lr']=learning_rate(c,progress['tokens']+count)
            opt.step()
            row['optimizer']=dict(parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()),
                embedding_changed=not bool(torch.equal(before,model.embedding.weight)),
                tied_head_alias=model.embedding.weight is model.head.weight)
        result['variants'][variant]=row
        atomic_json(a.output/'result.json',result)
    result.update(state='completed',seconds=time.monotonic()-started,
        forward_bitwise_equal_all_microbatches=hashes['original']==hashes['ordered_aggregate_quotient'],
        axes=dict(execution='completed',strict_numerical='local_vjp_reference_checked_separately; old_failures_retained',
                  discrete_semantics='see_support_errors',recovery='checkpoint_verified; new_candidate_resume_unchecked',
                  learning='unchecked_no_training_trajectory'))
    atomic_json(a.output/'result.json',result)
    print(json.dumps(dict(state='completed',seconds=result['seconds'],config_id=c['id'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-source','config','data','checkpoint','batches','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
