#!/usr/bin/env python3
"""Bounded read-only-weight causal/packing diagnosis; no language training."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,now
from tide_lab.train import runtime,source_identity,cpu_tree
from tide_lab.campaign_runtime import make_model,optimizer_for,update,synthetic_batch
from tide_lab.language_checks import finite_delta


def causal_diagnosis(model,c,device):
    b=synthetic_batch(2,3,991,device);x=b['input_ids'];changed=x.clone();changed[:,2]=(changed[:,2]+17)%4096
    model.eval()
    def capture():
        captured={};handles=[]
        for i,block in enumerate(model.blocks):
            for name in ('norm1','out','norm2','ffn'):
                key=f'{i}.{name}'
                handles.append(getattr(block,name).register_forward_hook(
                    lambda mod,inp,out,k=key:captured.__setitem__(k,out.detach().cpu().clone())))
        with torch.no_grad():y=model(x)
        original=dict(captured);routes=[block.ffn.stats['indices'].reshape(2,3,2)[:,:2].cpu().clone() for block in model.blocks]
        with torch.no_grad():z=model(changed)
        new=dict(captured);other=[block.ffn.stats['indices'].reshape(2,3,2)[:,:2].cpu().clone() for block in model.blocks]
        for h in handles:h.remove()
        return y,z,original,new,routes,other
    y,z,a,d,r,s=capture()
    deltas={k:finite_delta(a[k][:,:2],d[k][:,:2],0,0) for k in a}
    result=dict(causal_zero_tolerance=finite_delta(y[:,:2],z[:,:2],0,0),
        causal_existing_finite_threshold=finite_delta(y[:,:2],z[:,:2],1e-6,1e-5),layer_prefix_deltas=deltas,
        first_different_module=next((k for k,v in deltas.items() if v['state']=='failed'),None),
        prefix_routing_identical=all(torch.equal(q,w) for q,w in zip(r,s)))
    capture_input={}
    def retain(mod,inp,out):out.retain_grad();capture_input['x']=out
    handle=model.embedding.register_forward_hook(retain);model.zero_grad(set_to_none=True)
    y=model(x);y[:,:2].sum().backward();future=capture_input['x'].grad[:,2:]
    result['future_input_vjp_max_abs']=float(future.abs().max());handle.remove();model.zero_grad(set_to_none=True)
    implementation=[block.ffn.implementation for block in model.blocks]
    try:
        for block in model.blocks:block.ffn.implementation='reference'
        with torch.no_grad():ref_y=model(x);ref_z=model(changed)
        result['fixed_shape_npu_reference_causal']=finite_delta(ref_y[:,:2],ref_z[:,:2],0,0)
        result['bucket_vs_fixed_shape_forward']=finite_delta(y,ref_y,1e-6,1e-5)
    finally:
        for block,impl in zip(model.blocks,implementation):block.ffn.implementation=impl
    reference=make_model(c,torch.device('cpu'));reference.load_state_dict(cpu_tree(model.state_dict()))
    reference.eval()
    with torch.no_grad():cpu_y=reference(x.cpu());cpu_z=reference(changed.cpu())
    result['cpu_reference_causal']=finite_delta(cpu_y[:,:2],cpu_z[:,:2],0,0)
    result['mechanism_state']='passed' if (result['prefix_routing_identical'] and result['future_input_vjp_max_abs']==0
        and result['fixed_shape_npu_reference_causal']['state']=='passed' and result['cpu_reference_causal']['state']=='passed') else 'failed'
    result['scope']='B2 L3 last-token intervention; zero-threshold packing-dependent rounding retained, not bitwise causal certification'
    return result


def main(a):
    dev=runtime('npu:0',4,33);a.output.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',source=source_identity(),started_at=now(),cases=[])
    for path in a.configs:
        c=json.loads(path.read_text());model=make_model(c,dev);opt=optimizer_for(model,c)
        update(model,opt,synthetic_batch(2,3,991,dev),c,0,dev)
        item=dict(id=c['id'],**causal_diagnosis(model,c,dev));report['cases'].append(item)
        atomic_json(a.output/'result.json',report)
        print(json.dumps({k:v for k,v in item.items() if k!='layer_prefix_deltas'}),flush=True)
        del model,opt;import gc;gc.collect();torch.npu.empty_cache()
    report.update(state='completed',finished_at=now());atomic_json(a.output/'result.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
