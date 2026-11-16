#!/usr/bin/env python3
"""Bounded CPU/NPU qualification of the captured quad failure and local repair."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,digest,now,Metrics,project_to_lens
from tide_lab.train import runtime,source_identity,installed_identity
from tide_lab.direct_semantic_ops import allocation
from tide_lab.language_checks import finite_delta
from tests.test_quad_pivot import reference,stress_cases


def main(a):
    device=runtime(a.device,2,11);a.output.mkdir(parents=True,exist_ok=False)
    run='tl-v12-repair-qualification-'+a.device.split(':')[0]
    source=Path(__file__).resolve().parents[1]
    r=dict(schema='tide-hb-v12-quad-repair-check-v1',state='running',started_at=now(),
        source=source_identity(),installation=installed_identity(),device=a.device,
        witness_sha256=digest(a.witness),original_feasibility_threshold=2e-5,
        implementation_sha256={p:digest(source/p) for p in ('tide_lab/direct_semantic_ops.py','tide_lab/disk_usage.py')},
        scope='Captured row, independent scalar CPU FP64 roots, exact witness VJP, finite bounded stress and allocator-only timing; full-model recovery/learning separate.',rows=[])
    atomic_json(a.output/'result.json',r)
    atomic_json(a.output/'manifest.json',dict(project='tide-learning-lab',run_id=run,name=run,
        group='TL-HB-NUMERICAL-v12',study_id='TL-HB-NUMERICAL-v12'))
    metrics=Metrics(a.output,run)
    try:
        w=torch.load(a.witness,map_location='cpu',weights_only=False)
        for centered in (False,True):
            t=w['t'].to(device);mu=w['mu'].to(device);b=w['lower'].to(device)
            if centered:t=t-t.amax(-1,keepdim=True)
            t=t.detach().requires_grad_();mu=mu.detach().requires_grad_();b=b.detach().requires_grad_()
            c,res=allocation(t,mu,b,b,w['tau'])
            ref=torch.tensor([reference(t[0].detach().cpu().tolist(),mu[0].detach().cpu().tolist(),b.item(),w['tau'])],dtype=torch.float64)
            forward=finite_delta(c,ref,1e-6,1e-5)
            g=torch.arange(64,device=device,dtype=t.dtype)[None,:]
            dt,dm,db=torch.autograd.grad((c*g).sum(),(t,mu,b))
            want_mu=torch.zeros_like(dm);want_mu[0,26]=26-55
            gradients=dict(score=finite_delta(dt,torch.zeros_like(dt),1e-6,1e-5),
                cap=finite_delta(dm,want_mu,1e-6,1e-5),budget=finite_delta(db,torch.full_like(db,55),1e-6,1e-5))
            inside=(c>0)&(c<mu);expected=torch.zeros_like(inside);expected[0,55]=True
            item=dict(centered=centered,forward=forward,vjp=gradients,max_residual=float(res.max()),
                interior_mask_equal=bool(torch.equal(inside,expected)),nonzero_values=c[c>0].detach().cpu().tolist())
            r['rows'].append(item)
            if forward['state']!='passed' or not item['interior_mask_equal'] or item['max_residual']>2e-5 or any(v['state']!='passed' for v in gradients.values()):
                raise AssertionError('captured-row forward/interior/VJP qualification failed')
        stress=[]
        for offset,tau,t,mu,b in stress_cases():
            ref=torch.tensor([reference(s,m,q[0],tau) for s,m,q in zip(t.tolist(),mu.tolist(),b.tolist())],dtype=torch.float64)
            c,res=allocation(t.to(device),mu.to(device),b.to(device),b.to(device),tau)
            checked=finite_delta(c,ref,1e-6,1e-5)
            item=dict(offset=offset,tau=tau,shape=list(t.shape),forward=checked,max_residual=float(res.max()),
                finite=bool(torch.isfinite(c).all()),bounds=bool(((c>=0)&(c<=mu.to(device))).all()),
                absent_exact_zero=bool((c[mu.to(device)==0]==0).all()))
            stress.append(item)
            if not item['finite'] or not item['bounds'] or not item['absent_exact_zero'] or item['max_residual']>2e-5 or checked['state']!='passed':
                raise AssertionError('independent stress failed: '+str(item))
        r['stress']=stress
        # Matched local performance only, not end-to-end training speedup.
        spec=importlib.util.spec_from_file_location('original_direct_ops',a.parent_source/'tide_lab/direct_semantic_ops.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        gen=torch.Generator().manual_seed(17);t=torch.randn(4096,64,generator=gen).to(device)
        mu=torch.ones_like(t);b=torch.full((4096,1),1.5,device=device)
        def sync():
            if device.type=='npu':torch.npu.synchronize()
        times={}
        for name,fn in [('original',old.allocation),('pivot',allocation)]:
            fn(t,mu,b,b,.2);sync();start=time.monotonic()
            for _ in range(8):fn(t,mu,b,b,.2)
            sync();times[name]=(time.monotonic()-start)/8
        r['allocator_seconds']=times;r['allocator_time_ratio']=times['pivot']/times['original']
        metrics.log(0,{'repair/witness_residual':max(x['max_residual'] for x in r['rows']),
            'repair/stress_max_residual':max(x['max_residual'] for x in stress),'repair/stress_rows':sum(x['shape'][0] for x in stress),
            'repair/allocator_time_ratio':r['allocator_time_ratio']})
        r.update(state='completed',axes=dict(execution='passed',local_strict_numerical='passed',
            witness_discrete_and_vjp='passed',full_model_recovery='required_per_chain',learning='unchecked'))
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        r['finished_at']=now();metrics.close();atomic_json(a.output/'result.json',r)
        atomic_json(a.output/'completed.json',dict(producer_finished=True,state=r['state']));project_to_lens(a.output,a.lens_cli)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('output','witness','parent-source'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--lens-cli',required=True)
    main(p.parse_args())
