#!/usr/bin/env python3
"""Independent reference check of the explicitly selected production quotient VJP."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.records import atomic_json,digest,now
from tide_lab.train import runtime,installed_identity,source_identity
from tide_lab.ordered_divide import OrderedDivide


def report(x):
    x=x.detach().cpu();f=torch.isfinite(x);v=x[f].double()
    return dict(shape=list(x.shape),nan=int(x.isnan().sum()),inf=int(x.isinf().sum()),
                finite_max_abs=float(v.abs().max()) if v.numel() else None,
                finite_l2=float(torch.linalg.vector_norm(v)) if v.numel() else 0.,
                values=[float(t) if math.isfinite(float(t)) else str(float(t)) for t in x.flatten()] if x.numel()<=8 else None)


def comparison(value,reference):
    x=value.detach().cpu().double();r=reference.detach().cpu().double()
    finite=bool(torch.isfinite(x).all()) and bool(torch.isfinite(r).all())
    if not finite:return dict(state='nonfinite',atol=1e-6,rtol=1e-5,value=report(x),reference=report(r))
    delta=x-r;bad=delta.abs()>1e-6+1e-5*r.abs()
    return dict(state='failed' if bool(bad.any()) else 'passed',atol=1e-6,rtol=1e-5,
                failed_elements=int(bad.sum()),max_abs=float(delta.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(r).clamp_min(1e-300)))


def evaluate(a,b,g,device,dtype,method):
    aa=a.to(device=device,dtype=dtype).detach().requires_grad_(True)
    bb=b.to(device=device,dtype=dtype).detach().requires_grad_(True)
    yy=aa/bb if method=='ordinary' else OrderedDivide.apply(aa,bb)
    da,db=torch.autograd.grad(yy,(aa,bb),g.to(device=device,dtype=dtype))
    return tuple(x.detach().cpu() for x in (yy,da,db))


def main(args):
    device=runtime(args.device,2,11);started=time.monotonic();rows=[]
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[('tiny_nonzero_vjp',5e-38,1.2e-38,1e-5),
           ('tiny_zero_vjp',5e-38,1.2e-38,0.),
           ('ordinary',5e-8,1.2e-8,1e-5),
           ('genuine_overflow',5e-38,1.2e-38,10.)]
    for name,aa,bb,gg in cases:
        a,b,g=[torch.tensor(v,dtype=torch.float32) for v in (aa,bb,gg)]
        ref=evaluate(a,b,g,'cpu',torch.float64,'ordinary')
        for method in ('ordinary','ordered'):
            y,da,db=evaluate(a,b,g,device,torch.float32,method)
            rows.append(dict(case=name,method=method,forward=report(y),grad_a=report(da),grad_b=report(db),
                reference_grad_b=report(ref[2]),
                checks={k:comparison(x,r) for k,x,r in zip(('forward','grad_a','grad_b'),(y,da,db),ref)}))
    a=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    b=torch.rand(2,1,dtype=torch.float64,requires_grad=True)+.3
    gradcheck=bool(torch.autograd.gradcheck(OrderedDivide.apply,(a,b)))
    result=dict(schema='tide-v10-division-probe-v1',state='completed',created_at=now(),
        source=source_identity(),installation=installed_identity(),device=str(device),
        cases=rows,broadcast_fp64_gradcheck=gradcheck,
        scope='Local first-order arithmetic only; no training convergence or complete optimizer/recovery qualification.')
    if args.witness:
        w=torch.load(args.witness,map_location='cpu',weights_only=False)
        if w['node']!='DivBackward0':raise ValueError('only DivBackward0 witnesses supported')
        a=w['saved']['_saved_self'];b=w['saved']['_saved_other'];g=w['grad_outputs'][0]
        ref=evaluate(a,b,g,'cpu',torch.float64,'ordinary')
        variants={}
        for method in ('ordinary','ordered'):
            out=evaluate(a,b,g,device,torch.float32,method)
            variants[method]={k:dict(report(x),check=comparison(x,r)) for k,x,r in zip(('forward','grad_a','grad_b'),out,ref)}
            if method=='ordinary':old_y=out[0]
            else:variants[method]['forward_bitwise_equal_to_ordinary']=bool(torch.equal(old_y,out[0]))
        u=a/b
        overflowing=torch.isinf(u/b)
        coordinates=overflowing.nonzero()[:8];bb=b.expand_as(a);examples=[]
        for index in coordinates:
            ix=tuple(index.tolist())
            examples.append(dict(index=list(ix),numerator=float(a[ix]),denominator=float(bb[ix]),
                quotient=float(u[ix]),upstream=float(g[ix]),
                unsafe_quotient_over_denominator=str(float((u/b)[ix])),
                fp64_element_grad_b=float(-g[ix].double()*a[ix].double()/bb[ix].double().square())))
        result['witness']=dict(sha256=digest(args.witness),microbatch=w['microbatch'],
            node=w['node'],upstream_finite=bool(torch.isfinite(g).all()),examples=examples,
            overflowing_intermediate_elements=int(overflowing.sum()),variants=variants)
    result['seconds']=time.monotonic()-started
    atomic_json(args.output/'result.json',result)
    print(json.dumps(dict(state=result['state'],output=str(args.output),seconds=result['seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=('cpu','npu:0'),default='cpu')
    p.add_argument('--witness',type=Path);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
