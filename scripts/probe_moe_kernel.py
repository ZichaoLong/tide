#!/usr/bin/env python3
"""Bounded capability probe, not a training or speed qualification."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tide_lab.train import runtime,installed_identity
from tide_lab.records import atomic_json,now
from tide_lab.language_checks import finite_delta


def main(a):
    dev=runtime('npu:0',4,19)
    import torch_npu
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(state='running',started_at=now(),installation=installed_identity(),cases={})
    x=torch.randn(9,32,requires_grad=True);w=torch.randn(4,32,48,requires_grad=True)
    sizes=[3,0,4,2];g=torch.tensor([3,3,7,9],dtype=torch.int64,device=dev)
    y=torch.cat([t@we for t,we in zip(x.split(sizes),w)],0)
    v=torch.randn_like(y);y.backward(v)
    xn=x.detach().to(dev).requires_grad_();wn=w.detach().to(dev).requires_grad_()
    try:
        yn=torch_npu.npu_grouped_matmul([xn],[wn],group_list=g,split_item=3,group_type=0)[0]
        torch.npu.synchronize()
        result['cases']['forward']=dict(state='completed',delta=finite_delta(yn,y,1e-6,1e-5),
                                        shape=list(yn.shape),requires_grad=yn.requires_grad)
        try:
            yn.backward(v.to(dev));torch.npu.synchronize()
            result['cases']['autograd']=dict(state='completed',dx=finite_delta(xn.grad,x.grad,1e-6,1e-5),
                                             dw=finite_delta(wn.grad,w.grad,1e-6,1e-5))
        except Exception as e:result['cases']['autograd']=dict(state='failed',error=repr(e))
        try:
            dx=torch_npu.npu_grouped_matmul([v.to(dev)],[wn.detach().transpose(1,2)],group_list=g,split_item=3,group_type=0)[0]
            dw=torch_npu.npu_grouped_matmul_add(torch.zeros_like(wn),xn.detach(),v.to(dev),g,
                transpose_x=True,transpose_weight=False,group_type=2,group_list_type=0)
            torch.npu.synchronize()
            result['cases']['explicit_vjp']=dict(state='completed',dx=finite_delta(dx,x.grad,1e-6,1e-5),
                                                 dw=finite_delta(dw,w.grad,1e-6,1e-5))
        except Exception as e:result['cases']['explicit_vjp']=dict(state='failed',error=repr(e))
    except Exception as e:result['cases']['forward']=dict(state='failed',error=repr(e))
    result.update(state='completed',finished_at=now(),scope='operator capability only; failures retained')
    atomic_json(a.output/'result.json',result)
    print(json.dumps(result,default=str),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
