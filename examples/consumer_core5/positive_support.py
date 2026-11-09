#!/usr/bin/env python3
"""Installed public API: TL-REQ-003 row, explicit policy and first-order roots."""
import argparse
import json
from pathlib import Path
import torch
from tidegraph.weighted import allocate, observe_support


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--row",type=Path,required=True,help="Tracked JSON with scores/caps/target/temperature")
    p.add_argument("--device",default="cpu",help="Explicit CPU or queue-assigned logical npu:0")
    p.add_argument("--dtype",choices=("float32","float64"),default="float32")
    p.add_argument("--policy",choices=("strict","positive-floor-v1"),default="positive-floor-v1")
    args=p.parse_args()
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
    data=json.loads(args.row.read_text())
    make=lambda x:torch.tensor(x,dtype=getattr(torch,args.dtype),device=args.device,requires_grad=True)
    scores,caps,target,tau=map(make,(data["scores"],data["caps"],data["target"],data["temperature"]))
    with observe_support() as obs:
        result=allocate(scores,caps,target,target,tau,regularizer="ent",ent_alpha=1.,support_policy=args.policy)
        gradients=torch.autograd.grad(result.controls,(scores,caps,target,tau),scores.new_tensor([0,0,1e30,0]))
    print(json.dumps(dict(policy=args.policy,device=args.device,dtype=args.dtype,
        controls=result.controls.detach().cpu().tolist(),backward=result.backward,
        gradients=[v.detach().cpu().tolist() for v in gradients],support=obs.metrics()),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
