"""Two separate invocations verify exact same-policy next window and optimizer."""
import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
import torch
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from positive_support_case import fixture, execute, start, optimizer


def plain(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu()
    if is_dataclass(value):
        return {f.name:plain(getattr(value,f.name)) for f in fields(value) if f.name!="stats"}
    if isinstance(value,dict):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(plain(v) for v in value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("write","resume"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--directory",type=Path,required=True)
    args=p.parse_args()
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    b,xs,stop=fixture(args.device); opt=optimizer(b)
    checkpoint=args.directory/"cut.pt"
    if args.phase=="write":
        args.directory.mkdir(parents=True,exist_ok=False)
        first=execute(b,start(b),xs,stop)
        objective(first).backward();opt.step();opt.zero_grad(set_to_none=True)
        head=execute(b,start(b),[x for x in xs if x.time<2],2)
        assert head.continuation.pending
        q=head.continuation.detach()
        save(checkpoint,b.graph,b.model,q,opt)
        torch.save(plain(dict(q=q,weights=b.model.state_dict(),optimizer=opt.state_dict())),
                   args.directory/"raw.pt")
    else:
        q=load(checkpoint,b.graph,b.model,opt)
        raw=torch.load(args.directory/"raw.pt",weights_only=True)
        equivalent(plain(dict(q=q,weights=b.model.state_dict(),optimizer=opt.state_dict())),
                   raw,atol=0,rtol=0)
    result=execute(b,q,[x for x in xs if x.time>=2],stop)
    objective(result).backward()
    grads=[p.grad for p in b.model.parameters()]
    opt.step()
    record=plain(dict(result=result,grads=grads,weights=b.model.state_dict(),optimizer=opt.state_dict()))
    if args.phase=="write":
        torch.save(record,args.directory/"next.pt")
    else:
        equivalent(record,torch.load(args.directory/"next.pt",weights_only=True),atol=0,rtol=0)
    print(dict(phase=args.phase,device=args.device,state="passed"),flush=True)


if __name__=="__main__":
    main()
