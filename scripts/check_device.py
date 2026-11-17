"""Small real-device mechanism checks; numerical differences are explicit data."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from tide import ModelConfig,TopologyConfig,TideModel,save_checkpoint,load_checkpoint,detach_state


def difference(actual,reference):
    a,b=actual.detach().float().cpu(),reference.detach().float().cpu()
    return dict(max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-20)),
                finite=bool(torch.isfinite(a).all()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="npu:0")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if device.type=="npu":
        torch.npu.set_device(device)
    configs=[
        ModelConfig(topology=TopologyConfig(n=8,layers=4),d=64),
        ModelConfig(topology=TopologyConfig(n=8,layers=4),d=64,memory="S",allocation="pool",
                    aggregation="incoming_wide",sharing="region_shared"),
        ModelConfig(topology=TopologyConfig(kind="exchange",n=8,depth=1,hubs=4),d=64,memory="X",
                    allocation="shannon",aggregation="incoming_light_64",sharing="layer_shared"),
        ModelConfig(topology=TopologyConfig(kind="multiscale",n=8,profile="compact-5-dd4",hubs=4,
                    timing="timed"),d=64,memory="X",sharing="region_shared"),
    ]
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    rows=[]
    for index,cfg in enumerate(configs):
        torch.manual_seed(514+index)
        cpu=TideModel(cfg)
        candidate=copy.deepcopy(cpu).to(device)
        x=torch.randn(2,5,64,requires_grad=True)
        y=x.detach().to(device).requires_grad_()
        row=dict(config=cfg.to_dict(),device=args.device,precision="fp32",state="running")
        try:
            reference=cpu(x)
            seed=torch.randn_like(reference.readout())
            (reference.readout()*seed).mean().backward()
            result=candidate(y)
            loss=(result.readout()*seed.to(device)).mean()
            loss.backward()
            row["output"]=difference(result.readout(),reference.readout())
            row["input_gradient"]=difference(y.grad,x.grad)
            row["strength"]=difference(result.strengths,reference.strengths)
            row["output_support_disagreements"]=int(((result.strengths.detach().cpu()>0)!=(reference.strengths>0)).sum())
            grads=[]; none=[]
            for (name,a),(_,b) in zip(candidate.named_parameters(),cpu.named_parameters()):
                if a.grad is None or b.grad is None:
                    if (a.grad is None)!=(b.grad is None):
                        none.append(name)
                else:
                    grads.append((name,difference(a.grad,b.grad)))
            row["disconnected_gradient_mismatches"]=none
            row["parameter_gradient_max_relative_l2"]=max(v["relative_l2"] for _,v in grads)
            row["largest_gradient_differences"]=sorted(grads,key=lambda item:item[1]["relative_l2"],reverse=True)[:5]
            with torch.no_grad():
                a=candidate(y[:,:2]); b=candidate(y[:,2:],a.state)
                row["continuation"]=difference(torch.cat((a.readout(),b.readout()),1),result.readout())
            opt=torch.optim.AdamW(candidate.parameters(),lr=1e-4,foreach=True)
            losses=[]
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                z=candidate(y); objective=z.readout().float().square().mean()
                objective.backward(); opt.step(); losses.append(float(objective.detach()))
            row["three_step_losses"]=losses
            with tempfile.TemporaryDirectory() as tmp:
                save_checkpoint(Path(tmp)/"check.pt",candidate,state=detach_state(z.state),optimizer=opt)
                restored,state,optimizer_state=load_checkpoint(Path(tmp)/"check.pt",device=device)
                with torch.no_grad():
                    row["checkpoint_next_window"]=difference(restored(y,state).readout(),candidate(y,detach_state(z.state)).readout())
            finite=all(v["finite"] for _,v in grads) and all(row[k]["finite"] for k in ("output","input_gradient","continuation","checkpoint_next_window"))
            tol=.002
            close=row["output"]["relative_l2"]<tol and row["input_gradient"]["relative_l2"]<5*tol
            row["state"]="within_threshold" if finite and close and not none else "finite_deviation" if finite else "nonfinite"
            row["output_relative_l2_threshold"]=tol
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed",error=repr(exc))
        rows.append(row)
        temporary=path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(dict(torch=torch.__version__,rows=rows),stream,indent=2)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        print(json.dumps(row),flush=True)
    return int(any(r["state"] in ("execution_failed","nonfinite") for r in rows))


if __name__=="__main__":
    sys.exit(main())
