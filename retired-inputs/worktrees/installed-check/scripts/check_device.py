"""Small real-device mechanism checks; numerical differences are explicit data."""
import argparse
import copy
from dataclasses import fields,is_dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tide import ModelConfig,TopologyConfig,TideModel,save_checkpoint,load_checkpoint,detach_state


THRESHOLDS = {
    "output_state_continuation": {"atol": 1e-6, "rtol": 2e-3},
    "input_gradient": {"atol": 1e-6, "rtol": 1e-2},
    "parameter_gradient": {"atol": 1e-6, "rtol": 2e-3},
    "checkpoint": "exact values, metadata, parameters and optimizer state on the same device",
}


class NonfiniteFailure(RuntimeError):
    pass


def difference(actual,reference,*,atol=1e-6,rtol=2e-3):
    metadata=actual.shape==reference.shape and actual.dtype==reference.dtype
    a,b=actual.detach().cpu().double(),reference.detach().cpu().double()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result=dict(metadata_matches=metadata,finite=finite,exact=False,within_threshold=False,
                max_abs=None,relative_l2=None,scaled_error_max=None)
    if not metadata or not finite:
        return result
    delta=(a-b).abs()
    scaled=float((delta/(atol+rtol*b.abs())).max()) if a.numel() else 0.0
    result.update(exact=bool(torch.equal(a,b)),within_threshold=scaled<=1,
                  max_abs=float(delta.max()) if a.numel() else 0.0,
                  relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-20)),scaled_error_max=scaled)
    return result


def state_comparison(actual,reference,root="state"):
    floats=[];metadata=[];support=[]
    def visit(a,b,path):
        if type(a) is not type(b):
            metadata.append((path+".type",False));return
        if isinstance(a,torch.Tensor):
            metadata.append((path+".shape",a.shape==b.shape))
            metadata.append((path+".dtype",a.dtype==b.dtype))
            if a.shape!=b.shape:return
            if a.is_floating_point():
                floats.append((path,difference(a,b)))
                if path.rsplit(".",1)[-1] in ("weight","strengths"):
                    support.append((path,torch.equal(a.detach().cpu()>0,b.detach().cpu()>0)))
            else:metadata.append((path,torch.equal(a.cpu(),b.cpu())))
        elif is_dataclass(a):
            for field in fields(a):visit(getattr(a,field.name),getattr(b,field.name),path+"."+field.name)
        elif isinstance(a,dict):
            metadata.append((path+".keys",a.keys()==b.keys()))
            for key in sorted(a.keys()&b.keys(),key=str):visit(a[key],b[key],path+"."+str(key))
        elif isinstance(a,(list,tuple)):
            metadata.append((path+".length",len(a)==len(b)))
            for index,(x,y) in enumerate(zip(a,b)):visit(x,y,path+f".{index}")
        else:metadata.append((path,a==b))
    visit(actual,reference,root)
    return dict(metadata_mismatches=[p for p,ok in metadata if not ok],
                support_mismatches=[p for p,ok in support if not ok],
                float_tensors=len(floats),finite=all(v["finite"] for _,v in floats),
                exact=all(ok for _,ok in metadata+support) and all(v["exact"] for _,v in floats),
                threshold_failures=[p for p,v in floats if not v["within_threshold"]],
                max_relative_l2=max((v["relative_l2"] or 0 for _,v in floats),default=0.0),
                max_scaled_error=max((v["scaled_error_max"] or 0 for _,v in floats),default=0.0),
                inexact_tensors=[p for p,v in floats if not v["exact"]],
                largest_absolute=sorted(floats,key=lambda p:p[1]["max_abs"] or 0,reverse=True)[:3])


def result_fields(result):
    return dict(values=result.values,strengths=result.strengths,times=result.times,readout=result.readout())


def nonfinite_names(named):
    present=[(name,value) for name,value in named if value is not None]
    if not present:return []
    finite=torch.stack([torch.isfinite(value.detach()).all() for _,value in present]).cpu().tolist()
    return [name for (name,_),ok in zip(present,finite) if not ok]


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
        ModelConfig(topology=TopologyConfig(n=8,layers=4),d=64,allocation="pool",
                    aggregation="incoming_light_128",sharing="region_shared"),
        ModelConfig(topology=TopologyConfig(kind="multiscale",n=8,profile="compact-5-dd4",hubs=4,
                    timing="timed",partition="mixed"),d=64,memory="K",sharing="region_shared"),
    ]
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    source=os.environ.get("TIDE_SOURCE_COMMIT") or subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    record=dict(source=source,TIDE_SOURCE_COMMIT=os.environ.get("TIDE_SOURCE_COMMIT"),torch=str(torch.__version__),
                torch_npu=str(torch_npu.__version__) if device.type=="npu" else None,
                allow_hf32={"matmul":bool(torch.npu.matmul.allow_hf32),"conv":bool(torch.npu.conv.allow_hf32)}
                if device.type=="npu" else None,thresholds=THRESHOLDS,
                exit_policy="finite_deviation continues; structural_failed/execution_failed/nonfinite fail")
    rows=[]
    for index,cfg in enumerate(configs):
        row=dict(config=cfg.to_dict(),device=args.device,precision="fp32",state="running")
        try:
            torch.manual_seed(514+index)
            cpu=TideModel(cfg)
            candidate=copy.deepcopy(cpu).to(device)
            x=torch.randn(2,5,64,requires_grad=True)
            y=x.detach().to(device).requires_grad_()
            reference=cpu(x)
            seed=torch.randn_like(reference.readout())
            (reference.readout()*seed).mean().backward()
            result=candidate(y)
            loss=(result.readout()*seed.to(device)).mean()
            loss.backward()
            row["output"]=difference(result.readout(),reference.readout())
            none=[]
            if y.grad is None or x.grad is None:
                none.append("inputs");row["input_gradient"]=None
            else:
                row["input_gradient"]=difference(y.grad,x.grad,**THRESHOLDS["input_gradient"])
            row["strength"]=difference(result.strengths,reference.strengths)
            row["state_comparison"]=state_comparison(result.state,reference.state)
            comparisons={"result":state_comparison(result_fields(result),result_fields(reference),"result"),
                         "state":row["state_comparison"]}
            row["comparisons"]=comparisons
            grads=[];metadata=[]
            actual_parameters=dict(candidate.named_parameters());reference_parameters=dict(cpu.named_parameters())
            if actual_parameters.keys()!=reference_parameters.keys():metadata.append("parameter names")
            for name in sorted(actual_parameters.keys()&reference_parameters.keys()):
                a,b=actual_parameters[name],reference_parameters[name]
                if a.grad is None or b.grad is None:
                    if (a.grad is None)!=(b.grad is None):
                        none.append(name)
                else:
                    delta=difference(a.grad,b.grad,**THRESHOLDS["parameter_gradient"])
                    grads.append((name,delta))
                    if not delta["metadata_matches"]:metadata.append("gradient."+name)
            row["disconnected_gradient_mismatches"]=none
            row["parameter_gradient_max_relative_l2"]=max((v["relative_l2"] or 0 for _,v in grads),default=0)
            row["parameter_gradient_threshold"]=THRESHOLDS["parameter_gradient"]
            row["parameter_gradient_threshold_failures"]=sorted(name for name,v in grads if not v["within_threshold"])
            row["largest_gradient_differences"]=sorted(grads,key=lambda item:item[1]["relative_l2"] or 0,reverse=True)[:5]
            with torch.no_grad():
                a=candidate(y[:,:2]); b=candidate(y[:,2:],a.state)
                row["continuation"]=difference(torch.cat((a.readout(),b.readout()),1),result.readout())
                continued=dict(values=torch.cat((a.values,b.values),2),strengths=torch.cat((a.strengths,b.strengths),2),
                               times=torch.cat((a.times,b.times)),readout=torch.cat((a.readout(),b.readout()),1))
                comparisons["continuation_result"]=state_comparison(continued,result_fields(result),"continuation")
                comparisons["continuation_state"]=state_comparison(b.state,result.state,"continuation.state")
            opt=torch.optim.AdamW(candidate.parameters(),lr=1e-4,foreach=True)
            losses=[];steps=[]
            row["three_step_losses"]=losses;row["optimizer_steps"]=steps
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                z=candidate(y.detach()); objective=z.readout().float().square().mean()
                if not bool(torch.isfinite(objective)):raise NonfiniteFailure(f"step {step}: loss")
                objective.backward()
                bad_gradients=nonfinite_names((name,p.grad) for name,p in candidate.named_parameters())
                observation=dict(step=step+1,nonfinite_gradients=bad_gradients,
                                 none_gradients=[name for name,p in candidate.named_parameters() if p.grad is None])
                steps.append(observation);losses.append(float(objective.detach()))
                if bad_gradients:raise NonfiniteFailure(f"step {step}: gradients")
                opt.step()
                observation["nonfinite_parameters"]=nonfinite_names(candidate.named_parameters())
                if observation["nonfinite_parameters"]:raise NonfiniteFailure(f"step {step}: parameters")
            with tempfile.TemporaryDirectory() as tmp:
                saved=detach_state(z.state)
                save_checkpoint(Path(tmp)/"check.pt",candidate,state=saved,optimizer=opt)
                restored,state,optimizer_state=load_checkpoint(Path(tmp)/"check.pt",device=device)
                with torch.no_grad():
                    resumed=restored(y.detach(),state);continued=candidate(y.detach(),saved)
                    row["checkpoint_next_window"]=difference(resumed.readout(),continued.readout())
                    checkpoint=state_comparison(
                        dict(parameters=restored.state_dict(),optimizer=optimizer_state,saved_state=state,
                             result=result_fields(resumed),final_state=resumed.state),
                        dict(parameters=candidate.state_dict(),optimizer=opt.state_dict(),saved_state=saved,
                             result=result_fields(continued),final_state=continued.state),"checkpoint")
                    row["checkpoint_exact"]=checkpoint
            differences=[v for _,v in grads]+([row["input_gradient"]] if row["input_gradient"] is not None else [])
            finite=checkpoint["finite"] and all(c["finite"] for c in comparisons.values()) and all(v["finite"] for v in differences)
            metadata += [name+":"+path for name,c in comparisons.items() for path in c["metadata_mismatches"]]
            support=[name+":"+path for name,c in comparisons.items() for path in c["support_mismatches"]]
            if row["input_gradient"] is not None and not row["input_gradient"]["metadata_matches"]:metadata.append("input_gradient")
            row["structural_failures"]=dict(metadata=metadata,support=support,gradient_none=none,
                                             checkpoint_not_exact=not checkpoint["exact"])
            structural=bool(metadata or support or none or not checkpoint["exact"])
            close=all(not c["threshold_failures"] for c in comparisons.values()) and all(v["within_threshold"] for v in differences)
            row["state"]=("nonfinite" if not finite else "structural_failed" if structural else
                          "within_threshold" if close else "finite_deviation")
        except NonfiniteFailure as exc:
            row.update(state="nonfinite",error=str(exc))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row.update(state="execution_failed",error=repr(exc))
        rows.append(row)
        temporary=path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(dict(**record,rows=rows),stream,indent=2,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        json.loads(path.read_text())
        print(json.dumps(row),flush=True)
    return int(any(r["state"] in ("structural_failed","execution_failed","nonfinite") for r in rows))


if __name__=="__main__":
    sys.exit(main())
