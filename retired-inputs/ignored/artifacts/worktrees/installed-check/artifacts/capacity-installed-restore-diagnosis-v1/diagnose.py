"""One bounded checkpoint witness: owners, repeated backward, serial scheduling."""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import torch
from tide import detach_state,load_checkpoint,placement_adamw,serial_region_audit
from tide.precision import precision_context


def scalar_update(p,g,m,v,step,*,lr,betas,eps,weight_decay):
    b1,b2=betas
    m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g;step+=1
    return p*(1-lr*weight_decay)-lr/(1-b1**step)*m/(math.sqrt(v)/math.sqrt(1-b2**step)+eps)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise RuntimeError("new output required")
    spec=importlib.util.spec_from_file_location("installed_harness",Path(__file__).with_name("harness.py"))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.torch=torch;h.precision_context=precision_context;h.placement_adamw=placement_adamw;h.detach_state=detach_state
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=args.input;case=job/"results"/"placed-bf16"
    expected=torch.load(case/"expected.pt",map_location="cpu",weights_only=False)
    installed=argparse.Namespace(wheel=str(job/"dist/tide_instances-0.1.0-py3-none-any.whl"),
        venv=str(job/"env"),source_manifest=str(job/"source-manifest.json"))
    result=dict(state="running",pid=os.getpid(),scope=__doc__,checkpoint_sha256=h.sha(case/"checkpoint.pt"),
        expected_sha256=h.sha(case/"expected.pt"),harness_sha256=h.sha(Path(__file__).with_name("harness.py")),
        diagnostic_sha256=h.sha(__file__),argv=sys.argv,threshold=dict(atol=1e-6,rtol=2e-3),
        installation=h.installation(installed),backwards=[],comparisons={})
    subjects={};snapshots={};observers={}
    for label in ("A","B"):
        model,state,slots=load_checkpoint(case/"checkpoint.pt",placement_devices=["cpu:0","cpu:1"])
        optimizer=placement_adamw(model,lr=1e-4,foreach=False);optimizer.load_state_dict(slots)
        subjects[label]=(model,state,optimizer);observers[label]=h.DtypeObserver(model)
        result[label+"_saved_data"]=h.compare(h.snapshot(model,optimizer,state),expected["saved"])
    h.atomic(args.output,result)
    schedule=(("A_per_device1","A",False),("B_per_device1","B",False),
              ("A_per_device2","A",False),("B_per_device2","B",False),
              ("A_serial1","A",True),("A_serial2","A",True))
    for label,owner,serial in schedule:
        model,state,optimizer=subjects[owner];optimizer.zero_grad(set_to_none=True)
        before=h.clone(model.state_dict());observers[owner].phase=label
        x=expected["input"].clone().requires_grad_()
        context=serial_region_audit(model) if serial else nullcontext(dict(
            requested_region_concurrency="per_device",effective_region_concurrency="per_device",workers=2))
        # Keep the explicit serial context across recomputation in backward.
        with context as execution:
            with precision_context("cpu","bf16"):values,loss=h.forward(model,x,state,split=True)
            loss.backward()
        gradients=h.gradients(model)
        snap=h.clone(dict(results=values,loss=loss,input_gradient=x.grad,gradients=gradients))
        snapshots[label]=snap
        bad=h.finite(snap)
        if bad:raise RuntimeError("nonfinite diagnostic: "+repr(bad))
        row=dict(label=label,owner=owner,execution=execution,loss=float(loss),nonfinite=bad,
                 none_gradients=sorted(name for name,value in gradients.items() if value is None),
                 unchanged_parameters=h.compare(model.state_dict(),before),
                 forward_vs_original=h.compare(snap["results"],expected["next_forward"]),
                 gradients_vs_original=h.compare(gradients,expected["gradients"],atol=1e-6,rtol=2e-3))
        result["backwards"].append(row);h.atomic(args.output,result)
    for key,a,b in (("same_A_per_device","A_per_device2","A_per_device1"),
                    ("same_B_per_device","B_per_device2","B_per_device1"),
                    ("same_A_serial","A_serial2","A_serial1"),
                    ("cross_owner_first","B_per_device1","A_per_device1"),
                    ("cross_owner_second","B_per_device2","A_per_device2"),
                    ("same_A_serial_vs_per_device","A_serial2","A_per_device2")):
        aa,bb=snapshots[a],snapshots[b]
        result["comparisons"][key]=dict(
            forward=h.compare(aa["results"],bb["results"]),
            gradients=h.compare(aa["gradients"],bb["gradients"],atol=1e-6,rtol=2e-3),
            input_gradient=h.compare(aa["input_gradient"],bb["input_gradient"],atol=1e-6,rtol=1e-2))
    modelA,stateA,optA=subjects["A"];modelB,stateB,optB=subjects["B"]
    before=h.clone(modelA.state_dict())
    moments={name:h.clone(optA.state[p]) for name,p in modelA.named_parameters() if p in optA.state}
    hyper={key:optA.param_groups[0][key] for key in ("lr","betas","eps","weight_decay")}
    optA.step();optB.step()
    actual={}
    for owner,snapshot_name in (("A","A_serial2"),("B","B_per_device2")):
        model,state,opt=subjects[owner]
        actual[owner]=h.snapshot(model,opt,detach_state(snapshots[snapshot_name]["results"][-1].state))
        result[owner+"_updated_vs_original"]=h.compare(actual[owner],expected["updated"],atol=1e-6,rtol=2e-3)
    result["actual_updated_comparison"]=h.compare(actual["B"],actual["A"],atol=1e-6,rtol=2e-3)
    candidates=[]
    for name in moments:
        ga,gb=snapshots["A_serial2"]["gradients"][name],snapshots["B_per_device2"]["gradients"][name]
        if ga is None or gb is None:continue
        a,b=actual["A"]["parameters"][name],actual["B"]["parameters"][name]
        error=(a.double()-b.double()).abs().flatten()
        scaled=error/(1e-6+2e-3*a.double().abs().flatten());index=int(scaled.argmax())
        if error[index]:candidates.append((float(scaled[index]),name,index))
    result["scalar_adamw_witnesses"]=[]
    for scaled,name,index in sorted(candidates,reverse=True)[:4]:
        old=float(before[name].flatten()[index]);slot=moments[name]
        m,v,step=float(slot["exp_avg"].flatten()[index]),float(slot["exp_avg_sq"].flatten()[index]),float(slot["step"])
        ga,gb=(float(snapshots[s]["gradients"][name].flatten()[index]) for s in ("A_serial2","B_per_device2"))
        pa,pb=scalar_update(old,ga,m,v,step,**hyper),scalar_update(old,gb,m,v,step,**hyper)
        aa,bb=(float(actual[o]["parameters"][name].flatten()[index]) for o in ("A","B"))
        result["scalar_adamw_witnesses"].append(dict(parameter=name,index=index,scaled_update_error=scaled,
            initial_parameter=old,first_moment=m,second_moment=v,step=step,gradient_A=ga,gradient_B=gb,
            predicted_A=pa,predicted_B=pb,actual_A=aa,actual_B=bb,
            predicted_difference=pb-pa,actual_difference=bb-aa))
    result["optimizer_hyperparameters"]=hyper
    result["observed_dtypes"]={owner:observer.report() for owner,observer in observers.items()}
    for observer in observers.values():observer.close()
    torch.save(dict(snapshots=snapshots,updated=actual),args.output.with_suffix(".pt"))
    result["tensor_witness_sha256"]=h.sha(args.output.with_suffix(".pt"))
    result["final_installation"]=h.installation(installed)
    structural=any(not result[o+"_saved_data"]["exact"] for o in ("A","B")) or any(
        not row["unchanged_parameters"]["exact"] or row["forward_vs_original"]["metadata_failures"] or
        row["none_gradients"]!=expected["none_gradients"] for row in result["backwards"])
    result["state"]="structural_failed" if structural else "completed_with_separate_numerical_checks"
    h.atomic(args.output,result);print(json.dumps(dict(state=result["state"],output=str(args.output))))
    return int(structural)


if __name__=="__main__":raise SystemExit(main())
