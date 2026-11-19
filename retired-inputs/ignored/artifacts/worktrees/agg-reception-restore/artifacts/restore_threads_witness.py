"""Bounded replay of the unchanged FP32 Timed Reception restore fixture."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tests"))
from tide import TideModel, detach_state, save_checkpoint, load_checkpoint
from test_reception_execution import config
from test_training_optimizations import update
from benchmarks.continuous import nested_comparison
from scripts.job import atomic
from scripts.qualify_precision import cpu_tree, fingerprint, tensors


def differences(actual, expected):
    left, right = dict(tensors(actual)), dict(tensors(expected))
    rows = []
    for name in sorted(left.keys() & right.keys()):
        a,b = left[name],right[name]
        if torch.equal(a,b):
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append(dict(field=name,metadata=True))
            continue
        locations = (a != b).reshape(-1).nonzero().flatten()
        maximum = int((a.double()-b.double()).abs().reshape(-1).argmax())
        rows.append(dict(field=name,shape=list(a.shape),changed=int(locations.numel()),
                         max_abs=float((a.double()-b.double()).abs().max()),
                         max_flat_index=maximum,actual=float(a.reshape(-1)[maximum]),
                         expected=float(b.reshape(-1)[maximum])))
    return rows


def observe_update(model,opt,state,x):
    following,grads,results = update(model,opt,state,x)
    return cpu_tree(dict(state=following,results=results,input_gradient=grads[0],
        gradients=dict(zip((name for name,_ in model.named_parameters()),grads[1:])),
        parameters=model.state_dict(),optimizer=opt.state_dict()))


def vjp(model,state,inputs):
    model.zero_grad(set_to_none=True)
    x=inputs.clone().requires_grad_()
    first=model(x[:,:2],state)
    second=model(x[:,2:],first.state)
    results=[first,second,model.flush(second.state)]
    values=torch.cat([r.readout() for r in results],1)
    probe=torch.sin(torch.arange(values.numel(),dtype=values.dtype)).reshape_as(values)
    (values*probe).mean().backward()
    return cpu_tree(dict(results=detach_state(results),input_gradient=x.grad,
                         gradients={name:p.grad for name,p in model.named_parameters()}))


directory=Path(sys.argv[1])
directory.mkdir(parents=True,exist_ok=False)
record=dict(source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            production_tree=subprocess.check_output(["git","rev-parse","HEAD:src/tide"],text=True).strip(),
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            torch=str(torch.__version__),cpu_affinity=sorted(os.sched_getaffinity(0)),
            seed=520,observations=[],scope="finite CPU multithread restore and same-owner repeated-VJP replay; original exact failures retained")
for run,threads in enumerate((4,4,1)):
    torch.set_num_threads(threads)
    torch.manual_seed(520)
    model=TideModel(config("incoming_light_128","timed",reception_qkv_backend="fused",reception_mask_padding=True))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    state=None
    for _ in range(3):
        state,_,_=update(model,opt,state,torch.randn(1,4,64),"fp32")
    path=directory/f"run-{run}-threads-{threads}.pt"
    save_checkpoint(path,model,state=state,optimizer=opt)
    restored,rs,slots=load_checkpoint(path)
    ropt=torch.optim.AdamW(restored.parameters());ropt.load_state_dict(slots)
    inputs=torch.randn(1,4,64)
    saved=nested_comparison(dict(parameters=model.state_dict(),optimizer=opt.state_dict(),state=state),
                           dict(parameters=restored.state_dict(),optimizer=ropt.state_dict(),state=rs),atol=0,rtol=0)
    # First run the original test's exact sequence before any repeat diagnosis.
    observations=[observe_update(owner,optimizer,old,inputs) for owner,optimizer,old in
                  ((model,opt,state),(restored,ropt,rs))]
    row=dict(run=run,threads=torch.get_num_threads(),input_sha256=fingerprint(inputs),saved=saved,
             restored_update_exact=nested_comparison(*observations,atol=0,rtol=0),
             restored_update_tolerance=nested_comparison(*observations),
             restored_update_differences=differences(*observations))
    owner,old,_=load_checkpoint(path)
    repeats=[vjp(owner,old,inputs) for _ in range(2)]
    row.update(same_owner_repeat_exact=nested_comparison(*repeats,atol=0,rtol=0),
               same_owner_repeat_tolerance=nested_comparison(*repeats),
               same_owner_repeat_differences=differences(*repeats))
    record["observations"].append(row)
    atomic(directory/"report.json",record)
    print(json.dumps(dict(run=run,threads=threads,saved_exact=saved["exact"],
          restored_exact=row["restored_update_exact"]["exact"],
          restored_max_abs=row["restored_update_tolerance"]["max_abs"],
          restored_tolerant=row["restored_update_tolerance"]["within_threshold"],
          restored_fields=len(row["restored_update_differences"]),
          repeat_exact=row["same_owner_repeat_exact"]["exact"],
          repeat_fields=len(row["same_owner_repeat_differences"]))),flush=True)
