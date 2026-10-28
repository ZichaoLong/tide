#!/usr/bin/env python3
"""Locate the retained eager FP16 GDN/AdamW witness without changing its policy.

Uses the exact qualification fixture; the candidate still owns its schedule.
Reports first strict parameter divergence and the incoming gradients at that
coordinate. A collected diagnostic is not a passing equivalence certificate.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def mismatch(a, b):
    from tidegraph.compare import equivalent
    try:
        equivalent(a,b)
    except AssertionError as error:
        return str(error)
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--implementation", choices=("python","native"), required=True)
    p.add_argument("--native-library")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/"python"),str(root/"tests")]
    from tidegraph import Continuation
    from tidegraph.compare import objective
    from tidegraph.reference import run as reference
    from tidegraph.placement import ExecutionPlacement, place_model
    from tidegraph.precision import FP32MasterOptimizer
    from test_weighted_precision import fixture, run, half_close
    if a.implementation=="native":
        if not a.native_library: p.error("native implementation requires --native-library")
        from tidegraph.native_loader import load_native
        load_native(a.native_library,backend="cpu")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source,dirty=source_state(root)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema="tide-weighted-numerical-diagnostic-v1",source=source,dirty=dirty,
        implementation=a.implementation,fixture="tests/test_weighted_precision.py::fixture(gated_delta)",
        precision="eager half payload; explicit FP32 scoring; FP32 masters; AdamW lr=.001, loss_scale=16",
        tolerance="unchanged public half tolerance; exact discrete/None",state="running",steps=[])
    write_json(out/"result.json",report)
    left,xl,stop=fixture("gated_delta");right,xr,_=fixture("gated_delta")
    view=place_model(left.graph,left.model,ExecutionPlacement(scoring_dtype="float32"))
    execute=run(right,a.implementation=="native")
    q=Continuation(left.graph.identity,1)
    parameters=[dict(b.model.named_parameters()) for b in (left,right)]
    optimizers=[FP32MasterOptimizer((x for x in b.model.parameters() if x.requires_grad),optimizer="adamw",lr=.001,loss_scale=16)
                for b in (left,right)]
    torch.save(dict(weights=left.model.state_dict(),inputs=xl),out/"fixture.pt")
    report["fixture_sha256"]=hashlib.sha256((out/"fixture.pt").read_bytes()).hexdigest()
    try:
        for step in range(3):
            for opt in optimizers: opt.zero_grad()
            rl=reference(left.graph,view,q,xl,stop,sealed_until=stop)
            rr=execute(q,xr,stop,sealed_until=stop)
            half_close(rl,rr)
            losses=[objective(r) for r in (rl,rr)]
            for opt,loss in zip(optimizers,losses): opt.backward(loss)
            grads=[{n:None if x.grad is None else x.grad.detach().float()/16 for n,x in ps.items()} for ps in parameters]
            for opt in optimizers: opt.step()
            row=dict(step=step,complete_forward="passed",losses=[float(x.detach()) for x in losses],parameter_witnesses=[])
            watch="nodes.5.kernel.conv.weight"
            row["watched_parameter"]=dict(name=watch,unscaled_gradients=[g[watch].tolist() for g in grads],
                                          updated_values=[ps[watch].detach().float().tolist() for ps in parameters])
            for name,x in parameters[0].items():
                y=parameters[1][name]
                detail=mismatch(x,y)
                if not detail: continue
                mask=~torch.isclose(x,y,atol=1e-3,rtol=2e-2)
                index=tuple(int(i) for i in mask.nonzero()[0])
                gx,gy=(g[name] for g in grads)
                row["parameter_witnesses"].append(dict(name=name,index=list(index),reference=float(x[index].detach()),
                    candidate=float(y[index].detach()),max_abs=float((x-y).abs().max().detach()),
                    reference_gradient=None if gx is None else float(gx[index]),
                    candidate_gradient=None if gy is None else float(gy[index]),strict_error=detail))
            report["steps"].append(row);write_json(out/"result.json",report)
            if row["parameter_witnesses"]:
                torch.save(dict(reference=left.model.state_dict(),candidate=right.model.state_dict(),
                                reference_gradients=grads[0],candidate_gradients=grads[1],
                                reference_optimizer=optimizers[0].state_dict(),candidate_optimizer=optimizers[1].state_dict()),out/"witness.pt")
                report["witness_sha256"]=hashlib.sha256((out/"witness.pt").read_bytes()).hexdigest()
                report["state"]="strict_mismatch";break
        else: report["state"]="passed"
    except BaseException as error:
        report.update(state="failed",error=repr(error));raise
    finally: write_json(out/"result.json",report)
    print(report["state"])
    return 2 if report["state"]=="strict_mismatch" else 0


if __name__=="__main__":
    sys.exit(main())
