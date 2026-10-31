#!/usr/bin/env python3
"""Bounded two-device eager VJP isolation, with explicit diagnostic-only copies.

Stock copies deliberately do not certify None/zero VJPs. They isolate whether
the runtime error requires our custom transfer functions; no production path
or support claim is silently changed by this diagnostic.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext
import torch
from durable_records import write_json
from source_identity import source_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",required=True)
    p.add_argument("--peer-device",required=True)
    p.add_argument("--copy-policy",choices=("public","stock","cpu_bridge"),default="public")
    p.add_argument("--scope", choices=("first-vjp", "complete"), default="first-vjp")
    p.add_argument("--case",default="delta-pdg")
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.scope == "complete" and a.copy_policy != "cpu_bridge":
        p.error("complete alternative assessment requires explicit cpu_bridge")
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/"python"))
    from tidegraph.runtime import resolve_device, synchronize, manifest
    from tidegraph.placement import ExecutionPlacement, place_model
    from tidegraph import Continuation
    from tidegraph.reference import run as reference
    from tidegraph.compare import objective, equivalent
    from tidegraph.qualification_checks import vjp
    from weighted_device_cases import CASES, fixture, leaves
    from weighted_device_checks import locate, runner, check_case
    from weighted_peer_bridge import cpu_bridge
    device,reason=resolve_device(a.device);peer,_=resolve_device(a.peer_device)
    from cuda_precision import ieee_policy
    cuda_policy=ieee_policy(device)
    if peer==device or peer.type!=device.type or device.type=="cpu":p.error("two distinct accelerators required")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    case=next(c for c in CASES if c.name==a.case)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    source,dirty=source_state(root)
    report=dict(schema="tide-weighted-async-diagnostic-v1",state="running",source=source,dirty=dirty,
        argv=sys.argv[1:],runtime=manifest(device,reason,torch.float32),cuda_precision=cuda_policy,peer=str(peer),copy_policy=a.copy_policy,
        launch_blocking=os.environ.get("ASCEND_LAUNCH_BLOCKING"),phases=[],
        scope="first complete forward and all-root VJP only; stock copies do not certify independent None/zero roots")
    def phase(name):
        report["phases"].append(name);write_json(out/"result.json",report);print(name,flush=True)
    try:
        if a.scope == "complete":
            report["scope"] = "full graph, independent roots, retained state, four AdamW updates, exact decode and remapped-peer/CPU restore"
            phase("before_complete_bridge")
            with cpu_bridge() as counts:
                report["bridge_counts"] = counts
                report["checks"] = check_case(case, device, "python", ExecutionPlacement(preset="mixed-c"),
                                             out, peers=(device, peer))
            for d in (device, peer):synchronize(d)
            report["state"] = "passed"
            phase("complete_bridge_passed")
            return
        left,xl,stop=fixture(case);right,_,_=fixture(case)
        xr,_=locate(right,xl,(device,peer))
        q=Continuation(left.graph.identity,2)
        expected=reference(left.graph,left.model,q,xl,stop,sealed_until=stop)
        if a.copy_policy=="stock":
            import tidegraph.transfer as transport
            import tidegraph.weighted.placement_tensors as placed
            transport.copy_rows=lambda rows,destination,budget=transport.PACK_BYTES:[x.to(destination) for x in rows]
            def stock_mapper(values,target,dtype):
                copied={id(x):x.to(device=target,dtype=dtype if x.is_floating_point() else x.dtype) for x in values}
                return lambda x:copied[id(x)]
            placed.mapper=stock_mapper
            # The placement adapter imports mapper directly.
            import tidegraph.weighted.placement as placement_module
            placement_module.mapper=stock_mapper
            placed.move=lambda x,target,dtype=None:None if x is None else x.to(device=target,dtype=dtype if x.is_floating_point() and dtype else x.dtype)
        phase("before_candidate")
        with cpu_bridge() if a.copy_policy == "cpu_bridge" else nullcontext():
            actual=runner(right,"python",ExecutionPlacement(preset="mixed-c"))(q,xr,stop,sealed_until=stop)
            equivalent(expected,actual,check_device=False)
            phase("complete_forward_passed")
            gl=vjp(objective(expected),leaves(left,xl))
            phase("before_all_root_vjp")
            gr=vjp(objective(actual),leaves(right,xr))
            phase("all_root_vjp_returned")
            equivalent(gl,gr,check_device=False)
        for d in (device,peer):synchronize(d)
        report["state"]="passed"
    except BaseException as error:
        report.update(state="failed",error=repr(error));raise
    finally:write_json(out/"result.json",report)
    print(json.dumps(dict(state=report["state"],scope=report["scope"])))


if __name__=="__main__":main()
