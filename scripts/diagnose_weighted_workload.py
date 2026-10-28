#!/usr/bin/env python3
"""Separate discrete/float/gradient observations for a fixed static workload.

An explicit FP64 promotion is a distinct diagnostic, never a repair or passing
replacement for the original FP32 measurement. No reference guides a candidate.
"""
import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
import os
from pathlib import Path
import sys
import torch
from durable_records import write_json
from source_identity import source_state


def metadata(value):
    if isinstance(value, torch.Tensor):
        return ("floating", tuple(value.shape), str(value.dtype)) if value.is_floating_point() else value
    if is_dataclass(value):
        return {f.name: metadata(getattr(value, f.name)) for f in fields(value) if f.name != "stats"}
    if isinstance(value, dict):
        return {k: metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(metadata(v) for v in value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--implementation", choices=("python", "native"), required=True)
    p.add_argument("--native-library")
    p.add_argument("--steps", type=int, choices=(1, 2), default=2)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--capture-allocations", action="store_true", help="Python only: actual scalar/batch operands and branch/free masks, outside performance timing")
    args = p.parse_args()
    if args.capture_allocations and args.implementation != "python":
        p.error("allocation observation currently requires the independent Python schedules")
    if args.implementation == "native" and not args.native_library:
        p.error("native needs its explicit matching library")
    if args.native_library:
        args.native_library = str(Path(args.native_library).resolve())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"python"))
    from tidegraph import Continuation
    from tidegraph.compare import equivalent
    from tidegraph.reference import run as scalar
    from tidegraph.placement import ExecutionPlacement
    from weighted_plan_materialize import load, materialize
    from weighted_benchmark_cases import window
    from weighted_benchmark_runners import loss_from_outputs, work
    from weighted_device_checks import runner
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan, config = load(args.workload.resolve())
    source, dirty = source_state(root)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    report = dict(schema="tide-weighted-workload-diagnostic-v1", source=source, dirty=dirty,
        workload_sha256=plan["sha256"], implementation=args.implementation, dtype=args.dtype,
        input_policy="same seeded FP32 material; explicit promotion before both schedules",
        scope="CPU complete records, discrete structure, output-loss gradients, SGD updates; not throughput",
        tolerances="unchanged public compare: FP32 atol1e-6/rtol1e-5; FP64 atol1e-10/rtol1e-8",
        state="running", steps=[])
    write_json(out/"result.json", report)

    def check(a, b):
        try:
            equivalent(a, b)
        except AssertionError as error:
            return str(error)
        return None

    from weighted_allocation_observer import observe_allocations
    observer = observe_allocations() if args.capture_allocations else nullcontext(None)
    allocation_records = observer.__enter__()
    try:
        bindings, data = [], []
        for _ in range(2):
            bound, xs, span = materialize(plan, max_bytes=1024**3)
            bound.model.to(dtype=getattr(torch, args.dtype))
            copies = {id(x.value): x.value.to(dtype=getattr(torch, args.dtype)) for x in xs}
            bindings.append(bound)
            data.append([replace(x, value=copies[id(x.value)]) for x in xs])
        a, b = bindings
        execute = runner(b, args.implementation, ExecutionPlacement(), args.native_library)
        states = [Continuation(a.graph.identity, config.samples), Continuation(b.graph.identity, config.samples)]
        optimizers = [torch.optim.SGD(x.model.parameters(), lr=.001, momentum=.3, weight_decay=.01) for x in bindings]
        for step in range(args.steps):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            xs = [window(x, span, config.positions, step, "cpu") for x in data]
            left = scalar(a.graph, a.model, states[0], xs[0], (step+1)*span, sealed_until=(step+1)*span)
            right = execute(states[1], xs[1], (step+1)*span, sealed_until=(step+1)*span)
            row = dict(step=step, discrete=check(metadata(left), metadata(right)),
                       complete_records=check(left, right), reference_work=work(left), candidate_work=work(right))
            report["steps"].append(row)
            write_json(out/"result.json", report)
            losses = [loss_from_outputs(r) for r in (left, right)]
            for loss in losses:
                loss.backward()
            gradients = [{n: v.grad for n, v in x.model.named_parameters()} for x in bindings]
            row.update(losses=[float(x.detach()) for x in losses], gradients=check(*gradients))
            for optimizer in optimizers:
                optimizer.step()
            row["parameters"] = check(*(dict(x.model.named_parameters()) for x in bindings))
            states = [r.continuation.detach() for r in (left, right)]
            write_json(out/"result.json", report)
        report["state"] = "strict_mismatch" if any(
            row[k] is not None for row in report["steps"]
            for k in ("discrete", "complete_records", "gradients", "parameters")) else "passed"
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        observer.__exit__(None, None, None)
        if allocation_records is not None:
            write_json(out/"allocations.json", allocation_records)
            report["allocation_observations"] = "allocations.json"
        write_json(out/"result.json", report)
    raise SystemExit(0 if report["state"] == "passed" else 2)


if __name__ == "__main__":
    main()
