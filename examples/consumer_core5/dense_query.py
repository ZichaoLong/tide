#!/usr/bin/env python3
"""Installed query consumer: full training and fresh-process exact recovery.

Finite numerical disagreement is reported, never called strict qualification.
Discrete/structure/nonfinite/exact-recovery failures stop this check.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.checkpoint import save, load
from tidegraph.comparison import collect_comparison, equivalent
from tidegraph.compare import objective
from tidegraph.experiment.returns import snapshot
from tidegraph.runtime import resolve_device
from tidegraph.weighted import fully_connected, bind_layered, layered_inputs, KVConfig, RegionalConfig
from tidegraph.weighted.execution import eager_runner


def setup(args, *, reference=False):
    device = "cpu" if reference else args.device
    spec, b = bind_layered(fully_connected(2, 2), memory=KVConfig(width=4, summary_width=2,
        adapter_rank=2, heads=2, ffn_width=8, memory_read="qry"),
        regions=RegionalConfig(k=1, gamma_max=1.5), dtype=getattr(torch, args.dtype), seed=11)
    b.model.to(device)
    x = torch.linspace(-.4, .5, 24, dtype=getattr(torch, args.dtype)).reshape(2, 3, 4).to(device).requires_grad_()
    if reference:
        from tidegraph.reference import run
        execute = lambda *a, **kw: run(b.graph, b.model, *a, trace=not args.no_trace, **kw)
    else:
        execute = eager_runner(b, args.client,
            ExecutionPlacement(preset="cpu" if device == "cpu" else "mixed-c", scoring_dtype=args.dtype),
            native_library=args.native_library, packing="packed", query_execution=args.query_execution, trace=not args.no_trace)
    opt = torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad], lr=.001, foreach=False)
    return b, spec, x, 3*spec.stride, execute, opt


def step(state):
    b, spec, x, stop, execute, opt = state
    opt.zero_grad(set_to_none=True); x.grad = None
    inputs = layered_inputs(spec, x)
    result = execute(Continuation(b.graph.identity, 2), inputs, stop, sealed_until=stop)
    objective(result).backward()
    gradients = snapshot(dict(parameters={n: p.grad for n, p in b.model.named_parameters()}, input=x.grad))
    opt.step()
    return snapshot(dict(result=result, gradients=gradients, parameters=b.model.state_dict(), optimizer=opt.state_dict()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--client", choices=("python", "native"), default="python")
    p.add_argument("--native-library")
    p.add_argument("--query-execution", choices=("auto", "legacy", "dense-v1", "packed-v2"), default="auto")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--resume", type=Path)
    p.add_argument("--child", action="store_true")
    p.add_argument("--no-trace", action="store_true", help="retain all output/boundary/parameter roots, omit internal event handles")
    args = p.parse_args()
    resolve_device(args.device)  # Before first CPU or device autograd use.
    torch.set_num_threads(1 if args.device == "cpu" else 2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(11)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    identity = dict(query_execution=args.query_execution, device=args.device, dtype=args.dtype, client=args.client)
    if args.no_trace:
        identity["trace"] = False
    record = dict(identity=identity, execution="running", strict_numerical="not_checked", recovery="not_checked", comparisons=[])
    try:
        state = setup(args); b, _, _, _, _, opt = state
        if args.resume:
            meta = json.loads(args.resume.with_suffix(".json").read_text())
            if meta["identity"] != identity:
                raise ValueError("exact recovery requires identical execution selection")
            load(args.resume, b.graph, b.model, opt)
        oracle = None if args.child else setup(args, reference=True)
        for i in range(args.steps):
            actual = step(state)
            if oracle is not None:
                expected = step(oracle)
                comparison = collect_comparison(actual, expected, path=f"step{i}")
                if comparison["discrete"] != "passed":
                    raise AssertionError(comparison)
                record["comparisons"].append(comparison)
        torch.save(actual, out/"observation.pt")
        save(out/"checkpoint.pt", b.graph, b.model, Continuation(b.graph.identity, 2), opt)
        (out/"checkpoint.json").write_text(json.dumps(dict(identity=identity)))
        if oracle is not None:
            record["strict_numerical"] = "passed" if all(r["strict_passed"] for r in record["comparisons"]) else "failed"
            base = [sys.executable, str(Path(__file__).resolve()), "--child", "--device", args.device,
                    "--dtype", args.dtype, "--client", args.client, "--query-execution", args.query_execution]
            if args.native_library:
                base += ["--native-library", args.native_library]
            if args.no_trace:
                base += ["--no-trace"]
            for name, extra in (("whole", ["--steps", "3"]), ("prefix", ["--steps", "2"]),
                                ("resumed", ["--steps", "1", "--resume", str(out/"prefix/checkpoint.pt")])):
                with (out/(name+".log")).open("w") as log:
                    subprocess.run([*base, *extra, "--output-dir", str(out/name)], check=True,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=240)
            left = torch.load(out/"whole/observation.pt", weights_only=False)
            right = torch.load(out/"resumed/observation.pt", weights_only=False)
            equivalent(left, right, atol=0, rtol=0)
            record["recovery"] = "passed"
        record["execution"] = "completed"
    except BaseException as error:
        record.update(execution="failed", error=repr(error)); raise
    finally:
        # Only this short result is published; opaque tensor files are local artifacts.
        from recording import json_record
        json_record(out/"result.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "comparisons"}))


if __name__ == "__main__":
    main()
