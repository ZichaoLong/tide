#!/usr/bin/env python3
"""Bounded same-FP32-material layer audit against CPU FP64 scalar formulas.

This projection audit retains forward/state and output-loss gradients; complete
None/zero and intermediate-root qualification stays in the graph tests. It does
not modify a numerical tolerance or infer learning quality from finite errors.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import torch
from durable_records import write_json
from layer_tensor_baseline import fixture, external, run as tensor_run, numerics
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.comparison import collect_comparison
from tidegraph.reference import run as scalar_run
from tidegraph.runtime import resolve_device
from tidegraph.weighted.execution import eager_runner


def observe(base, spec, original, kind, device, dtype, library):
    binding = replace(base, model=deepcopy(base.model).to(device=device, dtype=dtype))
    x = original.to(device=device, dtype=dtype).detach().requires_grad_()
    if kind == "tensor":
        output, states = tensor_run(binding, spec, x)
    else:
        q = Continuation(binding.graph.identity, len(x))
        xs, stop = external(spec, x), x.shape[1]*spec.stride
        if kind == "scalar":
            result = scalar_run(binding.graph, binding.model, q, xs, stop, sealed_until=stop)
        else:
            execute = eager_runner(binding, kind,
                ExecutionPlacement(preset="cpu" if str(device) == "cpu" else "mixed-c",
                                   scoring_dtype=str(dtype).split(".")[-1]),
                native_library=library, trace=False)
            result = execute(q, xs, stop, sealed_until=stop)
        output = torch.stack([p[1:] for *_, p in result.outputs]).reshape(x.shape[1], len(x), -1).transpose(0, 1)
        states = []
        for v in range(len(binding.graph.nodes)):
            rows = [result.continuation.states[b, v].slots for b in range(len(x))]
            states.append({name: rows[0][name] if name in {"time", "times", "conv_times"}
                           else torch.stack([s[name] for s in rows]) for name in rows[0]})
    roots = dict(input=x, **{name: p for name, p in binding.model.named_parameters() if p.requires_grad})
    bars = torch.autograd.grad(output.square().sum(), list(roots.values()), allow_unused=True)
    gradient = {name: (torch.zeros_like(root) if bar is None else bar).detach().cpu().double()
                for (name, root), bar in zip(roots.items(), bars)}
    values = dict(output=output.detach().cpu())
    for v, fields in enumerate(states):
        values.update({f"state.{v}.{name}": t.detach().cpu() for name, t in fields.items()})
    for t in [*values.values(), *gradient.values()]:
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise FloatingPointError("nonfinite layer diagnostic result")
    return values, gradient


def distances(values, truth):
    result = {}
    for name, actual in values.items():
        expected = truth[name]
        if actual.is_floating_point():
            delta = (actual.double()-expected.double()).abs()
            result[name] = dict(max_abs=float(delta.max()) if delta.numel() else 0.,
                                reference_max_abs=float(expected.abs().max()) if expected.numel() else 0.)
        elif not torch.equal(actual, expected):
            raise AssertionError("integer layer metadata differs: "+name)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--native-library")
    p.add_argument("--precision-policy", choices=("ieee", "vendor-default"), default="ieee")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    device, _ = resolve_device(a.device)
    numerical_policy = numerics(device, a.precision_policy)
    report = dict(source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        device=str(device), numerical_policy=numerical_policy,
        material="CPU FP32 width4/depth2/heads2, B2/L3 linspace(-.2,.4), exactly promoted",
        scope="output and terminal state; output-square numerical derivatives; no graph connectivity or learning claim",
        families={})
    for family in ("kv", "ssd"):
        base, spec = fixture(family, width=4, depth=2, heads=2)
        original = torch.linspace(-.2, .4, 24).reshape(2, 3, 4)
        truth, truth_grad = observe(base, spec, original, "scalar", "cpu", torch.float64, a.native_library)
        cases = [("scalar32-cpu", "scalar", "cpu"), ("tensor32-cpu", "tensor", "cpu"),
                 ("tensor32-device", "tensor", device), ("python32-device", "python", device)]
        if a.native_library:
            cases.append(("native32-device", "native", device))
        rows = {}
        for name, kind, target in cases:
            values, gradient = observe(base, spec, original, kind, target, torch.float32, a.native_library)
            rounded = {n: t.float() if t.is_floating_point() else t for n, t in truth.items()}
            rows[name] = dict(values_vs_fp64=distances(values, truth), gradients_vs_fp64=distances(gradient, truth_grad),
                strict_values_vs_rounded_fp64=collect_comparison(values, rounded),
                strict_gradients_vs_rounded_fp64=collect_comparison({n:t.float() for n,t in gradient.items()},
                                                                  {n:t.float() for n,t in truth_grad.items()}))
        report["families"][family] = rows
    write_json(a.output, report)
    print({f: {n: dict(output=r["values_vs_fp64"]["output"]["max_abs"],
                      gradient=max(v["max_abs"] for v in r["gradients_vs_fp64"].values()))
               for n,r in rows.items()} for f,rows in report["families"].items()})


if __name__ == "__main__":
    main()
