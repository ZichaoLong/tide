#!/usr/bin/env python3
"""Bounded same-input FP64 audit of the GDN/HB evidence-root discrepancy.

This records distances; finite values alone are not proof of harmless rounding.
Run outside performance timing, with the site NPU admission if requested.
"""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import sys
import torch
from durable_records import write_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from test_component_window_weighted import fixture
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.reference import run as reference
from tidegraph.weighted.execution import eager_runner
from tidegraph.runtime import resolve_device
from isolated_cases import vjp
from test_weighted_execution import variables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    device, _ = resolve_device(a.device)
    b, xs, stop = fixture(torch.float32, "gated_delta", "hb", "cpu")
    result = {}
    for name, dtype, target, kind in (("reference64", torch.float64, "cpu", "reference"),
                                      ("reference32", torch.float32, "cpu", "reference"),
                                      ("candidate32", torch.float32, device, "python")):
        model = copy.deepcopy(b.model).to(device=target, dtype=dtype)
        binding = replace(b, model=model)
        inputs = [replace(x, value=x.value.detach().to(device=target, dtype=dtype).requires_grad_()) for x in xs]
        q = Continuation(b.graph.identity, 2)
        if kind == "reference":
            value = reference(b.graph, model, q, inputs, stop, sealed_until=stop)
        else:
            place = ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c")
            value = eager_runner(binding, "python", place)(q, inputs, stop, sealed_until=stop)
        evidence = value.trace[0]["descriptor"]["evidence"]
        gradients = vjp(evidence, variables(binding, inputs))
        result[name] = dict(evidence=evidence.detach().cpu().double().tolist(),
            gradients={k:g.detach().cpu().double() for k,g in gradients.items() if g is not None
                       and ("evidence_weight" in k or "evidence_bias" in k or k.startswith("input."))})
    truth = result["reference64"]["gradients"]
    for data in result.values():
        data["max_abs_vs_fp64"] = {k:float((g-truth[k]).abs().max()) for k,g in data["gradients"].items()}
        data["gradients"] = {k:g.tolist() for k,g in data["gradients"].items()}
    write_json(a.output, dict(witness="0.7*first evidence.square().sum; original FP32 inputs and parameters exactly promoted",
                             device=str(device), observations=result,
                             scope="Local root audit only; no tolerance change, learning or all-scale equivalence claim."))


if __name__ == "__main__":
    main()
