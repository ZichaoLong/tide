#!/usr/bin/env python3
"""Finite same-material audit of complete-root transpose and packed Full VJP."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import sys
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "npu:0"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    os.environ["TIDE_DENSE_DEVICE"] = a.device
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
    import torch
    from test_weighted_dense_query import independent
    from tidegraph.weighted.query_vjp import evaluate
    from tidegraph.weighted import query_roots
    from tidegraph.full import FullInput
    from tidegraph.comparison import collect_comparison
    from isolated_cases import vjp
    from durable_records import write_json
    torch.set_num_threads(1)
    w, requests, roots = independent(torch.float32)
    rows = [requests[i] for i in (1, 2, 5, 1)]
    rows = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.3)) for r in rows]
    high = copy.deepcopy(w).cpu().double()
    copies = {}
    def promote(t):
        if id(t) not in copies:
            copies[id(t)] = t.detach().cpu().to(torch.float64 if t.is_floating_point() else t.dtype).requires_grad_(t.requires_grad)
        return copies[id(t)]
    high_rows = [replace(r, comparison=replace(r.comparison, value=promote(r.comparison.value),
        slots={k: promote(t) for k, t in r.comparison.slots.items()}),
        content=replace(r.content, value=promote(r.content.value)), control=promote(r.control)) for r in rows]
    high_roots = dict(high.named_parameters())
    high_roots.update({name: promote(t) for name, t in roots.items() if name not in high_roots})
    records, gradients = {}, {}
    original = query_roots.coalesce
    try:
        for name in ("scalar64", "scalar32", "packed-unmerged32", "packed-merged32"):
            weights, reqs, leaves = (high, high_rows, high_roots) if name == "scalar64" else (w, rows, roots)
            if name.startswith("scalar"):
                values = [weights.full_program.step(weights, r, 1, "hard", 1.).value for r in reqs]
            else:
                weights.full_program.query_execution = "packed-v2"
                query_roots.coalesce = (lambda outputs, bars: (outputs, bars)) if name == "packed-unmerged32" else original
                values = [r.value for r in evaluate(weights.full_program, weights, reqs, kind="full", slots=1)]
            gradients[name] = {n: None if g is None else g.detach().cpu().double()
                               for n, g in vjp(sum(t.square().sum() for t in values), leaves).items()}
            records[name] = {n: None if g is None else g.tolist() for n, g in gradients[name].items()}
    finally:
        query_roots.coalesce = original
    reports = {name: collect_comparison(g, gradients["scalar64"], path=name, check_device=False)
               for name, g in gradients.items()}
    maxima = {name: max((float((g[n]-gradients["scalar64"][n]).abs().max())
                         for n in g if g[n] is not None and g[n].numel()), default=0.)
              for name, g in gradients.items()}
    comparison = collect_comparison(gradients["packed-merged32"], gradients["packed-unmerged32"],
                                     path="linear_transpose", atol=0., rtol=0., check_device=False)
    write_json(a.output, dict(device=a.device, witness="0.7*(sum of four Full squared norms)**2; duplicate row; original FP32 material promoted exactly",
        gradients=records, comparison_to_fp64=reports, max_abs_vs_fp64=maxima,
        exact_merge_comparison=comparison, note="Bounded diagnostic; original strict failure/tolerance remain. No general learning or scale claim."))
    print({"max_abs_vs_fp64": maxima, "merge_exact": comparison["strict_passed"]})


if __name__ == "__main__":
    main()
