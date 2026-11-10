#!/usr/bin/env python3
"""Bounded three-way audit of the retained dense-v1 FP32 evidence-root witness."""
import copy
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from test_weighted_dense_query import independent
from tidegraph.weighted.query_vjp import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-execution", choices=("dense-v1", "packed-v2"), default="dense-v1")
    p.add_argument("--witness", choices=("evidence-bias", "publication"), default="evidence-bias")
    args = p.parse_args()
    torch.set_num_threads(1)
    w, rows, _ = independent(torch.float32)
    high = copy.deepcopy(w).double()
    def convert(t):
        return t.detach().double() if t.is_floating_point() else t.detach()
    converted = []
    for row in rows:
        state = row.state
        if state is not None:
            state = replace(state, value=convert(state.value), slots={k: convert(t) for k, t in state.slots.items()})
        converted.append(replace(row, state=state, content=replace(row.content, value=convert(row.content.value))))
    outputs, gradients = {}, {}
    for name, weights, requests in (("scalar-fp32", w, rows), ("dense-fp32", w, rows),
                                    ("scalar-fp64", high, converted), ("dense-fp64", high, converted)):
        if args.witness == "publication":
            requests = [requests[1], requests[1], requests[2]]
        if name.startswith("dense"):
            weights.read_program.query_execution = args.query_execution
            results = evaluate(weights.read_program, weights, requests)
        else:
            results = [weights.read_program.step(weights, row) for row in requests]
        if args.witness == "publication":
            value = results[1]["score"].square()+results[2]["evidence"].square().sum()
            objective, parameter = value, weights.read_program.evidence_weight
        else:
            value = results[1]["evidence"]
            objective, parameter = .7*value.square().sum(), weights.read_program.evidence_bias
        outputs[name] = value.detach().tolist()
        gradients[name] = torch.autograd.grad(objective, parameter)[0].detach().double()
    truth = gradients["scalar-fp64"]
    report = dict(query_execution=args.query_execution,
        witness=args.witness, material="exact same FP32 parameters/inputs promoted to FP64",
        outputs=outputs, gradients={k: v.tolist() for k, v in gradients.items()},
        max_abs_vs_fp64={k: float((v-truth).abs().max()) for k, v in gradients.items()},
        note="Both finite FP32 reductions approximate the FP64 formula. This local audit does not relax strict tolerance or certify arbitrary scales.")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
