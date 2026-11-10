#!/usr/bin/env python3
"""Finite same-material audit of capability-selected Describe radial gradients."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from isolated_cases import vjp
from test_weighted_describe_capability import prepared
from test_weighted_read_batch import CASES
from tidegraph.comparison import collect_comparison
from tidegraph.weighted.query_execution import execution_view
from tidegraph.weighted.query_vjp import evaluate


def promote(weights, rows):
    """Promote existing FP32 material, preserving parameter aliases/int64 values."""
    high = copy.deepcopy(weights).double()
    roots = {n: p for n, p in high.named_parameters() if p.requires_grad}
    converted = []
    for i, row in enumerate(rows):
        def leaf(name, t):
            value = (t.detach().double().requires_grad_(t.requires_grad)
                     if t.is_floating_point() else t.detach().clone())
            if value.requires_grad:
                roots[f"row.{i}.{name}"] = value
            return value
        state = row.state
        if state is not None:
            state = replace(state, value=leaf("value", state.value),
                            slots={k: leaf(k, t) for k, t in state.slots.items()})
        converted.append(replace(row, state=state,
                                 content=replace(row.content, value=leaf("content", row.content.value))))
    model = torch.nn.Module(); model.nodes = torch.nn.ModuleList([high])
    return high, execution_view(model, "auto").nodes[0], converted, roots


def distance(a, b):
    return float((a.double()-b.double()).abs().max()) if a.numel() else 0.


def case(family, read):
    _, scalar, selected, rows, roots = prepared(torch.float32, family, read)
    high, high_selected, high_rows, high_roots = promote(scalar, rows)
    outputs = {
        "scalar32": [scalar.read_program.step(scalar, r) for r in rows],
        "batch32": evaluate(selected.read_program, selected, rows),
        "scalar64": [high.read_program.step(high, r) for r in high_rows],
        "batch64": evaluate(high_selected.read_program, high_selected, high_rows),
    }
    def combined(vals):
        return vals[1]["score"]*3+vals[2]["evidence"].sum()+vals[0]["mu"]*0+vals[-1]["w"]
    tests = [(f"{i}.{field}", lambda vals, i=i, field=field: vals[i][field])
             for i in (0, 1, 2, len(rows)-1) for field in outputs["scalar32"][i]]
    tests.append(("combined", combined))
    failures, high_max = [], 0.
    for label, root in tests:
        grads = {name: vjp(root(result), roots if name.endswith("32") else high_roots)
                 for name, result in outputs.items()}
        low = collect_comparison(grads["batch32"], grads["scalar32"], path=label)
        high_report = collect_comparison(grads["batch64"], grads["scalar64"], path=label)
        assert high_report["strict_passed"], high_report
        for param, a in grads["scalar64"].items():
            if a is not None:
                high_max = max(high_max, distance(a, grads["batch64"][param]))
        if not low["strict_passed"]:
            assert low["discrete"] == "passed" and not low["differences_truncated"], low
            detail = []
            for diff in low["differences"]:
                param = diff["path"][len(label)+1:-1]
                truth = grads["scalar64"][param]
                detail.append(dict(parameter=param, strict_max_abs=diff["max_abs_error"],
                    scalar32_vs64=distance(grads["scalar32"][param], truth),
                    batch32_vs64=distance(grads["batch32"][param], truth),
                    scalar64_batch64=distance(grads["batch64"][param], truth),
                    high_gradient_max=float(truth.abs().max())))
            failures.append(dict(root=label, differences=detail))
    return dict(family=family, read=read, fp64_all_roots_max_abs=high_max,
                strict_fp32_failures=failures)


def main():
    torch.set_num_threads(1)
    report = dict(material="same FP32 parameters, raw states and input leaves promoted to FP64; aliases retained",
                  thresholds=dict(atol=1e-6, rtol=1e-5), cases=[case(*c) for c in CASES],
                  note="Finite local diagnostic; strict failures remain failures, not whole-model precision qualification.")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
