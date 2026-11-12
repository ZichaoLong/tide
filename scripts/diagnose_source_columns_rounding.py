#!/usr/bin/env python3
"""Bounded general-port/source-column rounding audit, CPU native vs scalar.

Uses the independent scalar schedule and the existing adversarial port fixture.
No tolerance, production computation or failed job record is changed. Forward
fields, shared parameter/input VJPs and connected zeros are checked separately.
"""
import argparse
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys

if os.environ.get("TIDE_BUILD_DIR"):
    sys.path.insert(0, os.environ["TIDE_BUILD_DIR"])
import torch
import tidegraph
import test_settle_columns as fixture
from durable_records import write_json
from isolated_cases import vjp
from tidegraph import Continuation
from tidegraph.comparison import collect_comparison
from tidegraph.native_loader import load_native
from tidegraph.reference import run as reference
from tidegraph.weighted.settle_columns import SettleColumnPlan


MODES = ("regional_total", "regional_total_zero", "regional_threshold")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objective(fields):
    # Keep cotangents identical across precisions, too. FP32 is exactly the
    # original test objective; FP64 promotes its FP32 linspace coefficients.
    return sum((t * torch.linspace(-.4, .7, t.numel(), dtype=torch.float32)
                .to(t).reshape(t.shape)).sum() for t in fields.values()
               if t.is_floating_point())


def maximum(t):
    return float(t.detach().abs().max()) if t.numel() else 0.


def distance(a, b):
    return maximum(a.detach().double() - b.detach().double())


def observe(dtype, mode):
    binding, spec = fixture.general(dtype, mode, material_dtype=torch.float32)
    # Build all leaves in FP32 first: the fixture assigns its negative-strength
    # sentinel after constructing linspace, so calling inputs(FP64, FP32) alone
    # leaves that single -0.1 assignment in FP64 rather than promoting FP32.
    layout, ports = fixture.inputs(torch.float32)
    ports = {p: replace(x, strength=x.strength.detach().to(dtype).requires_grad_(),
                       content=x.content.detach().to(dtype).requires_grad_())
             for p, x in ports.items()}
    actual = SettleColumnPlan(binding, spec, implementation="native").run_ports(layout, ports)
    expected = reference(binding.graph, binding.model, Continuation(binding.graph.identity, 2),
                         fixture.external(spec, layout, ports), 50, sealed_until=50)
    # Also asserts event/support presence, physical edge/port multiplicity,
    # int64 logical times/counts and raw state slot correspondence.
    fields = fixture.observations(actual, expected, spec)
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{name}": getattr(x, name) for p, x in ports.items()
                  for name in ("strength", "content")})
    gradients = [vjp(objective(row), roots) for row in fields]
    zero = [vjp(objective(row), roots, zero=True) for row in fields]
    reports = {"forward": collect_comparison(*fields, path="forward"),
               "gradient": collect_comparison(*gradients, path="gradient"),
               "zero_gradient": collect_comparison(*zero, path="zero_gradient")}
    assert all(r["discrete"] == "passed" and not r["differences_truncated"]
               for r in reports.values()), reports
    assert reports["forward"]["strict_passed"] and reports["zero_gradient"]["strict_passed"], reports
    if dtype == torch.float64:
        assert reports["gradient"]["strict_passed"], reports
    detach = lambda row: {k: None if t is None else t.detach().clone() for k, t in row.items()}
    return dict(fields=[detach(row) for row in fields], gradients=[detach(row) for row in gradients],
                roots=detach(roots), reports=reports, stats=actual.stats,
                zero_support=dict(connected=sum(t is not None for t in zero[0].values()),
                                  disconnected=sum(t is None for t in zero[0].values())))


def first_difference(rows):
    # This is a declared observation order, not an assertion that a profiler
    # identified the unique causal instruction. Content precedes downstream
    # Describe/control, in topological node order for this fixed graph.
    left, right = rows
    ordered = sorted((k for k in left if isinstance(k[0], int)),
                     key=lambda k: (k[0], k[1], 0 if k[2] == "content" else 1, str(k[2])))
    ordered += [k for k in left if k not in ordered]
    for key in ordered:
        if not torch.equal(left[key], right[key]):
            return dict(field=str(key), max_abs=distance(left[key], right[key]),
                        native=left[key].tolist(), scalar=right[key].tolist())
    return None


def case(mode):
    low, high = observe(torch.float32, mode), observe(torch.float64, mode)
    assert low["roots"].keys() == high["roots"].keys()
    assert all(torch.equal(t.double(), high["roots"][k]) for k, t in low["roots"].items())
    left, right = low["gradients"]
    native64, scalar64 = high["gradients"]
    details = []
    for diff in low["reports"]["gradient"]["differences"]:
        name = diff["path"][len("gradient["):-1]
        a, b = left[name], right[name]
        mask = ~torch.isclose(a, b, atol=1e-6, rtol=1e-5)
        indices = mask.nonzero()
        details.append(dict(field=name, whole_field_max_abs=distance(a, b),
            failing_coordinate_count=len(indices),
            failing_coordinates=[dict(index=index.tolist(), native32=float(a[tuple(index)]),
                scalar32=float(b[tuple(index)]), native64=float(native64[name][tuple(index)]),
                scalar64=float(scalar64[name][tuple(index)])) for index in indices[:8]],
            native32_vs_scalar64=distance(a, scalar64[name]),
            scalar32_vs_scalar64=distance(b, scalar64[name]),
            native64_vs_scalar64=distance(native64[name], scalar64[name])))
    content = [{"field": str(k), "max_abs": distance(a, low["fields"][1][k])}
               for k, a in low["fields"][0].items() if isinstance(k[0], int) and k[2] == "content"]
    return dict(mode=mode, fp32=low["reports"], same_material_fp64=high["reports"],
        fp64_vjp_max_abs=max(distance(t, scalar64[n]) for n, t in native64.items() if t is not None),
        first_observed_fp32_difference=first_difference(low["fields"]),
        aggregate_content=content, failed_gradient_fields=details,
        zero_support=low["zero_support"], roots_exactly_promoted=True,
        scope="fixed seed-7 four-node ragged/duplicate-port fixture; no whole-model precision claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="fixed production commit, not diagnostic script identity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    native = load_native()
    report = dict(source=args.source, python_module=tidegraph.__file__,
        fixture_module=fixture.__file__, fixture_sha256=digest(fixture.__file__),
        diagnostic_script=str(Path(__file__).resolve()), diagnostic_sha256=digest(__file__),
        native_module=native.__file__, native_sha256=digest(native.__file__),
        torch_version=torch.__version__, device="cpu", threads=torch.get_num_threads(),
        material="FP32 parameters and input leaves promoted exactly, same FP32 objective cotangents; VJP uses existing squared-root loss",
        strict_tolerances=dict(fp32=dict(atol=1e-6, rtol=1e-5), fp64=dict(atol=1e-10, rtol=1e-8)),
        cases=[case(mode) for mode in MODES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print({c["mode"]: dict(fp32=c["fp32"]["gradient"]["mismatch_counts"],
                           fp64_max_abs=c["fp64_vjp_max_abs"],
                           first=c["first_observed_fp32_difference"]) for c in report["cases"]})


if __name__ == "__main__":
    main()
