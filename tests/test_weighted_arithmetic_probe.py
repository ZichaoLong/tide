"""Diagnostic interventions are finite, observable and cannot become fallback."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_arithmetic_probe import ArithmeticProbe
from weighted_numeric_differences import tensors
from tidegraph.weighted.normalization import norm, _Normalize


def probe(name):
    return ArithmeticProbe(name, dtype="float32", implementation="python")


def test_linear_cancellation_independent_vjp_and_optimizer_exclusion():
    p = probe("linear64")
    x = torch.tensor([[1e8, 1., -1e8]], requires_grad=True)
    w = torch.ones(3, requires_grad=True)
    with p.phase("forward"):
        y = x@w
    assert y.dtype == torch.float32 and y.item() == 1.
    with p.phase("backward"):
        dx, dw = torch.autograd.grad(y.sum(), (x, w))
    assert torch.equal(dx, torch.ones_like(x)) and torch.equal(dw, x.detach()[0])
    assert any(k.startswith("forward:") for k in p.record()["operation_counts"])
    assert any(k.startswith("backward:") for k in p.record()["operation_counts"])
    with pytest.raises(ValueError, match="optimizer"):
        with p.phase("optimizer"):
            pass


def test_backward_only_does_not_intercept_forward_and_preserves_int64():
    p = probe("backward_linear_reduce64")
    with p.phase("forward"):
        torch.ones(3)@torch.ones(3)
    assert not p.record()["operation_counts"]
    t = torch.tensor([2**60+3, 2**60+5], dtype=torch.int64)
    with p.phase("backward"):
        result = t.sum()
    assert result.dtype == torch.int64 and result.item() == 2**61+8
    assert not p.record()["operation_counts"]


def test_normalization_literal_jacobian_versions_and_patch_restoration():
    p = probe("norm64")
    original = _Normalize.__dict__["forward"]
    x = torch.tensor([.01, -.2, .4], requires_grad=True)
    b = torch.tensor([.3, .1, -.7])
    epsilon = .03
    with p.phase("forward"):
        y = norm(x, epsilon)
    assert _Normalize.__dict__["forward"] is original
    v = x.detach().double()
    d = (epsilon+v.square().mean()).sqrt()
    jac = torch.eye(3, dtype=torch.float64)/d-torch.outer(v, v)/(3*d**3)
    assert torch.equal(y, (v/d).float())
    with p.phase("backward"):
        actual, = torch.autograd.grad(y, x, b)
    torch.testing.assert_close(actual, (jac@b.double()).float(), atol=1e-7, rtol=1e-6)
    with pytest.raises(RuntimeError, match="sentinel"):
        with p.phase("forward"):
            raise RuntimeError("sentinel")
    assert _Normalize.__dict__["forward"] is original
    with p.phase("forward"):
        y = norm(x, epsilon)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        with p.phase("backward"):
            y.sum().backward()


@pytest.mark.parametrize("precision,implementation", [("float64", "python"), ("float32", "native")])
def test_unsupported_intervention_refused_before_work(precision, implementation):
    with pytest.raises(ValueError, match="Python CPU FP32"):
        ArithmeticProbe("linear64", dtype=precision, implementation=implementation)


def test_difference_report_retains_connections_small_rounding_and_nonfinite():
    a = dict(none=None, integer=torch.tensor([2**60+3]), bad=torch.tensor([float("inf")]),
             small=torch.tensor([1.]), far=torch.tensor([2.]))
    b = dict(none=torch.tensor(0.), integer=torch.tensor([2**60+4]), bad=torch.tensor([float("inf")]),
             small=torch.nextafter(a["small"], torch.tensor([2.])), far=torch.tensor([3.]))
    result = tensors(a, b)
    assert result["failing_tensors"] == 4 and result["nonidentical_tensors"] == 3
    assert {r["name"] for r in result["worst"]} == {"none", "integer", "bad", "far"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scoring_dtype", ["profile", "float64"])
def test_intervention_two_windows_optimizer_and_exact_resume(tmp_path, scoring_dtype):
    from weighted_topology import topology
    from weighted_workload_plan import make_plan
    from durable_records import write_json
    plan, _ = make_plan(topology("fc", depth=2, region_width=2), width=4, samples=1, positions=2)
    write_json(tmp_path/"plan.json", plan)
    out = tmp_path/"diagnostic"
    command = [sys.executable, str(ROOT/"scripts/diagnose_weighted_workload.py"), "--workload", str(tmp_path/"plan.json"),
               "--dtype", "float32", "--implementation", "python", "--checkpoint-check", "--output-dir", str(out),
               "--arithmetic", "linear_reduce_norm64", "--scoring-dtype", scoring_dtype]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    record = json.loads((out/"result.json").read_text())
    assert record["state"] == "passed" and record["steps"][0]["checkpoint_exact_decode"]
    assert record["steps"][1]["exact_resume"] == [dict(records=None, gradients=None, parameters=None, optimizer=None)]*2
    assert record["arithmetic"]["operation_counts"] and record["scoring_dtype"] == scoring_dtype
    assert all(s["gradient_detail"]["failing_tensors"] == s["parameter_detail"]["failing_tensors"] == 0
               for s in record["steps"])
