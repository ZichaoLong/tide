"""Allocation §6 / Describe §3: batch guards, branches and independent rows.

Invalid last rows must still fail; host scalar reads must not grow per row.
Literal FP64 descriptor derivatives are separate from scalar-vs-batch rounding.
"""
from types import SimpleNamespace
import os
import pytest
import torch
from tidegraph.readout import finite_batch
from tidegraph.weighted.allocation_rows import allocate_batch
from tidegraph.weighted.dense_describe import evaluate
from tidegraph.compare import equivalent
from tidegraph.runtime import resolve_device

DEVICE, _ = resolve_device(os.environ.get("TIDE_DENSE_DEVICE", "cpu"))


@pytest.mark.parametrize("bad", ["scores", "caps", "lower", "upper", "temperature", "shift"])
@pytest.mark.parametrize("fault", ["nonfinite", "shape", "value"])
def test_allocation_batch_last_row_guards(bad, fault):
    def tensor(value):
        return torch.tensor(value, device=DEVICE)
    rows = dict(scores=[tensor([.2, .7]) for _ in range(3)],
                caps=[tensor([.7, .8]) for _ in range(3)],
                lower=[tensor(.2) for _ in range(3)],
                upper=[tensor(.8) for _ in range(3)],
                temperature=[tensor(.9) for _ in range(3)],
                shift=[tensor(.1) for _ in range(3)])
    value = rows[bad][-1]
    if fault == "nonfinite":
        rows[bad][-1] = torch.full_like(value, float("nan"))
    elif fault == "shape":
        rows[bad][-1] = value.unsqueeze(0)
    elif bad in {"scores", "shift"}:
        # Every finite score/shift is allowed; an incompatible dtype is not.
        rows[bad][-1] = value.to(torch.int64)
    else:
        rows[bad][-1] = torch.full_like(value, -1.)
    with pytest.raises(ValueError):
        allocate_batch(rows["scores"], rows["caps"], rows["lower"], rows["upper"],
                       rows["temperature"], score_shift=rows["shift"])


def test_structured_finite_guard_checks_last_row_and_keeps_roots():
    x = torch.tensor([.2, .3], device=DEVICE, requires_grad=True)
    rows = [dict(s=x.sum(), v=x*float(i), empty=x[:0]) for i in range(33)]
    finite_batch(rows)
    assert torch.equal(torch.autograd.grad(rows[-1]["v"].sum(), x)[0], torch.full_like(x, 32.))
    rows[-1]["v"] = torch.tensor([.2, float("inf")], device=DEVICE)
    with pytest.raises(ValueError, match="nonfinite"):
        finite_batch(rows)


def test_allocation_guards_and_branch_reads_have_bounded_submissions():
    def count(n):
        scores = [torch.tensor([.1, .3], device=DEVICE, requires_grad=True) for _ in range(n)]
        cap = torch.tensor([.7, .9], device=DEVICE, requires_grad=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            outputs = allocate_batch(scores, [cap]*n, .6, .6)
            sum(row.controls.square().sum() for row in outputs).backward()
        return sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")
    # The complete groups take the same numerical branch. Row count does not
    # justify one device synchronization per row in forward or in the VJP.
    assert count(32) <= count(2)+2


def test_batched_evidence_literal_derivatives_and_submissions():
    from tidegraph.weighted.packing_observation import observe
    dtype = torch.float64
    g = torch.Generator().manual_seed(137)
    def leaf(*shape):
        return torch.randn(shape, dtype=dtype, generator=g).requires_grad_()
    n, d, h, e = 17, 3, 2, 4
    u, mass, short, long = leaf(n, d), leaf(n).square(), leaf(n, h), leaf(n, h)
    ws, wl = leaf(n).square(), leaf(n).square()
    width = d+4*h+3
    memory = SimpleNamespace(config=SimpleNamespace(epsilon=.01), summary=leaf(h, d))
    p = SimpleNamespace(config=SimpleNamespace(gamma_max=1.),
        score=SimpleNamespace(matrix=leaf(3, width), hidden_bias=leaf(3), vector=leaf(3), bias=leaf()),
        evidence_weight=leaf(e, width), evidence_bias=leaf(e))
    with observe() as observation:
        actual = evaluate(p, memory, mass, u, (short, long, ws, wl))
    assert observation.metrics()["packing/projection/weighted_mm/calls"] == 3
    def norm(x):
        return x/torch.sqrt(.01+x.square().mean(-1, keepdim=True))
    cur = norm(u)
    current = cur@memory.summary.t()
    f = torch.cat((cur, mass.log1p()[:, None], short, long, ws.log1p()[:, None],
                   wl.log1p()[:, None], current*short, current*long), -1)
    expected = norm(f@p.evidence_weight.t()+p.evidence_bias)
    equivalent(torch.stack([r["evidence"] for r in actual]), expected)
    roots = (u, short, long, memory.summary, p.evidence_weight, p.evidence_bias, p.score.matrix)
    bars = leaf(n, e).detach()
    for ids in ([5], [2, 5, 2, 16], list(range(n))):
        a = sum((actual[i]["evidence"]*bars[i]).sum() for i in ids)
        b = sum((expected[i]*bars[i]).sum() for i in ids)
        grads = [torch.autograd.grad(value, roots, retain_graph=True, allow_unused=True) for value in (a, b)]
        equivalent(*grads)
        assert grads[0][-1] is None
