"""KV §2–4: unique numeric storage is not a shared gradient identity."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.comparison import NumericMismatch
from tidegraph.weighted import snapshot_bank as bank
from tidegraph.weighted.packing_observation import observe
from isolated_cases import vjp


def test_shared_projection_literal_values_independent_adjoints_and_versions(dtype):
    from tidegraph.runtime import resolve_device
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    base = torch.arange(32, dtype=dtype, device=device).reshape(8, 4)/13
    # Overlap and a gap, nonzero offsets, empty row and independent leaf roots.
    rows = [base[a:b].detach().requires_grad_() for a, b in ((1, 4), (2, 5), (7, 8), (3, 3))]
    weight = torch.linspace(-.3, .8, 12, dtype=dtype, device=device).reshape(3, 4).requires_grad_()
    visible = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=torch.bool, device=device)
    with observe() as counts:
        actual = bank.project(rows, weight, visible)
    padded = torch.stack([torch.cat((r, r.new_zeros((3-len(r), 4)))) for r in rows])
    safe = torch.where(visible[..., None], padded, torch.zeros_like(padded))
    expected = (weight@safe.reshape(-1, 4).t()).t().reshape(4, 3, 3)
    equivalent(actual, expected)
    roots = dict(weight=weight, **{str(i): x for i, x in enumerate(rows)})
    for loss in (lambda x: x.square().sum(), lambda x: x[0].sum(), lambda x: 0*x[-1].sum()):
        equivalent(vjp(loss(actual), roots), vjp(loss(expected), roots))
    assert counts.metrics()["packing/projection/kv_unique_projection/rows"] == 5
    with torch.no_grad():
        rows[0].add_(1)
    with pytest.raises(RuntimeError, match="modified|version|inplace"):
        actual.sum().backward()


def test_bank_masks_ignored_huge_values_before_projection_and_declines_unshared(dtype):
    base = torch.ones(5, 2, dtype=dtype)
    base[1] = torch.finfo(dtype).max
    rows = [base[:3].detach().requires_grad_(), base[1:4].detach().requires_grad_()]
    visible = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    weight = torch.full((2, 2), 4., dtype=dtype, requires_grad=True)
    actual = bank.project(rows, weight, visible)
    assert bool(torch.isfinite(actual).all())
    for grad in torch.autograd.grad(actual.sum(), [weight, *rows]):
        assert bool(torch.isfinite(grad).all())
    assert bank.gather([t.clone() for t in rows]) is None
    assert bank.gather([base[:, ::2], base[:, ::2]]) is None


def query_case(dtype, device, kind):
    from tidegraph.content import Content
    from tidegraph.full import FullInput
    from tidegraph.readout import ReadInput
    from test_weighted_read_batch import fixture
    w, _ = fixture(dtype, "kv", "qry", False)
    w.to(device)
    clock = 2**54+17
    state = w.initial()
    for j in range(7):
        state = w.kernel.step(w, state, Content(torch.tensor([.3, .2+.02*j, -.4], dtype=dtype, device=device)), clock+j)
    slots = {k: t.detach() for k, t in state.slots.items()}
    roots = dict(w.named_parameters())
    requests = []
    for i, (a, b) in enumerate(((0, 3), (0, 4), (1, 5))):
        values = {k: t[a:b].detach().requires_grad_(t.is_floating_point()) for k, t in slots.items()}
        for name, t in values.items():
            if t.requires_grad: roots[f"{i}.{name}"] = t
        s = replace(state, slots=values)
        h = torch.tensor([.5, -.2+.1*i, .6], dtype=dtype, device=device, requires_grad=True)
        roots[f"{i}.content"] = h
        requests.append(ReadInput(s, clock+6+i, Content(h)))
    if kind == "read":
        p = w.read_program
        expected = [p.step(w, r) for r in requests]
    else:
        requests = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.4, requires_grad=True)) for r in requests]
        for i, r in enumerate(requests): roots[f"{i}.control"] = r.control
        p = w.full_program
        expected = [p.step(w, r, 2, "hard", 1.) for r in requests]
    return w, p, requests, roots, expected


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("kind", ["read", "full"])
def test_query_shared_storage_keeps_independent_roots_and_partial_cotangents(dtype, native, kind, monkeypatch):
    from tidegraph.runtime import resolve_device
    from tidegraph.weighted.query_vjp import evaluate
    from test_weighted_query_vjp import native_query
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    w, p, requests, roots, scalar = query_case(dtype, device, kind)
    p.query_execution = "packed-v2"
    # Original independently packed snapshots provide an incremental regression
    # anchor. Literal bank derivatives above and scalar FP64 below are independent.
    with monkeypatch.context() as m:
        m.setattr(bank, "plan", lambda _: None)
        expected = evaluate(p, w, requests, kind=kind, slots=2)
    actual = (native_query if native else evaluate)(p, w, requests, kind=kind, slots=2)
    equivalent(actual, expected)
    equivalent(actual, scalar)
    a = [[r[k] for k in sorted(r)] if kind == "read" else [r.value] for r in actual]
    e = [[r[k] for k in sorted(r)] if kind == "read" else [r.value] for r in expected]
    for aa, ee in zip(a, e):
        for x, y in zip(aa, ee):
            equivalent(vjp(x, roots), vjp(y, roots))
    equivalent(vjp(sum(r[0].square().sum() for r in a), roots),
               vjp(sum(r[0].square().sum() for r in e), roots))
    if dtype == torch.float64 or kind == "full":
        literal = [[r[k] for k in sorted(r)] if kind == "read" else [r.value] for r in scalar]
        for aa, ee in zip(a, literal):
            for x, y in zip(aa, ee):
                equivalent(vjp(x, roots), vjp(y, roots))


@pytest.mark.xfail(strict=True, raises=NumericMismatch,
                   reason="Existing packed cat-evidence FP32 scalar gradient gap; shared bank disabled is identical")
def test_snapshot_scalar_fp32_evidence_strict_witness():
    from tidegraph.weighted.query_vjp import evaluate
    w, p, requests, roots, scalar = query_case(torch.float32, torch.device("cpu"), "read")
    p.query_execution = "packed-v2"
    actual = evaluate(p, w, requests)[0]["evidence"]
    equivalent(vjp(actual, roots), vjp(scalar[0]["evidence"], roots))


def test_snapshot_fp32_gap_same_material_fp64_and_previous_path(monkeypatch):
    from tidegraph.weighted.query_vjp import evaluate
    w, p, requests, roots, scalar = query_case(torch.float32, torch.device("cpu"), "read")
    p.query_execution = "packed-v2"
    new = vjp(evaluate(p, w, requests)[0]["evidence"], roots)
    with monkeypatch.context() as m:
        m.setattr(bank, "plan", lambda _: None)
        old = vjp(evaluate(p, w, requests)[0]["evidence"], roots)
    equivalent(new, old, atol=0, rtol=0)
    name = "read_program.evidence_weight"
    raw = vjp(scalar[0]["evidence"], roots)[name]
    w.double()
    lifted = []
    for r in requests:
        s = r.state
        slots = {k: t.double().detach().requires_grad_(t.requires_grad) if t.is_floating_point() else t
                 for k, t in s.slots.items()}
        lifted.append(replace(r, state=replace(s, value=s.value.double().detach(), slots=slots),
                              content=replace(r.content, value=r.content.value.double().detach().requires_grad_())))
    roots64 = dict(w.named_parameters())
    candidate = vjp(evaluate(p, w, lifted)[0]["evidence"], roots64)
    literal = vjp(p.step(w, lifted[0])["evidence"], roots64)
    equivalent(candidate, literal)
    # Bounded witness only; the strict FP32 comparison above still fails.
    assert (new[name].double()-candidate[name]).abs().max() < 2e-6
    assert (raw.double()-candidate[name]).abs().max() < 4e-6
