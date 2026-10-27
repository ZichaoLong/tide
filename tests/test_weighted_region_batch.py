"""Allocation §2–6: exact complete region rows across the formal combinations."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.content import Content
from tidegraph.history import History
from tidegraph.readout import ReadInput
from tidegraph.region import RegionInput
from tidegraph.weighted.region_batch import evaluate
from test_weighted_scoring_execution import configured, CASES


def native_region(program, requests, *, grad=True):
    import _tide_native as core
    from tidegraph.region import Selection
    from tidegraph.weighted.native import region_config
    if not requests:
        return []
    w = core.RegionWeights(); w.extra = dict(program.named_parameters(remove_duplicate=False))
    core.configure_weighted_region(w, region_config(core, program.config))
    layout = requests[0].layout
    s = layout.spec
    spec = core.Region(s.budget, s.observe_all, s.count_priority, s.read_mode, s.selector)
    result = core.weighted_region_probe(w, spec, layout.members, [r.candidates for r in requests], [r.time for r in requests], grad)
    return [Selection(active, controls, History(last_time=time)) for active, controls, time in result]


def fixture(dtype, case):
    b, _, _ = configured(dtype, case, "fc")
    p = b.model.regions[0]; layout = b.graph.region_layouts[0]
    requests = []
    for i in range(5):
        candidates = []
        for v in layout.members[:1 if i == 4 else 2]:
            w = b.model.nodes[v]
            h = Content(torch.tensor([.2+.15*i+.02*v, .3+.11*v, -.4+.06*i], dtype=dtype))
            state = w.kernel.step(w, w.initial(), h, 2**54+17)
            d = w.read_program.step(w, ReadInput(state, 2**54+19, h))
            candidates.append((v, {name: t.detach().requires_grad_() for name, t in d.items()}))
        requests.append(RegionInput(History(last_time=2**54+15), 2**54+19+i, tuple(candidates), layout, dtype, torch.device("cpu")))
    return p, requests


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_regional_complete_numeric_batches(dtype, case, native, monkeypatch):
    p, requests = fixture(dtype, case)
    with torch.no_grad():
        expected = [p.step(r) for r in requests]
    def forbidden(*a, **kw):
        raise AssertionError("region numeric batch replayed scalar region step")
    monkeypatch.setattr(p, "step", forbidden)
    with torch.no_grad():
        actual = native_region(p, requests, grad=False) if native else evaluate(p, requests)
    equivalent(expected, actual)
    assert evaluate(p, []) == []


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_regional_batch_independent_control_and_descriptor_roots(dtype, case, native, monkeypatch):
    from tidegraph.weighted.region_vjp import evaluate as batch_vjp
    from isolated_cases import vjp
    p, requests = fixture(dtype, case)
    roots = dict(p.named_parameters())
    roots.update({f"frame.{i}.node.{v}.{k}": t for i, r in enumerate(requests) for v, d in r.candidates for k, t in d.items()})
    expected = [p.step(r) for r in requests]
    def forbidden(*a, **kw):
        raise AssertionError("regional batch VJP replayed scalar step")
    monkeypatch.setattr(p, "step", forbidden)
    execute = native_region if native else batch_vjp
    actual = execute(p, requests)
    equivalent(expected, actual)
    for a, b in zip(expected, actual):
        for v in a.controls:
            assert a.controls[v].requires_grad == b.controls[v].requires_grad
            if a.controls[v].requires_grad:
                equivalent(vjp(a.controls[v], roots), vjp(b.controls[v], roots))
                equivalent(vjp(a.controls[v], roots, zero=True), vjp(b.controls[v], roots, zero=True))
    def mixed(rows):
        return sum(list(r.controls.values())[i % len(r.controls)].square() for i, r in enumerate(rows))
    equivalent(vjp(mixed(expected), roots), vjp(mixed(actual), roots))
    assert execute(p, []) == []


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_region_alias_versions_and_first_order(native):
    from tidegraph.weighted.region_vjp import evaluate as batch_vjp
    from isolated_cases import vjp
    execute = native_region if native else batch_vjp
    p, rows = fixture(torch.float64, 3)
    p.query.out_weight = p.query.in_weight
    roots = dict(p.named_parameters())
    expected = sum(sum(r.controls.values())**2 for r in [p.step(r) for r in rows])
    actual = sum(sum(r.controls.values())**2 for r in execute(p, rows))
    equivalent(vjp(expected, roots), vjp(actual, roots))
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(actual, tuple(roots.values()), create_graph=True, allow_unused=True)
    actual = execute(p, rows)[0].controls[rows[0].candidates[0][0]]
    with torch.no_grad():
        p.query.in_weight.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(actual, tuple(roots.values()), allow_unused=True)
