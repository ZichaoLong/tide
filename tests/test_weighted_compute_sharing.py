"""Architecture §2.2 / FC §6.2: shared computation, every use, private state.

Scalar reference scheduling evaluates every node independently. Only identical
ordered value/class rows and parameter identities license candidate reuse.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import KVConfig, RegionalConfig, bind_layered, fully_connected, layered_inputs
from isolated_cases import vjp
from test_weighted_execution import variables
from test_weighted_source_binding import binding, MODES


def fixture(dtype, mode):
    spec, b = bind_layered(fully_connected(3, 3), memory=KVConfig(width=2, summary_width=2),
                           regions=RegionalConfig(k=3), source_binding=binding(mode),
                           alpha_agg=.3, source_hidden_width=3, dtype=dtype)
    x = torch.tensor([[[.2, -.4], [.5, .7]], [[.3, -.2], [.4, .6]]], dtype=dtype, requires_grad=True)
    return b, layered_inputs(spec, x), 2*spec.stride


def runner(b, native):
    opts = dict(aggregate_autograd="batched", full_autograd="batched")
    return Native(b.graph, b.model, algorithm="greedy", packed=True, **opts).run if native else (
        lambda *a, **kw: greedy(b.graph, b.model, *a, **opts, **kw))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_exact_aggregate_reuse_and_independent_roots(dtype, mode, native):
    b, xs, stop = fixture(dtype, mode)
    assert len({id(x.value) for x in xs}) == 4
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    roots = variables(b, xs)
    for left, right in ((objective(expected), objective(actual)),
                        (expected.trace[0]["content"], actual.trace[0]["content"]),
                        (expected.outputs[0][-1], actual.outputs[0][-1])):
        equivalent(vjp(left, roots), vjp(right, roots))
        equivalent(vjp(left*0, roots), vjp(right*0, roots))
    reused = actual.stats.get("aggregate_reused_events", 0)
    computed = actual.stats["aggregate_computed_events"]
    assert computed+reused == actual.stats["candidate_events"]
    if mode in {"region", "source"}:
        assert reused > 0 and computed*3 == actual.stats["candidate_events"]
        # Same aggregate does not merge node-specific memory/adapter states.
        assert not torch.equal(actual.continuation.states[0, 0].slots["keys"],
                               actual.continuation.states[0, 1].slots["keys"])
    else:
        assert reused == 0


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_equal_independent_inputs_are_not_shared_and_every_use_contributes(dtype, native):
    b, xs, stop = fixture(dtype, "region")
    # Same numbers, separately owned input roots. No equality-based merging.
    xs = [replace(x, value=x.value.detach().clone().requires_grad_()) for x in xs]
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    roots = variables(b, xs)
    first = actual.trace[0]
    a = vjp(first["content"], roots)
    for i, x in enumerate(xs):
        assert (a[f"input.{i}"] is not None) == (x.batch == first["batch"] and x.time == first["time"] and
                                                 b.graph.inputs[x.port] == first["node"])
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    # Subsequent FC regions receive actual shared broadcasts and can reuse.
    assert actual.stats["aggregate_reused_events"] > 0
    assert actual.stats["aggregate_computed_events"]*3 > actual.stats["candidate_events"]


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_shared_aggregate_optimizer_and_pending_resume(dtype, native, tmp_path):
    from tidegraph.checkpoint import save, load
    a, xa, stop = fixture(dtype, "region"); b, xb, _ = fixture(dtype, "region")
    def opt(model):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    oa, ob = opt(a.model), opt(b.model); execute = runner(b, native)
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ra = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        rb = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        equivalent(ra, rb)
        objective(ra).backward(); objective(rb).backward()
        for x, y in zip(a.model.parameters(), b.model.parameters()): equivalent(x.grad, y.grad)
        oa.step(); ob.step(); equivalent(a.model.state_dict(), b.model.state_dict())
    cut = 2
    first = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    assert first.continuation.pending
    path = tmp_path/"shared.pt"; save(path, b.graph, b.model, first.continuation, ob)
    c, _, _ = fixture(dtype, "region"); oc = opt(c.model)
    restored = load(path, c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored); equivalent(ob.state_dict(), oc.state_dict())
    tail = [x for x in xb if x.time >= cut]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    rb = execute(first.continuation.detach(), tail, stop, sealed_until=stop)
    rc = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(rb, rc)
    objective(rb).backward(); objective(rc).backward(); ob.step(); oc.step()
    equivalent(b.model.state_dict(), c.model.state_dict()); equivalent(ob.state_dict(), oc.state_dict())
