"""Generic StateClock composition, distinct from weighted.bind's original clock.

Independent scalar scheduling anchors every state root. The existing literal
memory formula tests anchor the local functions used by that schedule.
"""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, StateClock
from tidegraph.ops import Model
from tidegraph.reference import run as scalar
from tidegraph.frontier import run as packed
from tidegraph.compare import equivalent
from isolated_cases import vjp
from test_weighted_packed import binding


def fixture(dtype, family):
    bound = binding(dtype, family)
    kernel = bound.model.nodes[0].kernel
    graph = Graph((Node(0, memory=bound.graph.nodes[0].memory,
                        state_clock=StateClock(2, 0, 1)),), (), (Region(1),), (0,), (0,))
    model = Model(graph, width=3, dtype=dtype, state_programs={0: kernel})
    xs = [External(b, 0, j, 2*j, torch.tensor([mass, .3+b*.1, -.2+j*.1], dtype=dtype,
                                             requires_grad=True))
          for b, masses in enumerate(((.6, .4), (0., 0.), (.5, 0.))) for j, mass in enumerate(masses)]
    states, roots = {}, dict(model.named_parameters())
    for b in range(3):
        s = model.nodes[0].initial()
        s = replace(s, value=s.value.detach().requires_grad_(), slots={
            k: t.detach().requires_grad_(t.is_floating_point()) for k, t in s.slots.items()})
        states[b, 0] = s
        roots[f"initial.{b}.value"] = s.value
        roots.update({f"initial.{b}.{k}": t for k, t in s.slots.items() if t.requires_grad})
    roots.update({f"input.{i}": x.value for i, x in enumerate(xs)})
    return graph, model, Continuation(graph.identity, 3, states=states), xs, roots


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_clocked_weighted_independent_roots(dtype, family):
    g, m, q, xs, roots = fixture(dtype, family)
    a, b = (run(g, m, q, xs, 4, sealed_until=4) for run in (scalar, packed))
    equivalent(a, b)
    for owner in q.states:
        sa, sb = a.continuation.states[owner], b.continuation.states[owner]
        for x, y in [(sa.value, sb.value), *[(sa.slots[k], sb.slots[k]) for k in sa.slots
                                           if sa.slots[k].is_floating_point()]]:
            equivalent(vjp(x, roots), vjp(y, roots))
    assert b.stats["batched_state_events"] == len(xs)
    assert b.stats.get("semantic_state_replays", 0) == 0


def test_clocked_parameter_versions_are_checked():
    g, m, q, xs, roots = fixture(torch.float64, "kv")
    result = packed(g, m, q, xs, 4, sealed_until=4)
    with torch.no_grad():
        m.nodes[0].kernel.program.base.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        vjp(result.continuation.states[0, 0].slots["keys"], roots)


@pytest.mark.parametrize("family", ["kv", "delta"])
def test_clocked_shared_parameters_and_optimizer(dtype, family):
    g, m, _, xs, _ = fixture(dtype, family)
    if family == "delta":
        kernel = m.nodes[0].kernel.program
        # value and summary have the same shape in this fixture.
        kernel.value = kernel.summary
    candidate = deepcopy(m)
    optimizers = [torch.optim.SGD(x.parameters(), lr=.01, momentum=.2) for x in (m, candidate)]
    before = {k: p.detach().clone() for k, p in m.named_parameters()}
    for _ in range(3):
        for run, model, opt in zip((scalar, packed), (m, candidate), optimizers):
            opt.zero_grad(set_to_none=True)
            result = run(g, model, Continuation(g.identity, 3), xs, 4, sealed_until=4)
            state = result.continuation.states[0, 0]
            loss = sum(t.square().sum() for t in state.slots.values() if t.is_floating_point())
            loss.backward()
            opt.step()
        equivalent(dict(m.named_parameters()), dict(candidate.named_parameters()))
        equivalent({k: p.grad for k, p in m.named_parameters()},
                   {k: p.grad for k, p in candidate.named_parameters()})
    assert any(not torch.equal(before[k], p) for k, p in m.named_parameters())
    if family == "delta":
        assert candidate.nodes[0].kernel.program.value is candidate.nodes[0].kernel.program.summary
