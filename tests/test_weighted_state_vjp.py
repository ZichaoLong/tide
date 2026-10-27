"""Weighted Upd first-order batch VJP: independent roots and nested owners.

The scalar schedule is independent; literal memory formulas are anchored in
test_weighted_*_formulas. These cases expose spurious zero connections that a
single all-output loss would miss, including empty convolution/KV state slots.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import RecurrentConfig, bind
from test_weighted_packed import binding
from test_weighted_execution import variables
from isolated_cases import vjp


FAMILIES = ["kv", "affine", "delta", "gated_delta", "ssd"]


def fixture(dtype, family):
    b = binding(dtype, family); w = b.model.nodes[0]
    initial, roots = {}, {}
    for i in range(4):
        s = w.initial()
        s = replace(s, value=s.value.detach().requires_grad_(), slots={
            k: t.detach().requires_grad_(t.is_floating_point()) for k, t in s.slots.items()})
        initial[i, 0] = s
        roots[f"old.{i}.value"] = s.value
        roots.update({f"old.{i}.{k}": t for k, t in s.slots.items() if t.requires_grad})
    xs = [External(i, 0, 0, 0, torch.tensor([mass, .2+.1*i, -.4], dtype=dtype, requires_grad=True))
          for i, mass in enumerate((.7, 0., .3, .4))]
    xs += [External(i, 0, 1, 2, torch.tensor([.5, -.3, .7], dtype=dtype, requires_grad=True)) for i in (0, 2)]
    roots.update(variables(b, xs))
    return b, Continuation(b.graph.identity, 4, states=initial), xs, roots


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_weighted_independent_slots_mixed_zero_rows_and_empty_cache(dtype, family, native, monkeypatch):
    b, q, xs, roots = fixture(dtype, family)
    expected = reference(b.graph, b.model, q, xs, 4, sealed_until=4)
    if native:
        runner = Native(b.graph, b.model, algorithm="frontier", packed=True).run
    else:
        def forbidden(*args, **kwargs):
            raise AssertionError("batched state backward construction called scalar Upd")
        monkeypatch.setattr(b.model.nodes[0].kernel, "step", forbidden)
        runner = lambda *a, **kw: frontier(b.graph, b.model, *a, **kw)
    actual = runner(q, xs, 4, sealed_until=4)
    equivalent(expected, actual)
    pairs = [(objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1])]
    for i in range(4):
        a, c = expected.continuation.states[i, 0], actual.continuation.states[i, 0]
        pairs.append((a.value, c.value))
        pairs += [(a.slots[k], c.slots[k]) for k in a.slots if a.slots[k].is_floating_point()]
    # Different slot activity patterns in the same batch, including connected zero.
    key = "keys" if family == "kv" else "H"
    stat = "weights" if family == "kv" else "W"
    def mixed(r):
        s = r.continuation.states
        return s[0, 0].slots[key].square().sum()+0*s[2, 0].slots[stat].sum()+s[1, 0].value.sum()
    pairs.append((mixed(expected), mixed(actual)))
    for a, c in pairs:
        equivalent(vjp(a, roots), vjp(c, roots))
    assert actual.stats["batched_state_events"] > 0
    assert actual.stats.get("semantic_state_replays", 0) == 0


def test_nested_shared_parameters_are_captured_once_and_accumulate(dtype):
    c = RecurrentConfig(memory_family="delta", width=2, summary_width=2, key_width=2, value_width=2)
    b = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=c, dtype=dtype)
    m = b.model.nodes[0].kernel
    m.key = m.value = m.summary
    xs = [External(i, 0, j, j, torch.tensor([.4, .3+i*.1, -.2+j*.3], dtype=dtype, requires_grad=True))
          for i in range(3) for j in range(2)]
    q = Continuation(b.graph.identity, 3)
    a = reference(b.graph, b.model, q, xs, 3, sealed_until=3)
    c = frontier(b.graph, b.model, q, xs, 3, sealed_until=3)
    equivalent(a, c)
    equivalent(vjp(objective(a), variables(b, xs)), vjp(objective(c), variables(b, xs)))
    assert m.key is m.value is m.summary
    assert c.stats["semantic_state_replays"] == 0


def test_nested_parameter_version_and_first_order_boundary():
    for mutate in (False, True):
        b, q, xs, roots = fixture(torch.float64, "gated_delta")
        result = frontier(b.graph, b.model, q, xs, 4, sealed_until=4)
        target = result.continuation.states[0, 0].slots["H"].sum()
        if mutate:
            with torch.no_grad(): b.model.nodes[0].kernel.conv.weight.add_(.01)
        with pytest.raises((ValueError, RuntimeError), match="first-order|modified by an inplace"):
            torch.autograd.grad(target, tuple(roots.values()), create_graph=not mutate, allow_unused=True)
