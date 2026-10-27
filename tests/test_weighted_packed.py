"""Pinned KV §5 / recurrent §4: joint prefixes, raw states and causal roots.

Literal prefix equations and noncommuting compositions anchor the math;
independent scalar schedules anchor whole graph records and VJP connectivity.
"""
import math
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.content import Content
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.packing import PackedSequence
from tidegraph.reference import run as reference
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind
from tidegraph.weighted.packing import matrix_scan
from isolated_cases import vjp
from test_weighted_execution import variables


def binding(dtype, family, width=2, **kwargs):
    if family in {"gated_delta", "ssd"}:
        kwargs = dict(heads=4, groups=2 if family == "ssd" else 1, conv_size=3, conv_stride=2,
                      matrix_rates=(.03, .07, 0., .1), **kwargs)
    c = KVConfig(width=width, summary_width=3, window=5, **kwargs) if family == "kv" else RecurrentConfig(
        memory_family=family, width=width, summary_width=3, key_width=2, value_width=3, **kwargs)
    r = RegionalConfig(region_control_mode="fixed", region_evidence=None, total_fraction=.6)
    return bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=c, regions=r, dtype=dtype)


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_ragged_positive_prefixes_preserve_raw_zero_and_int64(dtype, family, monkeypatch):
    b = binding(dtype, family)
    w = b.model.nodes[0]; m = w.kernel; clock = 2**54+17
    old = [m.step(w, m.initial(w), Content(torch.tensor([.4, .2+i*.1, -.3], dtype=dtype)), clock)
           for i in range(4)]
    sequences = (([0., .4, 1., 0.], [1, 2, 4, 8]), ([0., 0., .3], [1, 4, 9]),
                 ([0., 0.], [2, 8]), ([1., .5], [3, 7]))
    rows, times, offsets, expected = [], [], [0], []
    for i, (masses, stamps) in enumerate(sequences):
        state = old[i]
        for j, (mass, stamp) in enumerate(zip(masses, stamps)):
            h = torch.tensor([mass, .2+.1*j, -.5+.05*i], dtype=dtype)
            rows.append(h); times.append(clock+stamp)
            state = m.step(w, state, Content(h), clock+stamp)
            expected.append(state)
        offsets.append(len(rows))
    batch = PackedSequence(torch.stack(rows), offsets, [(i, 0) for i in range(4)], times, [Content(h) for h in rows])
    def forbidden(*args, **kw):
        raise AssertionError("joint calculation called scalar Upd")
    monkeypatch.setattr(m, "step", forbidden)
    with torch.no_grad():
        actual, calls = m.packed_sequence(w, old, batch)
    equivalent(expected, actual)
    assert calls == (1 if family == "kv" else 2)
    assert actual[0] is old[0] and actual[3] is actual[2]
    assert actual[7] is old[2] and actual[8] is old[2]
    if family == "kv":
        assert actual[3].slots["times"].tolist() == [clock, clock+2, clock+4]
    else:
        assert actual[3].slots["time"].item() == clock+4
    assert m.joint_sequence


def test_noncommuting_affine_scan_and_direction(dtype):
    # Right multiplication order is observable; no candidate step is an oracle.
    a = torch.tensor([[[1., .2], [0., .7]], [[.4, 0.], [.3, .8]], [[1., -.1], [.4, .5]]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[[.1, .4]], [[.3, -.2]], [[-.1, .2]]], dtype=dtype, requires_grad=True)
    initial = torch.tensor([[.2, -.3]], dtype=dtype, requires_grad=True)
    expected = torch.stack((initial@a[0]+b[0], initial@a[0]@a[1]+b[0]@a[1]+b[1],
                            initial@a[0]@a[1]@a[2]+b[0]@a[1]@a[2]+b[1]@a[2]+b[2]))
    actual = matrix_scan(a, b, initial)
    equivalent(expected, actual)
    roots = dict(a=a, b=b, initial=initial)
    equivalent(vjp(expected.square(), roots), vjp(actual.square(), roots))


@pytest.mark.parametrize("family", ["affine", "delta"])
def test_literal_recurrent_positive_prefixes(dtype, family):
    from test_weighted_recurrent_formulas import local
    w, m, _ = local(dtype, family, width=1, summary_width=1, key_width=1, value_width=1, epsilon=1.)
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.key.fill_(math.sqrt(2)); m.value.fill_(math.sqrt(2)); m.summary.fill_(math.sqrt(2))
        h = torch.tensor([[1., 1.], [1., 1.], [0., 9.], [1., 1.]], dtype=dtype)
        batch = PackedSequence(h, [0, 4], [(0, 0)], [0, 1, 2, 4], [Content(row) for row in h])
        actual, _ = m.packed_sequence(w, [m.initial(w)], batch)
    gain, retain = (1., .5) if family == "affine" else (.5, .25)
    h1 = gain; h2 = gain*(1+retain); h3 = gain*(1+retain+retain**2)
    for state, value in zip(actual, (h1, h2, h2, h3)):
        equivalent(state.slots["H"][1], h.new_full((1, 1), value))
    expected_short = gain*(1+retain*math.exp(-.3)+retain**2*math.exp(-.4))
    equivalent(actual[-1].slots["H"][0], h.new_full((1, 1), expected_short))
    equivalent(actual[-1].slots["Z"], h.new_tensor([[1+math.exp(-.3)+math.exp(-.4)], [3.]]))
    assert actual[2] is actual[1]


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_joint_graph_ragged_samples_complete_records_and_independent_roots(dtype, family, native):
    b = binding(dtype, family)
    xs = [External(i, 0, j, time, torch.tensor([mass, .2+.1*j, -.4+.05*i], dtype=dtype, requires_grad=True))
          for i, values in enumerate((((0, .2), (1, .4), (4, .5)), ((0, .3), (2, 0.)), ((1, .8), (4, .2))))
          for j, (time, mass) in enumerate(values)]
    q = Continuation(b.graph.identity, 3)
    expected = reference(b.graph, b.model, q, xs, 6, sealed_until=6)
    runner = Native(b.graph, b.model, algorithm="frontier", packed=True).run if native else lambda *a, **kw: frontier(b.graph, b.model, *a, **kw)
    actual = runner(q, xs, 6, sealed_until=6)
    equivalent(expected, actual)
    roots = variables(b, xs)
    pairs = [(objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1])]
    for i in range(3):
        sa, sb = expected.continuation.states[i, 0], actual.continuation.states[i, 0]
        pairs += [(sa.slots[k], sb.slots[k]) for k in sa.slots if sa.slots[k].is_floating_point()]
    for a, c in pairs:
        equivalent(vjp(a, roots), vjp(c, roots))
    assert actual.stats.get("state_scalar_sequence_steps", 0) == 0
    assert actual.stats["max_state_sequence"] > 1
    assert actual.stats["max_state_batch"] > 1
    for cut in (1, 3, 5):
        first = runner(q, [x for x in xs if x.time < cut], cut, sealed_until=cut)
        rest = [x for x in xs if x.time >= cut]
        resumed = runner(first.continuation, rest, 6, sealed_until=6)
        equivalent(actual.continuation, resumed.continuation)
        reference_tail = reference(b.graph, b.model, first.continuation, rest, 6, sealed_until=6)
        equivalent(reference_tail, resumed)
        equivalent(vjp(objective(reference_tail), roots), vjp(objective(resumed), roots))


@pytest.mark.parametrize("family", ["gated_delta", "ssd"])
def test_joint_mixer_independent_dense_convolution_formula(dtype, family):
    from test_weighted_recurrent_formulas import local, dense_reference
    w, m, f = local(dtype, family, width=3, summary_width=2, key_width=2, value_width=3,
                    heads=4, groups=2 if family == "ssd" else 1, conv_size=3, conv_stride=7)
    u = torch.linspace(-.6, .8, 15, dtype=dtype).reshape(5, 3)
    h = torch.cat((torch.ones(5, 1, dtype=dtype), u), -1)
    times = [2+7*i for i in range(5)]
    batch = PackedSequence(h, [0, 5], [(0, 0)], times, [Content(row) for row in h])
    with torch.no_grad():
        states, calls = m.packed_sequence(w, [m.initial(w)], batch)
        expected = dense_reference(m, f, u)
        actual = (torch.stack([s.slots["H"] for s in states]),
                  torch.stack([m.mem(s, t, row)[1] for s, t, row in zip(states, times, u)]),
                  torch.stack([f(m, s, t, row) for s, t, row in zip(states, times, u)]))
    equivalent(expected, actual)
    assert calls == 1


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_no_positive_writes_and_observation_capacity(dtype, family):
    b = binding(dtype, family); w = b.model.nodes[0]; m = w.kernel
    old = m.step(w, m.initial(w), Content(torch.tensor([.3, .2, -.1], dtype=dtype)), 0)
    old = replace(old, value=old.value.detach().requires_grad_(), slots={
        k: v.detach().requires_grad_(v.is_floating_point()) for k, v in old.slots.items()})
    h = torch.tensor([[0., .2, -.1], [0., 4., -3.]], dtype=dtype, requires_grad=True)
    batch = PackedSequence(h, [0, 2], [(0, 0)], [20, 30], [Content(row) for row in h])
    states, calls = m.packed_sequence(w, [old], batch)
    assert calls == 0 and all(s is old for s in states)
    roots = dict(input=h, **dict(m.named_parameters()), old=old.value)
    grads = vjp(states[-1].value, roots)
    assert grads["old"] is not None and all(v is None for k, v in grads.items() if k != "old")
    full = replace(old, observations=2**63-1)
    positive = h.detach().clone(); positive[0, 0] = 1
    overflow = PackedSequence(positive, [0, 2], [(0, 0)], [20, 30], [Content(row) for row in positive])
    with pytest.raises((ValueError, OverflowError), match="overflow|int64"):
        m.packed_sequence(w, [full], overflow)
