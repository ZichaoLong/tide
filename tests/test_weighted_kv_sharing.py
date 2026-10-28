"""KV §1.2 / FC §6.2: common base, private entries, complete user VJPs."""
from dataclasses import replace
from types import SimpleNamespace
import math
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from tidegraph.weighted.kv_projection import prepare, projections, key
from isolated_cases import vjp
from test_weighted_compute_sharing import fixture, runner
from test_weighted_execution import variables


@pytest.mark.parametrize("prepared", [False, True])
def test_shared_base_literal_formula_and_all_user_jacobian(dtype, prepared):
    # Scalar textbook example; explicit analytic derivatives, no oracle schedule.
    h = torch.tensor([.4, 2.], dtype=dtype, requires_grad=True)
    base = torch.tensor([[2.], [3.]], dtype=dtype, requires_grad=True)
    a = torch.tensor([[.4], [-.2]], dtype=dtype, requires_grad=True)
    b = torch.tensor([[.5]], dtype=dtype, requires_grad=True)
    summary = torch.tensor([[.7]], dtype=dtype, requires_grad=True)
    config = SimpleNamespace(epsilon=1., width=1)
    memory = SimpleNamespace(config=config, base=base, adapter_u=a, adapter_v=b, summary=summary)
    shared = prepare(memory, h)
    batch = SimpleNamespace(prepared=[shared] if prepared else [])
    plan = SimpleNamespace(ids=[0], content=h[None, 1:])
    values, s = projections(memory, plan, batch)
    n = 2/math.sqrt(5)
    equivalent(values, torch.tensor([[2.2*n, 2.9*n]], dtype=dtype))
    equivalent(s, torch.tensor([[.7*n]], dtype=dtype))
    # Two independent private consumers contribute coefficients 1 and 3.
    loss = values.sum()+3*values.sum()+2*s.sum()
    dh, dw, da, db, ds = torch.autograd.grad(loss, (h, base, a, b, summary))
    equivalent(dh, torch.tensor([0., (4*(2.2+2.9)+2*.7)/(5**1.5)], dtype=dtype))
    equivalent(dw, torch.full_like(base, 4*n))
    equivalent(da, torch.full_like(a, 4*.5*n))
    equivalent(db, torch.full_like(b, 4*(.4-.2)*n))
    equivalent(ds, torch.full_like(summary, 2*n))
    assert key(memory, h) != key(memory, h.detach().clone().requires_grad_())
    assert key(memory, torch.tensor([0., 2.], dtype=dtype)) is None


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_shared_kv_complete_snapshots_and_independent_roots(dtype, native):
    b, xs, stop = fixture(dtype, "region")
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert actual.stats["state_preparation_calls"] > 0
    assert actual.stats["state_preparation_events"] == actual.stats["candidate_events"]
    assert actual.stats["state_preparation_reused_events"] == 2*actual.stats["state_preparation_calls"]
    assert actual.stats["max_state_private_projection_batch"] == 3
    assert actual.stats["state_private_projection_rows"] == 3*actual.stats["candidate_events"]
    roots = variables(b, xs)
    for field in ("keys", "values", "summaries", "weights"):
        a, c = (r.trace[0]["proposal_slots"][field] for r in (expected, actual))
        equivalent(vjp(a, roots), vjp(c, roots))
        equivalent(vjp(a*0, roots), vjp(c*0, roots))
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    cut = 2
    first = runner(b, native)(q, [x for x in xs if x.time < cut], cut, sealed_until=cut)
    tail = runner(b, native)(first.continuation, [x for x in xs if x.time >= cut], stop, sealed_until=stop)
    equivalent(expected.continuation, tail.continuation)
    joined = replace(tail, outputs=first.outputs+tail.outputs)
    equivalent(vjp(objective(expected), roots), vjp(objective(joined), roots))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_private_base_parameter_prevents_computation_sharing(dtype, native):
    b, xs, stop = fixture(dtype, "region")
    for node in b.model.nodes:
        node.kernel.base = torch.nn.Parameter(node.kernel.base.detach().clone())
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert actual.stats.get("state_preparation_events", 0) == 0
    assert actual.stats["aggregate_reused_events"] > 0
    equivalent(vjp(objective(expected), variables(b, xs)), vjp(objective(actual), variables(b, xs)))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("algorithm", ["streaming", "frontier"])
def test_shared_kv_online_schedules_and_inference(dtype, native, algorithm):
    from tidegraph.native import Native
    from tidegraph.streaming import run as streaming
    from tidegraph.frontier import run as frontier
    b, xs, stop = fixture(dtype, "region")
    q = Continuation(b.graph.identity, 2)
    options = dict(packed=True, aggregate_autograd="batched", full_autograd="batched")
    execute = Native(b.graph, b.model, algorithm=algorithm, **options).run if native else (
        lambda *a, **kw: (streaming if algorithm == "streaming" else frontier)(b.graph, b.model, *a, **options, **kw))
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = execute(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert actual.stats["state_preparation_reused_events"] > 0
    equivalent(vjp(objective(expected), variables(b, xs)), vjp(objective(actual), variables(b, xs)))
    with torch.no_grad():
        inference = execute(q, xs, stop, sealed_until=stop)
    equivalent(actual, inference)


@pytest.mark.parametrize("completed", [False, True])
def test_prepared_kv_mixed_rows_raw_zero_and_large_logical_time(dtype, completed):
    from tidegraph.content import Content
    from tidegraph.packing import PackedSequence
    from tidegraph.state_vjp import bind_sequence
    from test_weighted_packed import binding
    b = binding(dtype, "kv")
    w = b.model.nodes[0]; m = w.kernel; clock = 2**54+17
    old = [m.initial(w), m.initial(w)]
    shared = torch.tensor([.4, .2, -.3], dtype=dtype, requires_grad=True)
    zero = torch.tensor([0., 9., 8.], dtype=dtype, requires_grad=True)
    other = torch.tensor([.6, -.5, .7], dtype=dtype, requires_grad=True)
    rows, times = [shared, zero, other, shared], [clock+1, clock+9, clock+12, clock+2]
    common = prepare(m, shared)
    if completed:
        common = m.prepare_private_batch([m, m], common, {})[0]
    batch = PackedSequence(torch.stack(rows), [0, 3, 4], [(0, 0), (1, 0)], times,
                           [Content(h) for h in rows], [common, {}, {}, common])
    expected = []
    for i, state in enumerate(old):
        for j in range(batch.offsets[i], batch.offsets[i+1]):
            state = m.step(w, state, Content(rows[j]), times[j]); expected.append(state)
    with torch.no_grad():
        actual, _ = m.packed_sequence(w, old, batch)
    bind_sequence(w, old, batch, actual)
    equivalent(expected, actual)
    assert actual[1].slots["times"].tolist() == [clock+1]  # Raw expired entry retained by zero receive.
    assert actual[2].slots["times"].tolist() == [clock+12]
    roots = dict(m.named_parameters()) | dict(shared=shared, zero=zero, other=other)
    for i in range(4):
        for name in ("keys", "values", "summaries", "weights"):
            equivalent(vjp(expected[i].slots[name], roots), vjp(actual[i].slots[name], roots))


@pytest.mark.parametrize("mutate", [False, True])
def test_prepared_kv_original_parameter_versions_are_checked(mutate):
    b, xs, stop = fixture(torch.float64, "region")
    result = runner(b, False)(Continuation(b.graph.identity, 2), xs, stop, sealed_until=stop)
    if mutate:
        with torch.no_grad():
            b.model.nodes[0].kernel.base.add_(.1)
    with pytest.raises((ValueError, RuntimeError), match="first-order|modified by an inplace"):
        torch.autograd.grad(objective(result), tuple(variables(b, xs).values()),
                            create_graph=not mutate, allow_unused=True)
