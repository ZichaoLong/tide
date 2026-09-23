"""Actual frontier/Settle transport and independent-owner waves, with VJP roots."""
from dataclasses import replace
import pytest
import torch
import _tide_native as core
from tidegraph import Continuation, Edge, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.native_records import to_continuation
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from fiber_cases import fixture
from fiber_pool_cases import profile
from isolated_cases import vjp
from next_cases import blend_fixture
from test_native_settle import projected
from test_specialized_topologies import loss_vjp


ALL = dict(packed_sources=True, batch_next=True, parallel_regions=True,
           compact_events=True, defer_state_release=True)
FLAGS = [dict(packed_sources=True), dict(batch_next=True), dict(parallel_regions=True),
         dict(compact_events=True), dict(compact_events=True, defer_state_release=True), ALL]


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("policy", ["all", "selected", "clear", "old"])
@pytest.mark.parametrize("trace", [False, True])
def test_frontier_single_switches_and_combination(dtype, flags, policy, trace):
    g, m, q, xs, leaves = fixture(dtype, policy, profile=profile("all-softmax"))
    xs = [replace(x, time=2*x.position) for x in xs]
    saved = {owner: {k: v.detach().clone() for k, v in state.slots.items()} for owner, state in q.states.items()}
    expected = run(g, m, q, xs, 7, sealed_until=7, mode="hst")
    engine = Native(g, m, algorithm="frontier", workers=3, packed=True, mode="hst", trace=trace,
                    attention_packing="single", fiber_pooling="csr", fiber_cache="owned", attention_layout="head", **flags)
    actual = engine.run(q, xs, 7, sealed_until=7)
    equivalent(expected.outputs, actual.outputs); equivalent(expected.continuation, actual.continuation)
    if trace: equivalent(expected, actual)
    else: assert not actual.trace and not actual.messages
    for root in ("output", "state", "pending"):
        equivalent(loss_vjp(objective(expected, root), leaves), loss_vjp(objective(actual, root), leaves))
    equivalent(saved, {owner: state.slots for owner, state in q.states.items()})
    if flags.get("packed_sources"): assert actual.stats["packed_source_batches"] > 0
    if flags.get("batch_next"):
        assert actual.stats["max_next_batch"] >= 2
        assert actual.stats["semantic_next_replays"] == actual.stats["next_steps"]
        if policy == "clear": assert actual.stats["next_reset_batches"] > 0
    if flags.get("parallel_regions"): assert actual.stats["max_region_wave"] >= 2
    if flags.get("compact_events"): assert actual.stats["consumed_fibers"] == actual.stats["candidate_events"]
    if flags.get("defer_state_release") and not trace: assert actual.stats["deferred_state_releases"] > 0
    with torch.no_grad():
        inference = engine.run(q, xs, 7, sealed_until=7)
    equivalent(actual.outputs, inference.outputs); equivalent(actual.continuation, inference.continuation)
    assert inference.stats.get("semantic_next_replays", 0) == 0


@pytest.mark.parametrize("clear", [False, True])
def test_causal_next_fallback_is_counted(dtype, clear):
    g, m, q, xs, leaves = blend_fixture(dtype, clear=clear)
    expected = run(g, m, q, xs, 5, sealed_until=5)
    actual = Native(g, m, algorithm="frontier", packed=True, workers=3, **ALL).run(q, xs, 5, sealed_until=5)
    equivalent(expected, actual)
    equivalent(loss_vjp(objective(expected), leaves), loss_vjp(objective(actual), leaves))
    assert actual.stats["next_scalar_fallback_steps"] == actual.stats["next_steps"]
    assert actual.stats["state_prefill_fallback_events"] == actual.stats["candidate_events"]
    assert actual.stats["state_prefill_blocked_next"] > 0
    assert actual.stats.get("next_batches", 0) == 0


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta", "delta-rule-v1", profile("all-softmax")])
@pytest.mark.parametrize("clear", [False, True])
def test_cpp_settle_frontend_uses_transport_and_actual_prefill(dtype, kind, clear):
    graph = Graph(tuple(Node(r, memory=kind, clear=clear, query_heads=2, kv_heads=2,
                            window=3 if kind == "attention" else 0) for r in (0, 0, 1)),
                  (Edge(0, 2, 1), Edge(1, 2, 1)), (Region(1), Region(1)), (0, 1), (2,))
    spec = SettleGraph(graph, (1, 2)); m = Model(graph, width=4, dtype=dtype)
    x = (torch.arange(32, dtype=dtype).reshape(2, 4, 4) / 100).requires_grad_()
    q = Continuation(graph.identity, 2)
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode="hst")
    body = Native(graph, m)
    compiled = core.SettleGraph(body.compiled, spec.ranks)
    options = core.Options(); options.packed = True; options.workers = 3; options.mode = "hst"
    for name, value in ALL.items(): setattr(options, name, value)
    engine = core.SettleExecutor(compiled, body.weights, options)
    eq = compiled.embed_initial(to_continuation(core, graph, body.compiled, q))
    encoded = engine.run(eq, x); actual = projected(spec, compiled, encoded)
    equivalent(expected, actual)
    for root in ("output", "state"):
        equivalent(loss_vjp(objective(expected, root), leaves), loss_vjp(objective(actual, root), leaves))
    assert actual.stats["max_next_batch"] == 2 and actual.stats["packed_source_batches"] > 0
    if not clear:
        assert actual.stats["max_state_sequence"] == 4 and actual.stats["state_steps"] == 0
    else:
        assert actual.stats["state_prefill_selected_clear"] > 0


@pytest.mark.parametrize("algorithm", ["chain", "self_loop", "ring", "diamond"])
def test_independent_native_schedules_reuse_block_options(dtype, algorithm):
    from test_specialized_topologies import fixture as topology_fixture
    from test_specialized import fixture as old_fixture
    if algorithm in {"chain", "self_loop"}:
        g, m, q, xs, x, initial = old_fixture(dtype, algorithm, True)
        leaves = dict(m.named_parameters()) | {"input": x, "initial": initial}
    else:
        g, m, q, xs, leaves = topology_fixture(dtype, "diamond-region" if algorithm == "diamond" else "ring")
    expected = run(g, m, q, xs, 9, sealed_until=9, mode="hst")
    actual = Native(g, m, algorithm=algorithm, workers=3, packed=True, mode="hst", **ALL).run(q, xs, 9, sealed_until=9)
    equivalent(expected, actual)
    equivalent(loss_vjp(objective(expected), leaves), loss_vjp(objective(actual), leaves))
    assert actual.stats["next_batches"] > 0 and actual.stats["packed_source_batches"] > 0


def test_frontier_chunk_cut_pending_and_snapshot_lifetime(dtype):
    g, m, q, xs, leaves = fixture(dtype, profile=profile("all-softmax"))
    engine = Native(g, m, algorithm="frontier", workers=3, packed=True, mode="hst", **ALL)
    expected = run(g, m, q, xs, 9, sealed_until=9, mode="hst")
    events, outputs, messages, snapshots = [], [], [], []
    for stop in (3, 3, 6, 9):
        part = engine.run(q, [x for x in xs if q.cut <= x.time < stop], stop, sealed_until=stop)
        snapshots.append((part.continuation, {k: s.value.detach().clone() for k, s in part.continuation.states.items()}))
        q = part.continuation; events += part.trace; outputs += part.outputs; messages += part.messages
    from tidegraph.blocks import canonicalize
    actual = canonicalize(g, Result(q, events, outputs, messages, {}))
    equivalent(expected, actual)
    equivalent(loss_vjp(objective(expected), leaves), loss_vjp(objective(actual), leaves))
    for snapshot, values in snapshots:
        equivalent(values, {k: s.value for k, s in snapshot.states.items()})


@pytest.mark.parametrize("flags", [dict(attention_packing="single"), dict(fiber_pooling="csr"),
                                  dict(fiber_cache="owned"), dict(attention_layout="head")])
def test_policy_for_wrong_module_is_explicitly_rejected(dtype, flags):
    g = Graph((Node(0, memory="attention"),), (), (Region(1),), (0,), (0,))
    with pytest.raises(ValueError, match="same-fiber"):
        Native(g, Model(g, dtype=dtype), algorithm="frontier", **flags)


@pytest.mark.parametrize("algorithm", ["streaming", "frontier"])
@pytest.mark.parametrize("packed", [False, True])
def test_nondefault_policy_scalar_path_is_reported(dtype, algorithm, packed):
    g, m, q, xs, _ = fixture(dtype, "clear")
    engine = Native(g, m, algorithm=algorithm, packed=packed, attention_packing="single")
    result = engine.run(q, xs, 8, sealed_until=8)
    equivalent(run(g, m, q, xs, 8, sealed_until=8), result)
    counter = "fiber_policy_semantic_replays" if algorithm == "streaming" and packed else "fiber_policy_scalar_events"
    assert result.stats[counter] == result.stats["candidate_events"]
