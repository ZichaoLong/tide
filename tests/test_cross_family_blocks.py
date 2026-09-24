"""Independent block schedules, prefill gates, transitions and optimizer roots."""
import pytest
import torch
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as scalar
from tidegraph.streaming import run as stream
from tidegraph.specialized_blocks import run as block, settle_layered
from tidegraph.specialized import settle_layered as scalar_layered
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.blocks import canonicalize
from test_specialized_topologies import fixture, layered_fixture, loss_vjp
from foundation_training import trajectory
from isolated_cases import vjp

POLICY = dict(full_autograd="batched", aggregate_autograd="batched")


@pytest.mark.parametrize("memory", ["ema", "ssm", "attention", "linear", "delta", "delta-rule-v1"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", ["diamond", "stream", "settle"])
def test_all_observables_and_cuts(dtype, memory, clear, implementation):
    if implementation == "settle":
        spec, m, q, x = layered_fixture(dtype, memory, clear)
        variables = dict(m.named_parameters()) | {"input": x}
        expected = scalar_layered(spec, m, q, x, mode="hst")
        actual = settle_layered(spec, m, q, x, mode="hst", **POLICY)
        fn = lambda q, a, b: settle_layered(spec, m, q, x[:, a:b], mode="hst", prefill=b <= 3, **POLICY)
        stops, graph = (1, 3, 3, 4), spec.graph
    else:
        graph, m, q, xs, variables = fixture(dtype, "diamond-region", kind=memory, clear=clear)
        expected = scalar(graph, m, q, xs, 9, sealed_until=9, mode="hst")
        def fn(q, a, b):
            arguments = dict(mode="hst", sealed_until=b, **POLICY)
            if implementation == "diamond": arguments.update(topology="diamond", prefill=b <= 6)
            return (block if implementation == "diamond" else stream)(graph, m, q,
                [x for x in xs if q.cut <= x.time < b], b, **arguments)
        actual = fn(q, 0, 9)
        stops = (1, 3, 3, 6, 9)
    equivalent(expected, actual)
    equivalent(loss_vjp(objective(expected), variables), loss_vjp(objective(actual), variables))
    for root in (lambda r: r.outputs[0][-1], lambda r: r.trace[0]["content"]):
        if expected.outputs:
            for zero in (False, True): equivalent(vjp(root(expected), variables, zero), vjp(root(actual), variables, zero))
    events, outputs, messages, start, snapshots = [], [], [], 0, []
    for end in stops:
        part = fn(q, start, end)
        q, start = part.continuation, end
        snapshots.append((q, {k: s.value.detach().clone() for k, s in q.states.items()}))
        events += part.trace; outputs += part.outputs; messages += part.messages
    chunk = canonicalize(graph, Result(q, events, outputs, messages, {}))
    equivalent(expected, chunk)
    equivalent(loss_vjp(objective(expected), variables), loss_vjp(objective(chunk), variables))
    for q, values in snapshots: equivalent(values, {k: s.value for k, s in q.states.items()})
    if implementation == "settle":
        if not clear: assert actual.stats["max_state_sequence"] == 4
        else: assert actual.stats["state_prefill_selected_adoption"] > 0
    assert actual.stats.get("semantic_full_replays", 0) == 0


@pytest.mark.parametrize("family,implementation", [("pdg","python-packed"), ("dag","python-packed"),
    ("dag","python-block"), ("settle","python-packed"), ("settle","python-block")])
@pytest.mark.parametrize("memory", ["ssm", "attention"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_independent_training_trajectories(dtype, family, implementation, memory, optimizer):
    expected = trajectory(dtype, family, "reference", memory, optimizer)
    actual = trajectory(dtype, family, implementation, memory, optimizer, **POLICY)
    equivalent(expected, actual)


def test_unpacked_native_blocks_do_not_replay_full_aggregate(dtype):
    g, m, q, xs, _ = fixture(dtype, "diamond-region", kind="ssm")
    expected = scalar(g, m, q, xs, 9, sealed_until=9)
    for algorithm in ("frontier", "diamond"):
        actual = Native(g, m, packed=False, algorithm=algorithm).run(q, xs, 9, sealed_until=9)
        equivalent(expected, actual)
        assert actual.stats["max_full_batch"] == 1
        assert not actual.stats.get("semantic_full_replays", 0)
        assert not actual.stats.get("semantic_aggregate_replays", 0)


def test_checkpoint_resume_can_change_schedule_and_vjp_policy(dtype, tmp_path):
    from tidegraph.checkpoint import save, load
    g, m, q, xs, _ = fixture(dtype, "diamond-region", kind="attention")
    first = block(g, m, q, [x for x in xs if x.time < 3], 3, sealed_until=3, topology="diamond", **POLICY)
    path = tmp_path/"resume.pt"
    save(path, g, m, first.continuation)
    _, restored, _, _, _ = fixture(dtype, "diamond-region", kind="attention")
    rq = load(path, g, restored)
    suffix = [x for x in xs if x.time >= 3]
    expected = scalar(g, m, first.continuation.detach(), suffix, 9, sealed_until=9)
    actual = Native(g, restored, packed=True, workers=3, algorithm="streaming", **POLICY).run(rq, suffix, 9, sealed_until=9)
    equivalent(expected, actual)
