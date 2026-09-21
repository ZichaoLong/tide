from collections import defaultdict
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.fiber_attention import PROFILE
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from isolated_cases import vjp


def case(dtype, cache, ragged):
    g = Graph((Node(0, memory=PROFILE, query_heads=2, kv_heads=2),), (), (Region(1),), (0, 0, 0), (0,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 4); leaves = dict(m.named_parameters())
    lengths = ((2, 1, 3), (2, 2), (1, 2, 3)) if ragged else ((2, 1, 3),)*3
    xs, positions = [], defaultdict(int)
    for b, sizes in enumerate(lengths):
        old = m.nodes[0].initial()
        old.slots = {"key": torch.full((cache, 2, 2), .1+b/20, dtype=dtype, requires_grad=True),
                     "value": torch.full((cache, 2, 2), .3+b/10, dtype=dtype, requires_grad=True),
                     "log_bias": torch.full((cache,), -.13-b/20, dtype=dtype, requires_grad=True)}
        q.states[b, 0] = old
        leaves.update({f"initial.{b}.{name}": value for name, value in old.slots.items()})
        for j, size in enumerate(sizes):
            for p in range(size):
                position = positions[b, p]; positions[b, p] += 1
                x = torch.sin(torch.arange(4, dtype=dtype)*.17+(b+p+j+1)/10).requires_grad_()
                leaves[f"input.{b}.{p}.{position}"] = x
                xs.append(External(b, p, position, (2, 5, 9)[j], x))
    return g, m, q, xs, leaves, lengths


def execute(implementation, g, m, q, xs, prefill=True):
    if implementation == "python":
        return frontier(g, m, q, xs, 12, sealed_until=12, prefill=prefill)
    return Native(g, m, algorithm="frontier", workers=3, packed=implementation == "native-packed",
                  prefill=prefill).run(q, xs, 12, sealed_until=12)


@pytest.mark.parametrize("cache", [0, 2])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("implementation", ["python", "native-serial", "native-packed"])
def test_fiber_actual_batch_sequence_work_and_compact_persistent_storage(dtype, cache, ragged, implementation):
    g, m, q, xs, _, lengths = case(dtype, cache, ragged)
    with torch.inference_mode():
        expected = run(g, m, q, xs, 12, sealed_until=12)
        actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    assert (3, 0) not in actual.continuation.states  # No padding-induced event.
    assert actual.stats.get("semantic_state_replays", 0) == 0
    assert actual.stats.get("state_scalar_sequence_steps", 0) == 0
    calls = 3 if implementation == "native-serial" else 2 if ragged else 1
    assert actual.stats["state_sequence_calls"] == calls
    assert actual.stats["state_blocks"] == 3 and actual.stats["state_steps"] == 0
    if implementation.startswith("native"):
        assert actual.stats["max_state_batch"] == (1 if implementation == "native-serial" else 2 if ragged else 3)
        assert actual.stats["max_state_sequence"] == 3  # Events, not projected source rows.
        assert actual.stats["attention_score_elements"] == sum(2*sum(s)*(cache+sum(s)) for s in lengths)
    for a, b in zip(expected.trace, actual.trace):
        assert torch.equal(a["proposal_slots"]["log_bias"], b["proposal_slots"]["log_bias"])
    for state in actual.continuation.states.values():
        for t in (state.value, *state.slots.values()):
            assert t.untyped_storage().nbytes() == t.numel()*t.element_size()


@pytest.mark.parametrize("cache", [0, 2])
@pytest.mark.parametrize("implementation", ["python", "native-packed"])
def test_fiber_packing_preserves_first_output_absence_and_all_cache_vjps(dtype, cache, implementation):
    g, m, q, xs, leaves, _ = case(dtype, cache, True)
    expected = run(g, m, q, xs, 12, sealed_until=12)
    actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    eroot, aroot = expected.outputs[0][3], actual.outputs[0][3]
    eg, ag = vjp(eroot, leaves), vjp(aroot, leaves); equivalent(eg, ag)
    assert ag["input.0.0.1"] is None and ag["input.1.0.0"] is None
    assert ag["input_scale.2"] is None  # Absent here, present in a later fiber.
    if cache == 0:
        assert ag["nodes.0.extra.fiber_decay"] is None  # No first-event decay path.
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    for name in actual.continuation.states[0, 0].slots:
        equivalent(vjp(expected.continuation.states[0, 0].slots[name], leaves),
                   vjp(actual.continuation.states[0, 0].slots[name], leaves))


@pytest.mark.parametrize("implementation", ["python", "native-packed"])
def test_fiber_prefill_disabled_keeps_causal_batch_equivalent(dtype, implementation):
    g, m, q, xs, leaves, _ = case(dtype, 2, True)
    expected = run(g, m, q, xs, 12, sealed_until=12)
    actual = execute(implementation, g, m, q, xs, prefill=False)
    equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    assert actual.stats.get("state_sequence_calls", 0) == 0
    assert actual.stats["state_steps"] > 0
