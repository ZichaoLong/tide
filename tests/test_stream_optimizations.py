"""Optimized scheduling against independent scalar semantics and legacy native."""
import pytest
import torch
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.reference import run
from cursor_cases import fixture
from region_cases import fixture as history_fixture


FLAGS = [(True, False), (False, True), (True, True), (True, True, True)]


def gradients(result, leaves, root):
    values = torch.autograd.grad(objective(result, root), list(leaves.values()),
                                 allow_unused=True, retain_graph=True)
    return dict(zip(leaves, values))


def execute(make, dtype, mode, flags, *, trace, packed, workers, stop):
    g, m, q, xs, leaves = make(dtype)
    if flags is None:
        return run(g, m, q, xs, stop, sealed_until=stop, mode=mode), leaves
    engine = Native(g, m, workers=workers, packed=packed, mode=mode, trace=trace,
                    parallel_regions=flags[0], compact_events=flags[1],
                    defer_state_release=len(flags) == 3 and flags[2])
    cursor = engine.cursor(q); events, outputs, messages = [], [], []
    for end in sorted({0, 1, 4, stop-1, stop}):
        piece = cursor.advance([x for x in xs if cursor.cut <= x.time < end], end, sealed_until=end)
        events += piece.trace; outputs += piece.outputs; messages += piece.messages
    result = canonicalize(g, Result(cursor.snapshot(), events, outputs, messages, {}))
    if not trace:
        assert not result.trace and not result.messages
    return result, leaves


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("packed,workers,trace", [(False, 1, True), (False, 3, True),
                                                (True, 3, True), (True, 3, False)])
def test_mixed_memories_routes_cuts_and_isolated_vjps(dtype, flags, mode, packed, workers, trace):
    make = lambda dtype: fixture(dtype, selected_only=True)
    kwargs = dict(trace=trace, packed=packed, workers=workers, stop=12)
    expected, evars = execute(make, dtype, mode, None, **kwargs)
    legacy, _ = execute(make, dtype, mode, (False, False), **kwargs)
    actual, avars = execute(make, dtype, mode, flags, **kwargs)
    equivalent(legacy, actual)
    if trace:
        equivalent(expected, actual)
    else:
        equivalent(expected.outputs, actual.outputs)
        equivalent(expected.continuation, actual.continuation)
    for root in ("output", "state", "pending"):
        equivalent(gradients(expected, evars, root), gradients(actual, avars, root))


@pytest.mark.parametrize("policy", ["all", "selected", "clear", "blend"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("trace", [False, True])
def test_tensor_history_and_idle_owners_remain_independent(dtype, policy, mode, trace):
    make = lambda dtype: history_fixture(dtype, policy=policy)
    kwargs = dict(trace=trace, packed=True, workers=3, stop=6)
    expected, evars = execute(make, dtype, mode, None, **kwargs)
    actual, avars = execute(make, dtype, mode, (True, True), **kwargs)
    equivalent(expected.outputs, actual.outputs)
    equivalent(expected.continuation, actual.continuation)
    if trace:
        equivalent(expected, actual)
    for root in ("history", "output", "state"):
        equivalent(gradients(expected, evars, root), gradients(actual, avars, root))


@pytest.mark.parametrize("algorithm", ["frontier", "chain", "self_loop"])
def test_stream_flags_are_not_silently_ignored(dtype, algorithm):
    g, m, *_ = fixture(dtype)
    with pytest.raises(ValueError, match="streaming optimizations"):
        Native(g, m, algorithm=algorithm, parallel_regions=True)
