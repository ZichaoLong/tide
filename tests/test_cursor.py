from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.blocks import canonicalize
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.reference import run
from cursor_cases import fixture


def vjp(result, variables):
    return dict(zip(variables, torch.autograd.grad(objective(result), list(variables.values()), allow_unused=True)))


@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("selected_only", [False, True])
def test_cursor_window_trace_and_all_memory_vjps(dtype, mode, packed, selected_only):
    g, m, q, xs, variables = fixture(dtype, selected_only)
    expected = run(g, m, q, xs, 12, sealed_until=12, mode=mode); grad = vjp(expected, variables)
    g, m, q, xs, variables = fixture(dtype, selected_only)
    cursor = Native(g, m, workers=3 if packed else 1, packed=packed, mode=mode).cursor(q)
    trace, outputs, messages = [], [], []
    for stop in (0, 1, 4, 8, 12):
        inputs = [a for a in xs if cursor.cut <= a.time < stop]
        piece = cursor.advance(inputs, stop, sealed_until=stop)
        assert piece.cut == cursor.cut == stop and not hasattr(piece, "continuation")
        assert piece.stats["external_records"] == len(inputs)
        trace += piece.trace; outputs += piece.outputs; messages += piece.messages
    actual = canonicalize(g, Result(cursor.snapshot(), trace, outputs, messages, {}))
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))


def test_cursor_each_cut_snapshot_isolated_and_checkpoint(dtype, tmp_path):
    g, m, q, xs, _ = fixture(dtype)
    engine = Native(g, m, workers=3, packed=True); cursor = engine.cursor(q)
    for stop in (2, 6, 12):
        inputs = [a for a in xs if q.cut <= a.time < stop]
        expected = engine.run(q, inputs, stop, sealed_until=stop)
        actual = cursor.advance(inputs, stop, sealed_until=stop)
        snapshot = cursor.snapshot()
        equivalent(expected, Result(snapshot, actual.trace, actual.outputs, actual.messages, actual.stats))
        q = expected.continuation
        with torch.no_grad():
            for state in snapshot.states.values():
                state.value.fill_(99)
                for tensor in state.slots.values():
                    tensor.fill_(99)
            for atom in snapshot.pending:
                atom.value.fill_(99)
        equivalent(q, cursor.snapshot())
    path = tmp_path / "cursor.pt"; save(path, g, m, cursor.snapshot())
    restored = engine.cursor(load(path, g, m))
    a = cursor.advance([], 15, sealed_until=15); b = restored.advance([], 15, sealed_until=15)
    equivalent(a, b); equivalent(cursor.snapshot(), restored.snapshot())


def test_cursor_explicit_detach_cuts_pending_and_all_slots(dtype):
    g, m, q, xs, variables = fixture(dtype)
    cursor = Native(g, m, packed=True).cursor(q)
    cursor.advance([a for a in xs if a.time < 4], 4, sealed_until=4)
    before = cursor.snapshot(); assert before.pending
    cursor.detach(); after = cursor.snapshot(); equivalent(before, after)
    assert all(not t.requires_grad for s in after.states.values() for t in (s.value, *s.slots.values()))
    assert all(not a.value.requires_grad for a in after.pending)
    suffix = cursor.advance([], 8, sealed_until=8)
    result = Result(cursor.snapshot(), suffix.trace, suffix.outputs, suffix.messages, suffix.stats)
    grad, = torch.autograd.grad(objective(result), (variables["input"],), allow_unused=True)
    assert grad is None


def test_multiple_cursors_share_engine_without_pool_races(dtype):
    g, m, q, xs, _ = fixture(dtype); engine = Native(g, m, workers=3, packed=True)
    cursors = [engine.cursor(q), engine.cursor(q)]
    def advance(cursor):
        with torch.inference_mode():
            result = cursor.advance(xs, 12, sealed_until=12)
            return Result(cursor.snapshot(), result.trace, result.outputs, result.messages, result.stats)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(advance, cursors))
    with torch.inference_mode():
        expected = engine.run(q, xs, 12, sealed_until=12)
    for actual in results:
        equivalent(expected, actual)
        assert all(torch.is_inference(x) for _, _, _, x in actual.outputs)


def test_trace_free_cursor_only_materializes_requested_state(dtype):
    g, m, q, xs, _ = fixture(dtype)
    cursor = Native(g, m, trace=False, packed=True).cursor(q)
    piece = cursor.advance(xs, 12, sealed_until=12)
    assert not piece.trace and not piece.messages
    expected = run(g, m, q, xs, 12, sealed_until=12)
    equivalent(expected.outputs, piece.outputs); equivalent(expected.continuation, cursor.snapshot())
