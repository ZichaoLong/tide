import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from cursor_cases import fixture


@pytest.mark.parametrize("invalid", ["position", "shape", "nan", "seal"])
def test_cursor_invalid_input_is_retryable_without_partial_consumption(dtype, invalid):
    g, m, q, xs, _ = fixture(dtype); engine = Native(g, m, packed=True); cursor = engine.cursor(q)
    before = cursor.snapshot()
    bad = External(0, 1, 9 if invalid == "position" else 0, 0,
                   torch.zeros(5 if invalid == "shape" else 4, dtype=dtype))
    if invalid == "nan":
        bad.value[0] = torch.nan
    with pytest.raises(ValueError):
        cursor.advance([xs[0], bad], 1, sealed_until=0 if invalid == "seal" else 1)
    assert not cursor.failed and cursor.cut == 0
    equivalent(before, cursor.snapshot())
    actual = cursor.advance([xs[0]], 1, sealed_until=1)
    expected = engine.run(q, [xs[0]], 1, sealed_until=1)
    equivalent(expected.outputs, actual.outputs); equivalent(expected.continuation, cursor.snapshot())


def test_cursor_runtime_failure_requires_restore(dtype):
    g = Graph((Node(0),), (Edge(0, 0, 2**63-1),), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    engine = Native(g, m); cursor = engine.cursor(q); saved = cursor.snapshot()
    with pytest.raises(OverflowError, match="overflow"):
        cursor.advance([External(0, 0, 0, 1, torch.ones(3, dtype=dtype))], 2, sealed_until=2)
    assert cursor.failed
    for action in (cursor.snapshot, cursor.detach, lambda: cursor.cut, lambda: cursor.advance([], 2, sealed_until=2)):
        with pytest.raises(RuntimeError, match="restore"):
            action()
    restored = engine.cursor(saved)
    assert not restored.failed
    restored.advance([], 2, sealed_until=2)
    equivalent(engine.run(q, [], 2, sealed_until=2).continuation, restored.snapshot())


def test_cursor_import_is_checked_and_isolated(dtype):
    g, m, q, _, _ = fixture(dtype)
    engine = Native(g, m)
    cursor = engine.cursor(q); saved = cursor.snapshot()
    with torch.no_grad():
        for state in q.states.values():
            state.value.fill_(20)
            for tensor in state.slots.values():
                tensor.fill_(20)
    equivalent(saved, cursor.snapshot())
    q.states[0, 0].last_time = 9
    with pytest.raises(ValueError, match="clock"):
        engine.cursor(q)
    dag = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    with pytest.raises(ValueError, match="streaming"):
        Native(dag, Model(dag, dtype=dtype), algorithm="frontier").cursor(Continuation(dag.identity, 1))
