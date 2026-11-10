"""Batched guards reject late bad rows without publishing an input transaction."""
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.finite import all_finite
from tidegraph.ops import Model
from tidegraph.native import Native
from tidegraph.greedy import run


def test_finite_guard_checks_every_chunk_dtype_and_noncontiguous_tail():
    a = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2].requires_grad_()
    b = torch.ones(9, dtype=torch.float64, requires_grad=True)
    assert all_finite([a, b], budget=20)
    broken = b.detach().clone(); broken[-1] = float("nan")
    assert not all_finite([a, broken], budget=20)
    assert a.grad is None and b.grad is None
    with pytest.raises(ValueError, match="budget"):
        all_finite([a], budget=0)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_rejected_last_input_keeps_original_ledger_retryable(dtype, native):
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=2, dtype=dtype)
    q = Continuation(g.identity, 1)
    xs = [External(0, 0, i, i, torch.ones(2, dtype=dtype, requires_grad=True)) for i in range(17)]
    execute = Native(g, m, algorithm="window", packed=True).run if native else lambda *a, **k: run(g, m, *a, component_window=True, **k)
    with torch.no_grad(): xs[-1].value[1] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        execute(q, xs, 17, sealed_until=17)
    assert q.cut == 0 and not q.ledger and not q.states and not q.pending
    with torch.no_grad(): xs[-1].value[1] = 1.
    result = execute(q, xs, 17, sealed_until=17)
    assert result.continuation.ledger[0, 0] == (16, 16)
    result.outputs[-1][-1].sum().backward()
    assert xs[-1].value.grad is not None and torch.isfinite(xs[-1].value.grad).all()
