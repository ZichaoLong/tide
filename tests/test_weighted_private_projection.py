"""KV §1.2: real private matrix batches, independent roots, aliases and versions."""
from types import SimpleNamespace
import pytest
import torch
from tidegraph.weighted.row_linear import linear_rows
from tidegraph.weighted.kv_projection_batch import prepare


def test_private_linear_literal_jacobian_and_absent_rows(dtype):
    xs = [torch.tensor([2., -1.], dtype=dtype, requires_grad=True) for _ in range(3)]
    ws = [torch.tensor([[3., 4.], [-2., 5.]], dtype=dtype, requires_grad=True) for _ in range(3)]
    ys = linear_rows(xs, ws)
    for y in ys:
        assert torch.equal(y, y.new_tensor([2., -9.]))
    # Only row 1 is rooted. Equal-valued independent rows stay disconnected.
    roots = [t for row in zip(xs, ws) for t in row]
    bars = torch.autograd.grad(ys[1], roots, ys[1].new_tensor([.5, -2.]), allow_unused=True, retain_graph=True)
    for i in (0, 1, 4, 5):
        assert bars[i] is None
    assert torch.equal(bars[2], xs[1].new_tensor([5.5, -8.]))
    assert torch.equal(bars[3], ws[1].new_tensor([[1., -.5], [-4., 2.]]))
    zero = torch.autograd.grad(ys[1]*0, roots, torch.ones_like(ys[1]), allow_unused=True)
    assert zero[0] is None and zero[1] is None and zero[4] is None and zero[5] is None
    assert torch.equal(zero[2], torch.zeros_like(xs[1])) and torch.equal(zero[3], torch.zeros_like(ws[1]))


def test_private_linear_shared_parameter_and_mixed_frozen_rows(dtype):
    x = torch.tensor([2., -1.], dtype=dtype, requires_grad=True)
    w = torch.tensor([[3., 4.]], dtype=dtype, requires_grad=True)
    ys = linear_rows([x, x, x.detach()], [w, w, w.detach()])
    assert not ys[2].requires_grad
    dx, dw = torch.autograd.grad(ys[0]+3*ys[1], (x, w))
    assert torch.equal(dx, x.new_tensor([12., 16.]))
    assert torch.equal(dw, w.new_tensor([[8., -4.]]))


@pytest.mark.parametrize("mutate", [False, True])
def test_private_linear_versions_and_first_order(mutate):
    x = torch.ones(2, requires_grad=True);w = torch.ones(1, 2, requires_grad=True)
    y = linear_rows([x, x], [w, w])[0]
    if mutate:
        with torch.no_grad():
            w.add_(1)
    with pytest.raises((RuntimeError, ValueError), match="inplace|first-order"):
        torch.autograd.grad(y, (x, w), create_graph=not mutate)


def test_private_preparation_groups_shapes_without_merging_parameters(dtype):
    n = torch.tensor([2.], dtype=dtype, requires_grad=True)
    base = torch.tensor([3., 5.], dtype=dtype, requires_grad=True)
    def memory(rank):
        return SimpleNamespace(adapter_v=torch.full((rank, 1), .5, dtype=dtype, requires_grad=True),
            adapter_u=torch.full((2, rank), .25, dtype=dtype, requires_grad=True),
            summary=torch.full((1, 1), .75, dtype=dtype, requires_grad=True))
    ms = [memory(1), memory(2), memory(1)]
    stats = {};shared = {"normal": n, "base": base}
    results = prepare(ms, shared, stats)
    assert results[1] is shared  # Different rank singleton keeps its legal path.
    for i in (0, 2):
        assert torch.equal(results[i]["values"], base.new_tensor([3.25, 5.25]))
        assert torch.equal(results[i]["summaries"], n.new_tensor([1.5]))
    assert stats == dict(state_private_projection_calls=3, state_private_projection_rows=6,
                         max_state_private_projection_batch=2)
    roots = (n, base, ms[0].adapter_u, ms[0].adapter_v, ms[0].summary,
             ms[2].adapter_u, ms[2].adapter_v, ms[2].summary)
    bars = torch.autograd.grad(results[0]["values"].sum(), roots, allow_unused=True)
    for i in (4, 5, 6, 7):
        assert bars[i] is None
    for actual, expected in zip(bars[:4], ([.25], [1., 1.], [[1.], [1.]], [[1.]])):
        assert torch.equal(actual, actual.new_tensor(expected))
