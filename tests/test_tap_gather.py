"""Convolution's ordered transpose: independent incidence matrix and derivatives."""
import pytest
import torch
from tidegraph.weighted.tap_gather import tap_gather


def incidence(ids, n):
    # Literal linear map, including absent slots; no gather/scatter oracle.
    matrix = torch.zeros(ids.numel(), n, dtype=torch.float64)
    for row, source in enumerate(ids.flatten().tolist()):
        if source < n:
            matrix[row, source] = 1
    return matrix


@pytest.mark.parametrize("empty", [False, True])
def test_literal_linear_map_and_directional_derivative(empty):
    ids = torch.tensor([[6, 6, 6], [0, 6, 6], [2, 1, 0], [4, 3, 2]])
    if empty:
        ids = ids[:0]
    x = torch.randn(6, 2, 3, dtype=torch.float64, requires_grad=True)
    bar = torch.randn(*ids.shape, 2, 3, dtype=x.dtype)
    matrix = incidence(ids, len(x))
    actual = tap_gather(x, ids)
    expected = (matrix@x.flatten(1)).reshape(actual.shape)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for scale in (1., 0.):
        grad, = torch.autograd.grad(actual, x, bar*scale, retain_graph=True)
        literal = (matrix.t()@bar.flatten(0, 1).flatten(1)*scale).reshape(x.shape)
        torch.testing.assert_close(grad, literal)
    direction = torch.randn_like(x)
    h = 1e-5
    numerical = ((tap_gather(x.detach()+h*direction, ids)-tap_gather(x.detach()-h*direction, ids))*bar).sum()/(2*h)
    grad, = torch.autograd.grad(actual, x, bar)
    torch.testing.assert_close(numerical, (grad*direction).sum(), atol=2e-9, rtol=2e-9)


def test_disconnected_zero_version_and_first_order_contract():
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    unused = torch.ones_like(x, requires_grad=True)
    y = tap_gather(x, torch.tensor([[2, 2], [0, 2], [1, 0]]))
    a, b = torch.autograd.grad(y.sum()*0, (x, unused), allow_unused=True, retain_graph=True)
    assert b is None and torch.equal(a, torch.zeros_like(x))
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(y.sum(), x, create_graph=True)
    y = tap_gather(x, torch.tensor([[0, 2]]))
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


def test_invalid_layout_and_empty_bank():
    x = torch.ones(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="twice"):
        tap_gather(x, torch.tensor([[0, 2], [0, 1]]))
    for ids in (torch.tensor([[-1]]), torch.tensor([[3]]), torch.tensor([[0.]])):
        with pytest.raises(ValueError):
            tap_gather(x, ids)
    x = torch.empty(0, 3, requires_grad=True)
    y = tap_gather(x, torch.zeros(2, 2, dtype=torch.int64))
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.autograd.grad(y.sum(), x)[0].shape == x.shape
