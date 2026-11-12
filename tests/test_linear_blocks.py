"""Whole node operands: literal matrix Jacobians, root support and versions."""
import os
import pytest
import torch
from tidegraph.runtime import resolve_device
from tidegraph.weighted.row_linear import linear_blocks


def operator(client):
    if client == "python":
        return linear_blocks
    from tidegraph.native_loader import load_native
    return load_native(os.environ.get("TIDE_BUILD_DIR")).weighted_linear_blocks


def material(dtype):
    device, _ = resolve_device(os.environ.get("TIDE_COLUMN_DEVICE", "cpu"))
    x = torch.linspace(-.4, .8, 15, dtype=dtype, device=device).reshape(5, 3)
    w = torch.tensor([[.2, -.5, .9], [-.3, .7, .4]], dtype=dtype, device=device)
    return x, w


@pytest.mark.parametrize("client", ["python", "native"])
@pytest.mark.parametrize("zero", [False, True])
def test_literal_block_jacobians_and_unused_equal_parameters(dtype, client, zero):
    x, w = material(dtype)
    xs = [x.clone().requires_grad_() for _ in range(3)]
    ws = [w.clone().requires_grad_() for _ in range(3)]
    ys = operator(client)(xs, ws)
    bars = [torch.linspace(-.7+i, .6+i, 10, dtype=dtype, device=x.device).reshape(5, 2) for i in range(3)]
    for y in ys:
        torch.testing.assert_close(y, x@w.t())
    factor = 0. if zero else 1.
    loss = factor*((ys[0]*bars[0]).sum()+(ys[2]*bars[2]).sum())
    roots = [t for row in zip(xs, ws) for t in row]
    gradients = torch.autograd.grad(loss, roots, allow_unused=True)
    assert gradients[2:4] == (None, None)
    for i in (0, 2):
        torch.testing.assert_close(gradients[2*i], factor*(bars[i]@w))
        torch.testing.assert_close(gradients[2*i+1], factor*(bars[i].t()@x))


@pytest.mark.parametrize("client", ["python", "native"])
def test_shared_input_matrix_and_frozen_output(dtype, client):
    x, w = (t.requires_grad_() for t in material(dtype))
    # The first two uses share exact roots; the last has identical values but
    # no gradient roots. A shared GEMM must still collect both live uses.
    ys = operator(client)([x, x, x.detach()], [w, w, w.detach()])
    assert not ys[-1].requires_grad
    gx, gw = torch.autograd.grad(ys[0].sum()+3*ys[1].sum(), (x, w))
    torch.testing.assert_close(gx, 4*torch.ones_like(ys[0])@w)
    torch.testing.assert_close(gw, 4*torch.ones_like(ys[0]).t()@x)
    ys = operator(client)([x, x], [w, w])
    gx, gw = torch.autograd.grad(ys[0].sum()+3*ys[1].sum(), (x, w))
    torch.testing.assert_close(gx, 4*torch.ones_like(ys[0])@w)
    torch.testing.assert_close(gw, 4*torch.ones_like(ys[0]).t()@x)


@pytest.mark.parametrize("client", ["python", "native"])
def test_original_versions_first_order_and_inference(dtype, client):
    fn = operator(client)
    x, w = (t.requires_grad_() for t in material(dtype))
    unused = w.detach().clone().requires_grad_()
    ys = fn([x, x], [w, unused])
    with torch.no_grad():
        unused.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        ys[0].sum().backward()
    ys = fn([x, x], [w, w])
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(ys[0].sum(), x, create_graph=True)
    with torch.inference_mode():
        ys = fn([x.clone(), x.clone()], [w.clone(), w.clone()])
        assert all(not y.requires_grad for y in ys)
        torch.testing.assert_close(ys[0], x@w.t())
    with pytest.raises((ValueError, RuntimeError), match="metadata|shape"):
        fn([x, x[:2]], [w, w])
