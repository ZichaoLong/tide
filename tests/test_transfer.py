"""Packed copies retain isolated roots, aliases and the optimizer's missing grads."""
import os
import pytest
import torch
import _tide_native as core
from tidegraph.transfer import copy_rows
from tidegraph.ops import emit
from tidegraph.autograd import value
from tidegraph.compare import equivalent


@pytest.fixture
def transport_devices():
    backend = os.environ.get("TIDE_TRANSFER_BACKEND", "cpu")
    if backend == "cpu":
        return "cpu", "cpu"
    if backend == "npu":
        import torch_npu
        assert torch.npu.device_count() >= 2
    elif backend == "cuda":
        assert torch.cuda.device_count() >= 2
    else:
        raise ValueError("unknown transport test backend")
    return backend+":0", backend+":1"


@pytest.fixture(params=["python", "native"])
def copier(request):
    return copy_rows if request.param == "python" else core.copy_rows


@pytest.mark.parametrize("root", ["one", "zero", "both"])
@pytest.mark.parametrize("upstream", ["plain", "semantic", "hst"])
def test_isolated_roots(dtype, transport_devices, copier, root, upstream):
    first, target = transport_devices
    def run(packed):
        a = torch.linspace(-.3, .8, 8, dtype=dtype, device=first).requires_grad_()
        b = a.detach().add(.2).requires_grad_()
        rows = [a[::2]*.7, b[::2]*.9]
        if upstream == "semantic":
            rows = [value(x.detach(), x) for x in rows]
        if upstream == "hst":
            rows = [emit(x, x.sin(), x.sum().sigmoid(), "hst") for x in rows]
        ys = copier(rows, target) if packed else [r.to(target) for r in rows]
        loss = ys[0].square().sum()*(0 if root == "zero" else .7)
        if root == "both":
            loss = loss+ys[1].sum()
        grads = torch.autograd.grad(loss, (a, b), allow_unused=True, retain_graph=True)
        equivalent(grads, torch.autograd.grad(loss, (a,b), allow_unused=True))
        if root != "both":
            assert grads[1] is None
        assert grads[0] is not None
        return [y.detach().cpu() for y in ys], [g.cpu() if g is not None else None for g in grads]
    equivalent(run(False), run(True))


def test_frozen_aliases_noncontiguous_and_boundaries(dtype, transport_devices, copier):
    first, target = transport_devices
    x = torch.arange(12, dtype=dtype, device=first).requires_grad_()
    rows = [x[::3], x[::3], x[1::3].detach(), x[2::3]]
    # Two full packs and a partial group, a single-row fallback and an oversized row.
    for budget in (1, 4*x.element_size(), 8*x.element_size(), 12*x.element_size()):
        ys = copier(rows, target, budget)
        assert not ys[2].requires_grad and ys[0].requires_grad
        grad = torch.autograd.grad(ys[0].sum()+ys[1].sum()+ys[3].sum(), x)[0]
        expected = torch.tensor([2.,0.,1.]*4, dtype=dtype, device=first)
        torch.testing.assert_close(grad, expected)
        for a,b in zip(ys,rows):
            torch.testing.assert_close(a.cpu(),b.cpu())
    for mode in (torch.no_grad(), torch.inference_mode()):
        with mode:
            assert all(not y.requires_grad for y in copier(rows, target))
    assert copier([],target) == []


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_optimizer_skips_unused_owner(dtype, transport_devices, copier, zero, optimizer):
    first, target = transport_devices
    def run(packed):
        parameters = [torch.nn.Parameter(torch.full((4,),v,dtype=dtype,device=first)) for v in (.2,.4)]
        factory = torch.optim.SGD if optimizer == "sgd" else torch.optim.AdamW
        opt = factory(parameters,lr=.01,weight_decay=.1,foreach=False)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            rows = [p.sin() for p in parameters]
            ys = copier(rows,target) if packed else [v.to(target) for v in rows]
            (ys[0].square().sum()*(0 if zero else 1)).backward()
            assert parameters[0].grad is not None and parameters[1].grad is None
            opt.step()
        return parameters,opt.state_dict()
    equivalent(run(False),run(True))


def test_copy_metadata_refusals(copier):
    for rows,budget in [([torch.ones(3),torch.ones(4)],8),([torch.ones(2,dtype=torch.long)],8),
                        ([torch.ones(0)],8),([torch.ones(2,2)],8),([torch.ones(2)],0)]:
        with pytest.raises((ValueError,RuntimeError),match="transfer"):
            copier(rows,"cpu",budget)


def test_higher_order_copy_anchor(copier):
    # Linear transport can retain higher-order input/cotangent connectivity; it
    # does not broaden the independent first-order HST/module VJP contracts.
    a,b = [torch.randn(3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    fn = lambda a,b: tuple(copier([a,b],"cpu"))
    assert torch.autograd.gradcheck(fn,(a,b))
    assert torch.autograd.gradgradcheck(fn,(a,b))
