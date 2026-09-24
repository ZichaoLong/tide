"""Independent rows, upstream owners, undefined cotangents and dense VJP anchor."""
import pytest
import torch
import _tide_native as core
from tidegraph.autograd import value
from tidegraph.compare import equivalent
from tidegraph.ops import emit


@pytest.mark.parametrize('frozen', ['none', 'weight', 'row'])
@pytest.mark.parametrize('root', ['one', 'zero', 'both'])
@pytest.mark.parametrize('upstream', ['plain', 'semantic', 'hst'])
def test_linear_independent_roots_and_upstream(dtype, frozen, root, upstream):
    def run(native):
        w = torch.arange(12, dtype=dtype).reshape(3, 4).div(17).requires_grad_(frozen != 'weight')
        a = torch.linspace(-.3, .8, 4, dtype=dtype, requires_grad=True)
        b = a.detach().add(.2).requires_grad_(frozen != 'row')
        rows = [a * .7, b * .9]
        if upstream == 'semantic':
            rows = [value(x.detach(), x) for x in rows]
        if upstream == 'hst':
            rows = [emit(x, x.sin(), x.sum().sigmoid(), 'hst') for x in rows]
        ys = core.isolated_linear(rows, w) if native else [torch.nn.functional.linear(x, w) for x in rows]
        leaves = [x for x in (a, b, w) if x.requires_grad]
        loss = ys[0].square().sum() * (0 if root == 'zero' else .7)
        if root == 'both':
            loss = loss + ys[1].sum()
        grads = torch.autograd.grad(loss, leaves, allow_unused=True, retain_graph=True)
        again = torch.autograd.grad(loss, leaves, allow_unused=True, retain_graph=True)
        equivalent(grads, again)
        if root != 'both' and b.requires_grad:
            assert grads[1] is None
        return ys, grads
    equivalent(run(False), run(True))


def test_frozen_lane_aliases_and_caller_stack(dtype):
    w = torch.eye(4, dtype=dtype)
    x = torch.randn(4, dtype=dtype, requires_grad=True)
    ys = core.isolated_linear([x, x.detach()], w)
    assert ys[0].requires_grad and not ys[1].requires_grad
    for rows in ([x, x], list(torch.stack([x, x]).unbind())):
        ys = core.isolated_linear(rows, w)
        equivalent(torch.autograd.grad(ys[0].sum()+ys[1].sum(), x)[0], torch.full_like(x, 2))
    with torch.no_grad():
        ys = core.isolated_linear([x, x], w.requires_grad_())
    assert all(not y.requires_grad for y in ys)
    with torch.inference_mode():
        assert not core.isolated_linear([x], w)[0].requires_grad


def test_linear_versions_layout_and_gradcheck(dtype):
    a = torch.randn(4, dtype=dtype, requires_grad=True)
    raw = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    w = raw.t()  # non-contiguous/shared parameter view
    ys = core.isolated_linear([a, a*2], w)
    with torch.no_grad():
        raw.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace'):
        ys[0].sum().backward()
    if dtype == torch.float64:
        w = torch.randn(3, 4, dtype=dtype, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b, w: tuple(core.isolated_linear([a, b], w)), (a, b, w))
        with pytest.raises(RuntimeError, match='first-order'):
            torch.autograd.grad(core.isolated_linear([a], w)[0].sum(), a, create_graph=True)


@pytest.mark.parametrize('kind', ['sgd', 'adamw'])
@pytest.mark.parametrize('zero', [False, True])
def test_update_skips_unused_owner(dtype, kind, zero):
    def run(native):
        owners = [torch.nn.Parameter(torch.full((4,), v, dtype=dtype)) for v in (.2, .4)]
        weight = torch.nn.Parameter(torch.eye(4, dtype=dtype))
        optimizer = (torch.optim.SGD if kind == 'sgd' else torch.optim.AdamW)([*owners, weight], lr=.01, weight_decay=.1)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            xs = [x.sin() for x in owners]
            ys = core.isolated_linear(xs, weight) if native else [torch.nn.functional.linear(x, weight) for x in xs]
            (ys[0].square().sum()*(0 if zero else 1)).backward()
            assert owners[1].grad is None
            assert owners[0].grad is not None
            optimizer.step()
        return owners, weight, optimizer.state_dict()
    equivalent(run(False), run(True))
