"""Formula and finite-difference anchors independent of the graph scheduler."""
import pytest
import torch
import _tide_native as core
from tidegraph.compare import equivalent


def formula(xs, scales, coe, sources, mean):
    result = []
    for start in range(0, len(xs), sources):
        values = [xs[k]*scales[k] for k in range(start, start+sources)]
        row = coe[start//sources] if coe.ndim == 2 else coe
        values = [v/sources if mean else v*row[j] if coe.numel() else v for j, v in enumerate(values)]
        total = values[0]
        for value in values[1:]: total = total+value
        result += [total, *values]
    return result


@pytest.mark.parametrize('kind', ['sum', 'mean', 'weighted'])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('frozen', [False, True])
def test_independent_rows_roots_zero_and_shared_owners(dtype, kind, shared, frozen):
    def execute(native):
        xs = [(torch.arange(5, dtype=dtype)/7+i/13).requires_grad_(not frozen or i > 1) for i in range(6)]
        scales = [torch.tensor(.3+i/11, dtype=dtype, requires_grad=not frozen or i > 1) for i in range(6)]
        if shared: xs[3], scales[3] = xs[2], scales[2]
        coe = torch.tensor([.4, .6] if kind == 'weighted' else [], dtype=dtype, requires_grad=kind == 'weighted' and not frozen)
        outputs = (core.isolated_aggregate if native else formula)(xs, scales, coe, 2, kind == 'mean')
        variables = {id(v):v for v in [*xs, *scales, coe] if v.requires_grad}
        grads = []
        for i in range(len(outputs)):
            if not outputs[i].requires_grad: continue
            for zero in (False, True):
                root = outputs[i].square().sum()*(0 if zero else 1)
                grads.append(torch.autograd.grad(root, list(variables.values()), allow_unused=True, retain_graph=True))
        root = sum(v.sum() for v in outputs if v.requires_grad)
        grads.append(torch.autograd.grad(root, list(variables.values()), allow_unused=True))
        return tuple(v.requires_grad for v in outputs), outputs, grads
    equivalent(execute(False), execute(True))


def test_finite_differences_first_order_only():
    xs = [torch.randn(3, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    scales = [torch.randn((), dtype=torch.float64, requires_grad=True) for _ in range(4)]
    coe = torch.randn(2, dtype=torch.float64, requires_grad=True)
    def fn(*args): return tuple(core.isolated_aggregate(args[:4], args[4:8], args[8], 2, False))
    assert torch.autograd.gradcheck(fn, (*xs, *scales, coe))
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(fn(*xs, *scales, coe)[0].sum(), xs[0], create_graph=True)


def test_single_source_contribution_and_summary_both_receive_cotangents(dtype):
    x = torch.ones(3, dtype=dtype, requires_grad=True)
    s = torch.tensor(2., dtype=dtype, requires_grad=True)
    y, c = core.isolated_aggregate([x], [s], torch.empty(0, dtype=dtype), 1, False)
    dx, ds = torch.autograd.grad(y.sum()+3*c.sum(), (x, s))
    equivalent(dx, x.detach()*8); equivalent(ds, s.detach()*0+12)


def test_per_event_coefficients_finite_difference_and_isolation():
    xs = [torch.randn(3, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    scales = [torch.randn((), dtype=torch.float64, requires_grad=True) for _ in range(4)]
    coe = torch.randn(2,2, dtype=torch.float64, requires_grad=True)
    def fn(*args): return tuple(core.isolated_aggregate(args[:4], args[4:8], args[8], 2, False))
    assert torch.autograd.gradcheck(fn, (*xs, *scales, coe))
    a = formula(xs, scales, coe, 2, False)[1].sum()
    b = fn(*xs, *scales, coe)[1].sum()
    equivalent(torch.autograd.grad(a, (*xs, *scales, coe), allow_unused=True),
               torch.autograd.grad(b, (*xs, *scales, coe), allow_unused=True))
