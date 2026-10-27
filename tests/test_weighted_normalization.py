"""Architecture §1: literal RMS Jacobian and independent scalar differences."""
import math
import pytest
import torch
from tidegraph.weighted.tensors import norm
from tidegraph.compare import equivalent


@pytest.mark.parametrize("epsilon,scale", [(2., 1.), (1e-8, 1e-4)])
def test_literal_normalization_jacobian_and_row_independence(dtype, epsilon, scale):
    x = torch.tensor([[3., 4.], [-2., .5]], dtype=dtype).mul(scale).requires_grad_()
    actual = norm(x, epsilon)
    assert torch.equal(actual, x/torch.sqrt(epsilon+x.square().mean(-1, keepdim=True)))
    for owner in range(2):
        values = x[owner].detach().double().tolist()
        denominator = math.sqrt(epsilon+sum(v*v for v in values)/2)
        for column in range(2):
            expected = torch.zeros_like(x)
            for j in range(2):
                expected[owner, j] = (float(j == column)/denominator-
                                      values[j]*values[column]/(2*denominator**3))
            equivalent(torch.autograd.grad(actual[owner, column], x, retain_graph=True)[0], expected)
    # Explicit zero cotangent remains connected to the input vector.
    equivalent(torch.autograd.grad((actual*0).sum(), x)[0], torch.zeros_like(x))


def test_independent_scalar_directional_difference():
    x = torch.tensor([.2, -.7, .5], dtype=torch.float64, requires_grad=True)
    direction = x.new_tensor([.13, -.2, .15]); bar = x.new_tensor([.3, .4, -.6])
    eps = .17
    gradient, = torch.autograd.grad(norm(x, eps), x, bar)
    def at(sign):
        v = (x.detach()+sign*1e-5*direction).tolist()
        return sum(a*b for a, b in zip(v, bar.tolist()))/math.sqrt(eps+sum(a*a for a in v)/len(v))
    assert float((gradient*direction).sum()) == pytest.approx((at(1)-at(-1))/2e-5, rel=1e-8, abs=1e-10)


def test_normalization_version_and_first_order(dtype):
    x = torch.ones(2, dtype=dtype, requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(norm(x, .1).sum(), x, create_graph=True)
    value = norm(x, .1)
    with torch.no_grad(): x.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        value.sum().backward()
