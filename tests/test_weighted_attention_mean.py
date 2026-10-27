"""KV §3/4: literal softmax Jacobian, translation invariance, directional VJP."""
import math
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.attention_mean import attention_mean


@pytest.mark.parametrize("batched", [False, True])
def test_two_entry_literal_and_value_translation(dtype, batched):
    logits = torch.tensor([.2, -.3], dtype=dtype, requires_grad=True)
    values = torch.tensor([[.25, -.5], [.75, .5]], dtype=dtype, requires_grad=True)
    if batched:
        logits = logits.repeat(3, 1).detach().requires_grad_()
        values = values.repeat(3, 1, 1).detach().requires_grad_()
    bar = values.new_tensor([.3, -.7]).expand(*values.shape[:-2], 2)
    p = 1/(1+math.exp(-.5))
    expected_logits = logits.new_tensor([p*(1-p)*.55, -p*(1-p)*.55]).expand_as(logits)
    expected_values = logits.new_tensor([p, 1-p]).unsqueeze(-1)*bar.unsqueeze(-2)
    gradients = []
    for offset in (0., 2**16):
        shifted = (values.detach()+offset).requires_grad_()
        result = attention_mean(logits, shifted)
        # Forward expression/rounding remains the published probability sum.
        assert torch.equal(result, (logits.softmax(-1).unsqueeze(-1)*shifted).sum(-2))
        dl, dv = torch.autograd.grad(result, (logits, shifted), bar)
        equivalent(dl, expected_logits); equivalent(dv, expected_values)
        gradients.append(dl)
    assert torch.equal(*gradients)


@pytest.mark.parametrize("length", [1, 4])
def test_constant_values_have_connected_zero_logit_gradient(dtype, length):
    l = torch.linspace(-.2, .5, length, dtype=dtype, requires_grad=True)
    v = torch.full((length, 3), 713., dtype=dtype, requires_grad=True)
    dl, dv = torch.autograd.grad(attention_mean(l, v).sum(), (l, v))
    assert torch.equal(dl, torch.zeros_like(l))
    equivalent(dv, l.softmax(-1).unsqueeze(-1).expand_as(v))


def test_independent_directional_difference():
    l = torch.tensor([.6, -.4, .2], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[.4, .7], [-.5, .2], [.8, -.6]], dtype=l.dtype, requires_grad=True)
    dl = l.new_tensor([.1, -.2, .3]); dv = v.new_tensor([[.1, .3], [-.3, .2], [.2, -.1]])
    bar = l.new_tensor([.7, -.5])
    a, b = torch.autograd.grad(attention_mean(l, v), (l, v), bar)
    def scalar(sign):
        logits = (l.detach()+sign*1e-5*dl).tolist(); values = (v.detach()+sign*1e-5*dv).tolist()
        probabilities = [math.exp(x) for x in logits]; total = sum(probabilities)
        return sum(p/total*sum(x*y for x, y in zip(row, bar.tolist())) for p, row in zip(probabilities, values))
    assert float((a*dl).sum()+(b*dv).sum()) == pytest.approx((scalar(1)-scalar(-1))/2e-5, rel=1e-8, abs=1e-10)


def test_first_order_and_version_guard(dtype):
    l = torch.zeros(2, dtype=dtype, requires_grad=True); v = torch.ones(2, 1, dtype=dtype, requires_grad=True)
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(attention_mean(l, v).sum(), l, create_graph=True)
    result = attention_mean(l, v)
    with torch.no_grad(): l.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        result.sum().backward()
