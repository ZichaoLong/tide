"""Shared numerical work retains every use, its roots and original versions."""
import pytest
import torch
from tidegraph.weighted.column_sharing import shared_uses


def test_actual_alias_only_and_literal_sum_of_distinct_use_losses():
    mass = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    content = torch.tensor([[.2, .5], [.7, -.1]], dtype=torch.float64, requires_grad=True)
    value = torch.arange(2), mass, content
    independent = value[0], mass.clone(), content.clone()
    actual = shared_uses({3: value, 1: value, 7: independent, 8: None})
    assert actual[7] is independent and actual[8] is None
    assert actual[3][2].data_ptr() == actual[1][2].data_ptr() == content.data_ptr()
    a, b = actual[3], actual[1]
    loss = (a[1]*a[2].sum(1)).sum()+(b[1].square()+b[2].square().sum(1)).sum()
    gm, gu = torch.autograd.grad(loss, (mass, content), retain_graph=True)
    torch.testing.assert_close(gm, content.sum(1)+2*mass, atol=0, rtol=0)
    torch.testing.assert_close(gu, mass[:, None]+2*content, atol=0, rtol=0)
    # A disconnected field stays None; an observed zero remains connected.
    gm, gu = torch.autograd.grad(a[1].sum()*0, (mass, content), retain_graph=True, allow_unused=True)
    assert torch.equal(gm, torch.zeros_like(mass)) and gu is None


def test_stable_use_order_and_shared_original_version_guard():
    x = torch.ones(1, requires_grad=True)
    value = torch.arange(1), x, torch.zeros(1, 1)
    outputs = shared_uses({v: value for v in range(3)})
    loss = outputs[2][1].sum()+outputs[1][1].sum()*-1e8+outputs[0][1].sum()*1e8
    assert torch.equal(torch.autograd.grad(loss, x, retain_graph=True)[0], torch.ones_like(x))
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(loss, x)


def test_frozen_fields_no_grad_and_first_order_contract():
    x = torch.ones(2, requires_grad=True)
    value = torch.arange(2), x, torch.ones(2, 1)
    with torch.no_grad():
        assert shared_uses({1: value, 2: value})[1] is value
    outputs = shared_uses({1: value, 2: value})
    assert not outputs[1][2].requires_grad
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(outputs[1][1].sum(), x, create_graph=True)
