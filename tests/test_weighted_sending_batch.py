"""Allocation §4.2–4.4: positive support, ID ties, row totals and branch VJPs."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted import sending, sending_batch as batch
from isolated_cases import vjp


def implementation(native):
    if not native:
        return batch
    import _tide_native as core
    from types import SimpleNamespace
    def sorted_send(t, caps, nodes, *, target=None, k=None):
        return core.weighted_sorted_batch(t, caps, nodes, t.new_zeros(len(t)) if target is None else target, 0 if k is None else k)
    return SimpleNamespace(attention=core.weighted_attention_batch, proportional=core.weighted_proportional_batch, sorted_send=sorted_send)


def fixture(dtype):
    scores = torch.tensor([[.2, .2, -1., 5.], [-1., -2., -1., 0.], [0., 0., 0., 0.], [2., 1., -.2, 0.]], dtype=dtype, requires_grad=True)
    caps = torch.tensor([[.4, .7, .5, 0.], [0., .2, .8, 0.], [0., 0., 0., 0.], [.8, .7, .3, .5]], dtype=dtype, requires_grad=True)
    received = (caps.detach()*.5).requires_grad_()
    temperature = torch.tensor([.8, .9, .6, 1.1], dtype=dtype, requires_grad=True)
    nodes = torch.tensor([[3, 1, 7, 9], [2, 4, 6, 8], [7, 5, 3, 1], [8, 6, 2, 0]], dtype=torch.int64)
    return scores, caps, received, temperature, nodes


@pytest.mark.parametrize("mode,base", [("direct_attention", "mu"), ("direct_attention", "received"),
    ("unit_attention", "mu"), ("unit_attention", "uniform"), ("pooled_attention", "mu")])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_attention_batches_and_zero_base_right_derivatives(dtype, mode, base, native):
    t, caps, recv, tau, _ = fixture(dtype)
    expected = torch.stack([sending.attention_send(t[i], caps[i], recv[i], mode=mode, base=base, temperature=tau[i]) for i in range(len(t))])
    actual = implementation(native).attention(t, caps, recv, mode, base, tau)
    equivalent(expected, actual); equivalent(expected > 0, actual > 0)
    roots = dict(scores=t, caps=caps, received=recv, temperature=tau)
    for i in range(len(t)):
        equivalent(vjp(expected[i], roots), vjp(actual[i], roots))
    equivalent(vjp(expected, roots), vjp(actual, roots))
    equivalent(vjp(expected, roots, zero=True), vjp(actual, roots, zero=True))


@pytest.mark.parametrize("mode", ["total", "hard", "proportional"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_stable_sending_batches_and_fractional_support(dtype, mode, native):
    t, caps, _, _, nodes = fixture(dtype)
    candidate = implementation(native)
    target = (caps.detach().sum(1)*.6).requires_grad_()
    if mode == "proportional":
        expected = torch.stack([sending.proportional(row, 1) for row in caps])
        actual = candidate.proportional(caps, 1)
    elif mode == "hard":
        expected = torch.stack([sending.hard_selection(t[i], caps[i], 2, nodes[i].tolist()) for i in range(len(t))])
        actual = candidate.sorted_send(t, caps, nodes, k=2)
    else:
        expected = torch.stack([sending.total_zero(t[i], caps[i], target[i], nodes[i].tolist()) for i in range(len(t))])
        actual = candidate.sorted_send(t, caps, nodes, target=target)
    equivalent(expected, actual); equivalent(expected > 0, actual > 0)
    roots = dict(scores=t, caps=caps, target=target)
    for i in range(len(t)):
        equivalent(vjp(expected[i], roots), vjp(actual[i], roots))
    equivalent(vjp(expected, roots), vjp(actual, roots))
    equivalent(vjp(expected, roots, zero=True), vjp(actual, roots, zero=True))


def test_batched_sending_rejects_erased_positive_support():
    t = torch.tensor([[0., -1000.], [2., 3.]])
    with pytest.raises(ArithmeticError, match="support underflowed"):
        batch.shares(t, torch.ones_like(t), torch.ones(2))
