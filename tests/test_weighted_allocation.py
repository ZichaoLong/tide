"""0770593 allocation §2.5, §5.2, §6.2: independent active-set/KKT anchors.

Enumeration solves every face of the small feasible polytope; it shares no
breakpoint ordering or forward solver with the candidate implementation.
"""
from itertools import product
import pytest
import torch
from tidegraph.weighted import allocate_quad


def enumerate_quad(t, mu, lower, upper, tau):
    candidates = []
    for labels in product((0, 1, 2), repeat=len(t)):
        free = [i for i, label in enumerate(labels) if label == 1]
        capped = [i for i, label in enumerate(labels) if label == 2]
        for total in (None, lower, upper):
            if total is None:
                lam = t.new_zeros(())
            elif free:
                lam = (t[free].sum()+tau*(mu[capped].sum()-total))/len(free)
            else:
                continue
            c = torch.stack([t[i]*0 if label == 0 else (t[i]-lam)/tau if label == 1 else mu[i]
                             for i, label in enumerate(labels)])
            tolerance = 1e-5 if t.dtype == torch.float32 else 1e-10
            if ((c >= -tolerance).all() and (c <= mu+tolerance).all()
                    and c.sum() >= lower-tolerance and c.sum() <= upper+tolerance):
                candidates.append(c)
    return max(candidates, key=lambda c: float((t*c-tau*c.square()/2).sum().detach()))


def close(a, b):
    torch.testing.assert_close(a, b, atol=2e-6 if a.dtype == torch.float32 else 1e-11,
                               rtol=2e-5 if a.dtype == torch.float32 else 1e-9)


def test_quad_textbook_total_and_distinct_threshold_derivatives(dtype):
    t = torch.tensor([2., 1., 100.], dtype=dtype, requires_grad=True)
    mu = torch.tensor([1., 1., 0.], dtype=dtype, requires_grad=True)
    total = torch.tensor(1.4, dtype=dtype, requires_grad=True)
    result = allocate_quad(t, mu, total, total, 2.)
    close(result.controls, t.new_tensor([.95, .45, 0.]))
    close(torch.autograd.grad(result.controls[0], total)[0], total.new_tensor(.5))
    # §5.2/6.2: identical forward, different fixed-total / slack-threshold maps.
    s = torch.tensor([2., 1.], dtype=dtype, requires_grad=True)
    eta = torch.tensor(.8, dtype=dtype, requires_grad=True)
    total = torch.tensor(1.4, dtype=dtype, requires_grad=True)
    fixed = allocate_quad(s, torch.ones_like(s), total, total, .5).controls
    slack = allocate_quad(s-eta, torch.ones_like(s), 1., 2., .5).controls
    close(fixed, s.new_tensor([1., .4])); close(slack, fixed)
    close(torch.autograd.grad(fixed[1], total, retain_graph=True)[0], total.new_tensor(1.))
    close(torch.autograd.grad(fixed[1], s)[0], torch.zeros_like(s))
    close(torch.autograd.grad(slack[1], eta, retain_graph=True)[0], eta.new_tensor(-2.))
    close(torch.autograd.grad(slack[1], s)[0], s.new_tensor([0., 2.]))


def test_quad_enumerated_optima_and_chain_vjp(dtype):
    gen = torch.Generator().manual_seed(827)
    for case in range(24):
        t = (torch.randn(4, generator=gen, dtype=dtype)*1.4).requires_grad_()
        mu = (torch.rand(4, generator=gen, dtype=dtype)*.7+.2).requires_grad_()
        tau = torch.tensor(.73, dtype=dtype, requires_grad=True)
        low = mu.sum()*.24
        high = low if case % 3 == 0 else mu.sum()*.79
        actual = allocate_quad(t, mu, low, high, tau)
        expected = enumerate_quad(t, mu, low, high, tau)
        close(actual.controls, expected)
        root = t.new_tensor([.3, -.7, 1.2, .6])
        ga = torch.autograd.grad((actual.controls*root).sum(), (t, mu, tau), retain_graph=True)
        ge = torch.autograd.grad((expected*root).sum(), (t, mu, tau), allow_unused=True)
        for a, b in zip(ga, ge):
            close(a, torch.zeros_like(a) if b is None else b)
        shifted = allocate_quad(t+13, mu, low, low, tau).controls
        close(shifted, allocate_quad(t, mu, low, low, tau).controls)


def test_quad_temperature_uses_response_point_and_directional_derivative(dtype):
    t = torch.tensor([.6, .4], dtype=dtype, requires_grad=True)
    tau = torch.tensor(1., dtype=dtype, requires_grad=True)
    c = allocate_quad(t, torch.ones_like(t), 1., 1., tau).controls
    close(torch.autograd.grad(c[0], tau)[0], tau.new_tensor(-.1))
    if dtype == torch.float64:
        args = (t.detach().requires_grad_(), t.new_tensor([.9, .8], requires_grad=True),
                t.new_tensor(.7, requires_grad=True), t.new_tensor(.9, requires_grad=True))
        assert torch.autograd.gradcheck(lambda s, m, b, temp: allocate_quad(s, m, b, b, temp).controls,
                                       args, eps=1e-6, atol=1e-5)


def test_quad_singleton_ties_saturated_and_exact_zero(dtype):
    t = torch.tensor([0., 0., 0.], dtype=dtype, requires_grad=True)
    mu = torch.tensor([.2, .4, 0.], dtype=dtype, requires_grad=True)
    cap = allocate_quad(t, mu, mu.sum(), 2.)
    assert cap.constraint == "caps"
    assert torch.autograd.grad(cap.controls.sum(), t, allow_unused=True)[0] is None
    close(allocate_quad(t, mu, .3, .3).controls, t.new_tensor([.15, .15, 0.]))
    zeros = allocate_quad(t, mu, 0., 0.)
    assert not zeros.controls.requires_grad and zeros.constraint == "zero"
    saturated = allocate_quad(t.new_tensor([5., -3., 0.]), mu, .2, .2)
    close(saturated.controls, t.new_tensor([.2, 0., 0.]))
    close(allocate_quad(t[:1], mu[:1], .1, .1).controls, t.new_tensor([.1]))


@pytest.mark.parametrize("bad", ["caps", "feasibility", "temperature", "nan"])
def test_quad_rejects_invalid_domain(dtype, bad):
    t, mu = torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype)
    lo, hi, tau = 1., 2., 1.
    if bad == "caps": mu[0] = -1
    if bad == "feasibility": lo = 3
    if bad == "temperature": tau = 0
    if bad == "nan": t[0] = float("nan")
    with pytest.raises(ValueError):
        allocate_quad(t, mu, lo, hi, tau)
