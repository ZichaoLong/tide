"""Allocation §6/S.2: local affine faces amid distant saturated scores.

Literal solutions and VJPs do not call another allocation implementation.
The captured D2048 row is preserved verbatim, including its rejected controls.
"""
import json
from pathlib import Path
import pytest
import torch
from test_weighted_quad_offset import evaluate


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
@pytest.mark.parametrize("gap", [1e4, 1e8, 1e38])
def test_quad_wide_scores_single_free(dtype, backend, gap):
    s = torch.tensor([gap, 0.], dtype=dtype, requires_grad=True)
    mu = torch.ones_like(s, requires_grad=True)
    bound, tau, eta = [s.new_tensor(x, requires_grad=True) for x in (1.7, 1., 0.)]
    c = evaluate(backend, s, mu, bound, tau, eta)
    expected = torch.stack((mu[0].detach(), bound.detach()-mu[0].detach()))
    torch.testing.assert_close(c, expected, atol=4*torch.finfo(dtype).eps, rtol=0)
    bars = torch.autograd.grad(c, (s, mu, bound, tau, eta), s.new_tensor([2., -1.]))
    for actual, wanted in zip(bars, ([0., 0.], [3., 0.], -1., 0., 0.)):
        torch.testing.assert_close(actual, s.new_tensor(wanted), atol=4*torch.finfo(dtype).eps, rtol=0)


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
def test_quad_wide_scores_two_free_and_zero_cap(dtype, backend):
    s = torch.tensor([1e12, 1., 1.5, -1e12, 2e12], dtype=dtype, requires_grad=True)
    mu = s.new_tensor([.3, .8, .9, .5, 0.], requires_grad=True)
    bound, tau, eta = [s.new_tensor(x, requires_grad=True) for x in (1.3, 1., 0.)]
    c = evaluate(backend, s, mu, bound, tau, eta)
    remaining = bound.detach()-mu[0].detach()
    expected = torch.stack((mu[0].detach(), (remaining-.5)/2, (remaining+.5)/2, s.new_tensor(0.), s.new_tensor(0.)))
    tolerance = 8*torch.finfo(dtype).eps
    torch.testing.assert_close(c, expected, atol=tolerance, rtol=0)
    bars = torch.autograd.grad(c, (s, mu, bound, tau, eta), s.new_tensor([9., 2., 5., 7., 4.]))
    for actual, wanted in zip(bars, ([0., -1.5, 1.5, 0., 0.], [5.5, 0., 0., 0., 0.], 3.5, -.75, 0.)):
        torch.testing.assert_close(actual, s.new_tensor(wanted), atol=tolerance, rtol=0)


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
@pytest.mark.parametrize("material", ["wide2048", "matched2048"])
def test_quad_actual_d2048_failure_closed_form(backend, material):
    import math
    data = json.loads((Path(__file__).parent/f"data/quad-{material}-failure.json").read_text())
    tensor = lambda key: torch.tensor(data["tensors"][key]["values"], dtype=torch.float32)
    s, mu = tensor("scores"), tensor("caps")
    bound, tau, eta = tensor("lower")[0], tensor("temperature")[0], s.new_tensor(0.)
    c = evaluate(backend, s, mu, bound, tau, eta)
    expected = mu.clone()
    if material == "matched2048":
        expected[14] = 0
    expected[1] = 0
    expected[1] = float(bound)-math.fsum(expected.tolist())
    torch.testing.assert_close(c, expected, atol=8*torch.finfo(s.dtype).eps, rtol=0)
    assert abs(float(tensor("controls").sum()-bound)) > float(tensor("tolerance")[0])
