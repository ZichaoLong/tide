"""The probe changes analytic backward arithmetic, never forward or dtype."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from weighted_half_probe import HalfNormalizationProbe


def test_half_probe_matches_independent_double_jacobian_and_restores():
    from tidegraph.weighted.normalization import epsilon_l2, _FP32L2
    old = _FP32L2.backward
    x = torch.tensor([[.7, -.3]], dtype=torch.float16, requires_grad=True)
    bar = torch.tensor([[.2, .8]], dtype=x.dtype)
    public = epsilon_l2(x, 1e-4).detach()
    with HalfNormalizationProbe(True) as probe:
        y = epsilon_l2(x, 1e-4)
        assert torch.equal(y, public)
        grad, = torch.autograd.grad(y, x, bar)
    d = x.detach().double().requires_grad_()
    expected, = torch.autograd.grad(d/(1e-4+d.square().sum(-1, keepdim=True)).sqrt(), d, bar.double())
    assert grad.dtype == x.dtype and torch.equal(grad, expected.half())
    assert probe.counts == {"l2": 1} and _FP32L2.backward is old
