import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.lh_full import PROFILES
from tidegraph.ops import Model


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("mode", ["hard", "softp", "hst"])
def test_lh_full_value_and_independent_analytic_vjp(dtype, profile, mode):
    g = Graph((Node(0, full=profile),), (), (Region(1),), (0,), (0,))
    w = Model(g, width=3, dtype=dtype).nodes[0]
    act, norm = PROFILES[profile]
    with torch.no_grad():
        if norm != "identity":
            w.extra["lh_norm_weight"].copy_(torch.tensor([.7, 1.1, 1.4], dtype=dtype))
        if norm == "layer":
            w.extra["lh_norm_bias"].fill_(.13)
    s = torch.tensor([.3, -.7, 1.2], dtype=dtype, requires_grad=True)
    h = torch.tensor([-.2, .5, .7], dtype=dtype, requires_grad=True)
    p = torch.tensor(.3, dtype=dtype, requires_grad=True)
    a = s.detach().relu() if act == "relu" else s.detach()*s.detach().sigmoid() if act == "silu" else s.detach()
    da = (s.detach() > 0).to(dtype) if act == "relu" else (
        s.detach().sigmoid()*(1+s.detach()*(1-s.detach().sigmoid())) if act == "silu" else torch.ones_like(s))
    gamma = torch.ones_like(a) if norm == "identity" else w.extra["lh_norm_weight"].detach()
    if norm == "rms":
        inv = (a.square().mean()+1e-7).rsqrt(); fresh = gamma*a*inv
        derivative = gamma*inv-a*(gamma*a).mean()*inv**3
    elif norm == "layer":
        centered = a-a.mean(); inv = (centered.square().mean()+1e-5).rsqrt()
        fresh = gamma*centered*inv+.13
        derivative = inv*(gamma-gamma.mean()-centered*(gamma*centered).mean()*inv**2)
    else:
        fresh, derivative = a, torch.ones_like(a)
    actual = w.full(s, h, p, mode, 1.7)
    expected = h.detach()+p.detach()*(fresh-h.detach()) if mode == "softp" else fresh
    equivalent(actual, expected)
    ds, dh, dp, unused = torch.autograd.grad(actual.sum(), (s, h, p, w.weight), allow_unused=True)
    equivalent(ds, derivative*da*(.3 if mode == "softp" else 1))
    equivalent(dh, None if mode == "hard" else torch.full_like(h, .7 if mode == "softp" else 0))
    equivalent(dp, None if mode == "hard" else (fresh-h.detach()).sum()*(1.7 if mode == "hst" else 1))
    assert unused is None


@pytest.mark.parametrize("profile", PROFILES)
def test_lh_full_zero_variance_and_parameter_validation(dtype, profile):
    from tidegraph.full import validate_program
    from tidegraph.native import Native
    node = Node(0, full=profile); g = Graph((node,), (), (Region(1),), (0,), (0,))
    m = Model(g, width=3, dtype=dtype); w = m.nodes[0]
    s = torch.zeros(3, dtype=dtype, requires_grad=True)
    value = w.fresh(s, torch.ones_like(s))
    assert torch.equal(value, torch.zeros_like(s))
    assert torch.isfinite(torch.autograd.grad(value.sum(), s)[0]).all()
    if "lh_norm_weight" in w.extra:
        w.extra["lh_norm_weight"] = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        with pytest.raises(ValueError, match="normalization parameter"):
            validate_program(w, node, 1)
        with pytest.raises(ValueError, match="normalization parameter"):
            Native(g, m)
