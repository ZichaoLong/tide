"""Independent closed forms for the public role-bounded Shannon policy."""
import math
import os

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig, observe_shannon_support
from tide.allocation import allocate
from tide.weighted_mean import stable_weighted_mean


DEVICE = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401

DELTA = 2.**-149


def tensor(value, grad=False):
    return torch.tensor(value, dtype=torch.float32, device=DEVICE, requires_grad=grad)


def solve(scores, w, mu, lower, upper, k):
    return allocate(scores, w, mu, lower, upper, k, kind="shannon", support_policy="positive-floor-v1")


@pytest.mark.parametrize("weak,strong,repairs", [
    (-73.42066192626953, 32.303375244140625, 1), (-71.5, 32., 0), (-72.5, 32., 1),
    (-1000., 32., 1)])
def test_unit_total_preserves_support_and_both_score_roots(weak, strong, repairs):
    s, u, k = tensor([weak, strong], True), tensor([1., 1.], True), tensor(1., True)
    with observe_shannon_support() as counts:
        c = solve(s, u, u, torch.zeros_like(u), u, k)
    assert c.tolist() == [DELTA, 1.] and counts.repaired_coordinates == repairs
    g = tensor([1e30, 0.])
    ds, du, dk = torch.autograd.grad(c, (s, u, k), g)
    p = math.exp(weak - strong) / (1 + math.exp(weak - strong))
    expected = p * (1 - p) * g[0].item()
    torch.testing.assert_close(ds, tensor([expected, -expected]), atol=0, rtol=3e-5)
    torch.testing.assert_close(dk, tensor(p * g[0].item()), atol=0, rtol=3e-5)
    assert torch.equal(du, torch.zeros_like(u))


def test_variable_total_real_floors_caps_structural_zero_and_broadcast():
    # One lower face (.125), one upper face (.25), one structural zero.
    # The two free controls divide the remaining unit, in both batch rows.
    s = tensor([[-74., 35., -90., 60., 0.]], True)
    mu = tensor([1., 1., 1., 1., 0.])
    lo, hi = tensor([0., 0., .125, 0., 0.], True), tensor([1., 1., 1., .25, 0.], True)
    k = tensor([1.375, 1.375], True)
    with observe_shannon_support() as counts:
        c = solve(s, mu.expand(2,-1), mu, lo, hi, k)
    assert counts.repaired_coordinates == 2
    assert c.tolist() == [[DELTA, 1., .125, .25, 0.]] * 2
    g = tensor([[1e30, 0., 0., 0., 0.]]).expand_as(c)
    ds, dl, du, dk = torch.autograd.grad(c, (s, lo, hi, k), g)
    d = math.exp(-109) * g[0, 0].item()
    for got, expected in ((ds, [[2*d, -2*d, 0., 0., 0.]]), (dl, [0., 0., -2*d, 0., 0.]),
                          (du, [0., 0., 0., -2*d, 0.]), (dk, [d, d])):
        torch.testing.assert_close(got, tensor(expected), atol=0, rtol=3e-5)


@pytest.mark.parametrize("lower_active", [True, False])
def test_received_bound_and_slack_have_different_roots(lower_active):
    s, mu = tensor([-110., -1.], True), tensor([1., 1.])
    w = tensor([.5, .25] if lower_active else [.01, .01], True)
    c = solve(s, w, mu, torch.zeros_like(w), mu, 1.)
    assert c[0].item() == DELTA
    g = tensor([1e30, 0.])
    ds, dw = torch.autograd.grad(c, (s, w), g)
    if lower_active:
        a = math.exp(-109) * g[0].item()
        expected = [.75*a, -.75*a]
        torch.testing.assert_close(dw, tensor([a,a]), atol=0, rtol=3e-5)
    else:
        expected = [math.exp(-111) * g[0].item(), 0.]
        assert torch.count_nonzero(dw) == 0
    torch.testing.assert_close(ds, tensor(expected), atol=0, rtol=3e-5)


def test_true_zero_budget_refusal_and_multiple_stored_tail_consumers():
    s, mu = tensor([-110., -109., 32., 100.], True), tensor([1.,1.,1.,0.])
    with observe_shannon_support() as counts:
        c = solve(s, mu, mu, torch.zeros_like(mu), mu, 1.)
    assert counts.repaired_coordinates == 2 and c.tolist() == [DELTA, DELTA, 1., 0.]
    v = tensor([[1.],[3.]], True)
    y = stable_weighted_mean(c[:2].detach(), v)
    assert y.item() == 2.
    assert torch.autograd.grad(y.sum(), v)[0].tolist() == [[.5],[.5]]
    with pytest.raises(ArithmeticError, match="relative budget"):
        solve(tensor([-110.,-130.]), tensor([DELTA, DELTA]), tensor([1.,1.]),
              tensor([0.,0.]), tensor([1.,1.]), 1.)
    forced = solve(tensor([100.,-200.]), tensor([1.,1.]), tensor([1.,1.]),
                   tensor([1.,0.]), tensor([1.,1.]), 1.)
    assert forced.tolist() == [1.,0.]


def test_policy_identity_and_strict_default():
    default = ModelConfig(allocation="shannon")
    assert "shannon_support_policy" not in default.to_dict()
    assert ModelConfig.from_dict(default.to_dict()) == default
    floor = ModelConfig(allocation="shannon", shannon_support_policy="positive-floor-v1")
    assert ModelConfig.from_dict(floor.to_dict()) == floor
    with pytest.raises(ValueError, match="requires Shannon"):
        ModelConfig(allocation="quad", shannon_support_policy="positive-floor-v1")
    s, mu = tensor([-110.,32.]), tensor([1.,1.])
    with pytest.raises(ArithmeticError, match="underflow"):
        allocate(s, mu, mu, torch.zeros_like(mu), mu, 1., kind="shannon")


def test_mixed_rows_keep_ordinary_forward_and_backward():
    s = tensor([[3.25, 2.5, -.5], [32., -110., 0.]], True)
    mu = torch.ones_like(s)
    root = tensor([[4., -3., .5], [0., 1e30, 0.]])
    with observe_shannon_support() as counts:
        c = solve(s, mu, mu, torch.zeros_like(s), mu, 1.)
    ds, = torch.autograd.grad(c, s, root)
    ordinary = s[:1].detach().requires_grad_()
    u = torch.ones_like(ordinary)
    ref = allocate(ordinary, u, u, torch.zeros_like(u), u, 1., kind="shannon")
    expected, = torch.autograd.grad(ref, ordinary, root[:1])
    assert counts.repaired_rows == counts.refined_rows == 1
    assert torch.equal(c[:1], ref) and torch.equal(ds[:1], expected)


def test_saturated_peak_preserves_score_gaps_between_remaining_candidates():
    s = tensor([1e30, 32., 31., -110.], True)
    mu = torch.ones_like(s)
    hi = tensor([.25, 1., 1., 1.], True)
    with observe_shannon_support() as counts:
        c = solve(s, mu, mu, torch.zeros_like(mu), hi, 1.)
    p = 1 / (1 + math.exp(-1))
    torch.testing.assert_close(c, tensor([.25, .75*p, .75*(1-p), DELTA]), atol=0, rtol=2e-6)
    assert counts.repaired_coordinates == 1
    ds, du = torch.autograd.grad(c[1], (s, hi))
    torch.testing.assert_close(ds[:3], tensor([0., .75*p*(1-p), -.75*p*(1-p)]), atol=0, rtol=2e-6)
    torch.testing.assert_close(du, tensor([-p, 0., 0., 0.]), atol=0, rtol=2e-6)


def test_exact_cap_tie_on_slack_refined_row():
    s = tensor([1., -110.], True)
    hi = tensor([1., 1.], True)
    mu = torch.ones_like(s)
    c = solve(s, mu, mu, torch.zeros_like(s), hi, 2.)
    ds, du = torch.autograd.grad(c[0], (s, hi))
    assert c.tolist() == [1., DELTA]
    assert ds.tolist() == [0., 0.] and du.tolist() == [1., 0.]


def test_cpu_fp64_tail_survives_only_after_combining_with_its_root():
    s = torch.tensor([-748., 3.], dtype=torch.float64, requires_grad=True)
    mu = torch.ones_like(s)
    with observe_shannon_support() as counts:
        c = solve(s, mu, mu, torch.zeros_like(mu), mu, 1.)
    assert c.tolist() == [math.ulp(0.), 1.] and counts.repaired_coordinates == 1
    g = s.new_tensor([1e300, 0.])
    ds, = torch.autograd.grad(c, s, g)
    expected = math.exp(-751 + math.log(1e300))
    torch.testing.assert_close(ds, s.new_tensor([expected, -expected]), atol=0, rtol=2e-13)


def test_stored_tail_in_attention_and_continued_kv():
    from tide.attention import weighted_attention
    from tide.kv import KVMemory

    s = tensor([-110., -109., 32.])
    mu = torch.ones_like(s)
    mass = solve(s, mu, mu, torch.zeros_like(mu), mu, 1.)[:2]
    # Both inputs have the same *stored* mass. Normalized attention splits
    # its value gradient evenly. KV also retains the logical time of each write.
    q, k = tensor([[[[0.]]]]), tensor([[[[0.], [0.]]]])
    v = tensor([[[[1.], [3.]]]], True)
    y = weighted_attention(q, k, v, weights=mass, backend="math")
    assert y.item() == 2.
    assert torch.autograd.grad(y.sum(), v)[0].flatten().tolist() == [.5, .5]
    kv = KVMemory(1, 64, None, 4, 1, backend="math").to(DEVICE)
    u = torch.zeros(1, 1, 1, 64, device=DEVICE)
    owners = torch.tensor([0], device=DEVICE)
    state = None
    for i, m in enumerate(mass, 1):
        times = torch.tensor([[[i]]], device=DEVICE)
        description, context, state = kv.prepare(u, m.reshape(1, 1, 1), times, owners, state)
        assert description[-1].item() == i * DELTA
        assert bool(torch.isfinite(kv.finish(context, torch.ones_like(m).reshape(1, 1, 1))).all())
    assert state.weight.flatten().tolist() == [DELTA, DELTA] and state.times.flatten().tolist() == [1, 2]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_propagates_policy_to_parallel_lattice_regions(batch_size):
    torch.manual_seed(17)
    config = ModelConfig(topology=TopologyConfig(kind="lattice", n=2, depth=1), d=64,
                         allocation="shannon", shannon_support_policy="positive-floor-v1",
                         region_batch_size=batch_size, attention_backend="math")
    model = TideModel(config).to(DEVICE)
    # The middle layer has two compatible two-candidate regions. Scores are
    # controlled by ordinary model parameters in both execution paths.
    with torch.no_grad():
        model.describe.readout.value.zero_()
        for network in model.regional:
            network.readout.zero_()
        for region in model.layer_regions[1]:
            weak, strong = model.graph.regions[region]
            model.describe.offset.value[model.description_ids[weak]].fill_(-110)
            model.describe.offset.value[model.description_ids[strong]].fill_(32)
    x = torch.randn(1, 2, 64, device=DEVICE, requires_grad=True)
    with observe_shannon_support() as counts:
        out = model(x)
        out.readout().square().mean().backward()
    assert counts.repaired_coordinates > 0 and bool(torch.isfinite(x.grad).all())
