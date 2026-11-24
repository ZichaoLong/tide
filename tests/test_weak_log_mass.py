"""Independent positive log cotangents, real zero roots and state transport."""
import math
import os

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from tide.attention import weighted_attention
from tide.checkpoint import save_checkpoint, load_checkpoint
from tide.region import region_control
from tide.weak_mass import POLICY, reception_mean


@pytest.fixture(scope="module")
def device():
    name = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
    if name.startswith("npu"):
        import torch_npu
    return torch.device(name)


def test_prefloor_sender_score_gradient_composes_with_read(device):
    scores = torch.tensor([-82.26013, 27.21748], device=device).view(2,1,1).requires_grad_()
    controls = region_control(scores, torch.ones_like(scores), torch.zeros(2, dtype=torch.bool, device=device),
                              1, "shannon", "singleton", POLICY).flatten()
    gap = float(scores[1].item()-scores[0].item())
    assert controls[0].item() == pytest.approx(-gap, abs=5e-6)
    q = torch.ones(1,1,1, device=device)
    k = q.new_tensor([103.685, 0.]).view(1,2,1)
    v = q.new_tensor([1.,0.]).view_as(k)
    got = weighted_attention(q,k,v,controls, _weak_log=True)
    gradient, = torch.autograd.grad(got.sum(), scores)
    p = 1/(1+math.exp(gap-k[0,0,0].item()))
    assert got.item() == pytest.approx(p, rel=8e-6)
    torch.testing.assert_close(gradient.flatten(), q.new_tensor([p*(1-p), -p*(1-p)]), rtol=1e-5, atol=1e-8)


def test_real_zero_right_derivative_amplitude_and_padding(device):
    q = torch.zeros(1,1,1, device=device)
    k = q.expand(1,5,1)
    v = q.new_tensor([1.,2.,4.,8.,16.]).reshape_as(k)
    # Exactly one physical unit plus a weak positive lies *above* the KV cap.
    code = q.new_tensor([1., -100., 0., 0., 0.]).requires_grad_()
    mask = torch.tensor([True, True, True, True, False], device=device)
    proof = torch.tensor([False, False, False, True, False], device=device)
    out = weighted_attention(q,k,v,code,mask,mass_scaled=True,_weak_log=True,_constant_zero_prior=proof)
    grad, = torch.autograd.grad(out.sum(), code)
    assert out.item() == 1 and grad[0].item() == pytest.approx(0.,abs=1e-30)
    assert grad[2:].tolist() == [3.,0.,0.]
    # On the M=1 endpoint the ordinary clamp retains its amplitude derivative.
    endpoint = q.new_tensor([1.,0.]).requires_grad_()
    y = weighted_attention(q,k[:,:2],v[:,:2],endpoint,mass_scaled=True,_weak_log=True)
    torch.testing.assert_close(torch.autograd.grad(y.sum(),endpoint)[0],q.new_tensor([1.,2.]))
    # A weak-only reception has finite log cotangents despite unrepresentable dm.
    parts = q.new_tensor([[-115.,-116.]]).requires_grad_()
    t, mean = reception_mean(parts, q.new_tensor([[[2.],[5.]]]), None)
    p = 1/(1+math.exp(1))
    assert t.item() == pytest.approx(-115+math.log1p(math.exp(-1)),abs=6e-6)
    assert mean.item() == pytest.approx(2+3*p,rel=1e-6)
    torch.testing.assert_close(torch.autograd.grad(mean.sum(),parts)[0],q.new_tensor([[-3*p*(1-p),3*p*(1-p)]]))


def test_log_total_projects_absent_roots_and_preserves_a_movable_zero(device):
    from tide.scaled_arithmetic import GradientRangeError

    mass = torch.tensor([[-110., 0., 0.]], device=device, requires_grad=True)
    values = torch.ones(1, 3, 1, device=device)
    fixed = torch.tensor([[True, False, False]], device=device)
    total, _ = reception_mean(mass, values, fixed)
    assert total.item() == -110.
    torch.testing.assert_close(torch.autograd.grad(total.sum(), mass)[0], mass.new_tensor([[1., 0., 0.]]))

    movable = torch.tensor([[True, True, False]], device=device)
    total, _ = reception_mean(mass, values, movable)
    scale = 2. ** -40
    expected = mass.new_tensor([[scale, scale * math.exp(110.), 0.]])
    torch.testing.assert_close(torch.autograd.grad(total.sum() * scale, mass, retain_graph=True)[0], expected)
    with pytest.raises(GradientRangeError):
        torch.autograd.grad(total.sum(), mass)


def test_readout_gradient_when_a_weak_sender_takes_the_total_above_one(device):
    from tide.model import WeakLogResult
    from tide.weak_mass import physical

    code = torch.tensor([1., -100.], device=device).reshape(2, 1, 1).requires_grad_()
    values = torch.tensor([2., 4.], device=device).reshape(2, 1, 1, 1)
    result = WeakLogResult(values, physical(code), torch.arange(1, device=device), None,
                           strength_coordinates=code)
    grad, = torch.autograd.grad(result.readout().sum(), code)
    assert result.readout().item() == 2.
    assert abs(grad[0].item()) < 1e-30


def timed_weak_model(device, batch_regions=1):
    torch.manual_seed(18)
    topology = TopologyConfig(kind="multiscale", profile="compact-7", n=2, hubs=1,
                              partition="singleton", timing="timed", period=1)
    model = TideModel(ModelConfig(topology=topology,d=64,memory="K",allocation="shannon",budget=1,
                                  shannon_support_policy=POLICY,region_batch_size=batch_regions)).to(device)
    with torch.no_grad():
        for members in model.graph.regions:
            if len(members) == 2 and all(v in model.description_ids for v in members):
                owner = model.description_ids[members[0]]
                model.describe.offset.value[owner] = -115.
    return model


@pytest.mark.parametrize("batch_regions", [1, 4])
def test_timed_kv_pending_and_checkpoint_preserve_weak_support(tmp_path, device, batch_regions):
    from tide.model import WeakLogState
    model = timed_weak_model(device, batch_regions)
    with torch.no_grad():
        x = torch.randn(1,12,64,device=device)
        full = model(x)
        a = model(x[:,:8])
        assert isinstance(a.state, WeakLogState)
        assert any((m["K"].weight < 0).any() for m in a.state.memories if "K" in m)
        assert any((p.strengths < 0).any() for p in a.state.pending)
        file = tmp_path/"prefix.pt"
        save_checkpoint(file, model, state=a.state)
        restored, state, _ = load_checkpoint(file,device=device)
        assert isinstance(state, WeakLogState)
        b = restored(x[:,8:],state)
        assert torch.equal(b.strength_coordinates!=0,full.strength_coordinates[:,:,8:]!=0)
        torch.testing.assert_close(b.readout(),full.readout()[:,8:],rtol=3e-4,atol=3e-5)


def test_cpu_weak_log_adamw_restores_the_next_update(tmp_path):
    from benchmarks.continuous import checkpoint_resume

    model = timed_weak_model("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    prefix = model(torch.randn(1, 12, 64))
    assert any((bank.strengths < 0).any() for bank in prefix.state.pending)
    prefix.readout().square().mean().backward()
    optimizer.step()
    assert optimizer.state
    check = checkpoint_resume(model, optimizer, prefix.state, torch.randn(1, 4, 64),
                              "fp32", tmp_path / "weak-adamw.pt")
    assert check["saved_data"]["exact"]
    assert check["next_forward"]["exact"] and check["next_state"]["exact"]
    assert check["next_flush_result"]["exact"] and check["state"] == "within_threshold"
