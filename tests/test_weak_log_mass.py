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


@pytest.mark.parametrize("batch_regions", [1, 4])
def test_timed_kv_pending_and_checkpoint_preserve_weak_support(tmp_path, device, batch_regions):
    from tide.model import WeakLogState
    torch.manual_seed(18)
    topology = TopologyConfig(kind="multiscale", profile="compact-7", n=2, hubs=1,
                              partition="singleton", timing="timed", period=1)
    model = TideModel(ModelConfig(topology=topology,d=64,memory="K",allocation="shannon",budget=1,
                                  shannon_support_policy=POLICY,region_batch_size=batch_regions)).to(device)
    with torch.no_grad():
        for region, members in enumerate(model.graph.regions):
            if len(members) == 2 and all(v in model.description_ids for v in members):
                owner = model.description_ids[members[0]]
                model.describe.offset.value[owner] = -115.
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
