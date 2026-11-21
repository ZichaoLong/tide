"""CPU/Ascend support, row budgets, analytic gradients and regional integration."""
import math
import os

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from tide.allocation import allocate, pool
from tide.pool_support import FLOOR, observe_pool_support

DEVICE = os.environ.get("TIDE_STABLE_DEVICE", "cpu")
if DEVICE.startswith("npu"):
    import torch_npu  # noqa: F401

DELTA = 2.**-149


def tensor(values, grad=False):
    return torch.tensor(values, dtype=torch.float32, device=DEVICE, requires_grad=grad)


def test_fault_and_unrounded_derivatives_with_real_zero():
    s = tensor([-80.67035675048828, 25.64478302001953, 25.64478302001953], True)
    mu = tensor([1, 1, 0], True)
    with pytest.raises(ArithmeticError, match="erase a positive sender"):
        pool(s, mu)
    with observe_pool_support() as counts:
        c = pool(s, mu, support_policy=FLOOR)
    assert c.tolist() == [DELTA, 1, 0]
    assert (counts.rows, counts.repaired_rows, counts.repaired_coordinates) == (1, 1, 1)
    root = tensor([2.**100, 0., 1.])
    ds, du = torch.autograd.grad(c, (s, mu), root)
    g = math.exp(s[0].item()-s[1].item()) * 2.**100
    torch.testing.assert_close(ds, tensor([g, -g, 0.]), rtol=3e-5, atol=0)
    torch.testing.assert_close(du, tensor([g, -g, 1.]), rtol=3e-5, atol=0)


@pytest.mark.parametrize("priors,score,units", [
    ([2,2],[0,3],[1,3]), ([1,1,1],[-200,0,-300],[1,1,1]),
    ([0,4,2,3],[500,-100,0,-200],[0,1,7,1])])
def test_integer_storage_units_conserve_tiny_total(priors, score, units):
    mu = tensor([n*DELTA for n in priors])
    c = pool(tensor(score), mu, support_policy=FLOOR)
    assert c.tolist() == [n*DELTA for n in units] and c.sum().item() == mu.sum().item()


def test_broadcast_ordinary_rows_and_empty_support():
    s = tensor([[-108., 0.], [0., 0.]], True)
    mu = tensor([1., 1.], True)
    with observe_pool_support() as counts:
        c = allocate(s, mu, mu, 0., mu, 1, kind="pool", support_policy=FLOOR)
    assert c.tolist() == [[DELTA,1.],[.5,.5]] and counts.rows == 2
    assert torch.equal(c[1], pool(s[1], mu))
    assert torch.equal(pool(s, mu*0, support_policy=FLOOR), torch.zeros_like(s))
    assert pool(s[:,:0], mu[:0], support_policy=FLOOR).shape == (2,0)
    with pytest.raises(ValueError):
        pool(s, mu, support_policy="unrecognized")


@pytest.mark.skipif(DEVICE != "cpu", reason="FP64 CPU reference")
def test_fp64_least_subnormal():
    s = torch.tensor([-800.,0.], dtype=torch.float64)
    c = pool(s, torch.ones_like(s), support_policy=FLOOR)
    assert c.tolist() == [2.**-1074,1.]


@pytest.mark.parametrize("group", [1, 4])
def test_model_policy_serialization_routing_and_work(group):
    spec = ModelConfig(topology=TopologyConfig(kind="lattice", n=2, depth=1), d=64,
                       allocation="pool", pool_support_policy=FLOOR, region_batch_size=group,
                       attention_backend="math")
    assert ModelConfig.from_dict(spec.to_dict()) == spec
    assert "pool_support_policy" not in ModelConfig().to_dict()
    with pytest.raises(ValueError):
        ModelConfig(allocation="shannon", pool_support_policy=FLOOR)
    torch.manual_seed(71)
    model = TideModel(spec).to(DEVICE)
    with torch.no_grad():
        model.describe.readout.value.zero_()
        for net in model.regional:
            net.readout.zero_()
        for region in model.layer_regions[1]:
            a,b = model.graph.regions[region]
            model.describe.offset.value[model.description_ids[a]].fill_(-110)
            model.describe.offset.value[model.description_ids[b]].fill_(2)
    x = torch.randn(1,2,64,device=DEVICE,requires_grad=True)
    model(x)
    out = model(x, record_work=True)
    counts = out.work["pool_support_training_forwards"]
    assert counts["rows"] > counts["repaired_rows"] >= 8
    out.readout().square().mean().backward()
    assert bool(torch.isfinite(x.grad).all())
