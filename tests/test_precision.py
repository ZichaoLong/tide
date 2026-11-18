"""BF16 arithmetic witnesses, independent derivatives, and owned state."""
import pytest
import torch

from tide import ModelConfig, TopologyConfig, TideModel, detach_state, load_checkpoint, save_checkpoint
from tide.attention import weighted_attention
from tide.parameters import NodeLinear, NodeOwners
from tide.precision import ArithmeticAudit, precision_context, precision_policy


@pytest.mark.parametrize("backend", ["math", "tiled", "auto"])
@pytest.mark.parametrize("zero", [False, True])
def test_bf16_attention_independent_quotient_vjp(backend, zero):
    torch.manual_seed(5)
    q = torch.randn(2, 2, 3, 8).bfloat16().requires_grad_()
    k = torch.randn(2, 2, 4, 8).bfloat16().requires_grad_()
    v = torch.randn(2, 2, 4, 6).bfloat16().requires_grad_()
    w = torch.tensor([[0 if zero else .2, 2**-20, .7, 1.], [.4, .8, .3, .9]], requires_grad=True)
    qr, kr, vr, wr = [t.detach().double().requires_grad_() for t in (q, k, v, w)]
    mask = torch.tensor([[True, True, False, True], [True, True, True, False], [True]*4])
    logits = qr @ kr.transpose(-1, -2) / 8**.5
    # Independent normalized weighted quotient, including w=0 right derivatives.
    terms = logits.exp() * wr[:, None, None] * mask
    expected = (terms @ vr) / terms.sum(-1, keepdim=True)
    actual = weighted_attention(q, k, v, w, mask, backend)
    seed = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * seed).sum(), (q, k, v, w))
    expected_grad = torch.autograd.grad((expected * seed.double()).sum(), (qr, kr, vr, wr))
    torch.testing.assert_close(actual.double(), expected, rtol=.015, atol=.008)
    for actual_g, expected_g in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_g.double(), expected_g, rtol=.04, atol=.025)
    assert w.grad is None  # autograd.grad did not replace caller-owned accumulation.


def test_bf16_shared_linear_uses_bf16_but_accumulates_fp32_parameters():
    torch.manual_seed(6)
    module = NodeLinear(3, 16, 8, [0, 0, 0], rank=2)
    owners = NodeOwners(torch.tensor([2, 0, 2]), (0, 1, 2))
    x = torch.randn(3, 2, 16, requires_grad=True)
    with precision_context("cpu", "bf16"), ArithmeticAudit() as audit:
        y = module(x, owners)
    y.float().square().sum().backward()
    assert y.dtype == torch.bfloat16
    assert all(p.dtype == p.grad.dtype == torch.float32 for p in module.parameters())
    assert any("cpu:bfloat16" in r["inputs"] for r in audit.report())
    # Repeated owner contributions remain present; the unused adapter is disconnected numerically.
    assert module.left.grad[2].abs().sum() > 0
    assert torch.count_nonzero(module.left.grad[1]) == 0
    with precision_context("cpu", "bf16"):
        with precision_context("cpu", "fp32"):
            assert module(x, owners).dtype == torch.float32


@pytest.mark.parametrize("memory", ["K", "S"])
def test_bf16_model_updates_and_checkpoint(tmp_path, memory):
    torch.manual_seed(71)
    model = TideModel(ModelConfig(topology=TopologyConfig(n=8, layers=4), d=64,
                                  memory=memory, sharing="region_shared"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)
    x = torch.randn(1, 3, 64)
    state = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with precision_context("cpu", "bf16"):
            result = model(x, state)
            loss = result.readout().float().square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        state = detach_state(result.state)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, state=state, optimizer=optimizer)
    restored, saved, slots = load_checkpoint(path)
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), atol=0, rtol=0)
    torch.testing.assert_close(slots, optimizer.state_dict(), atol=0, rtol=0)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, foreach=True)
    restored_optimizer.load_state_dict(slots)
    with torch.no_grad(), precision_context("cpu", "bf16"):
        torch.testing.assert_close(restored(x, saved).readout(), model(x, state).readout(), atol=0, rtol=0)
    # Exact saved data is distinct from a bitwise subsequent optimizer trajectory.
    for subject, opt, carry in ((model, optimizer, state), (restored, restored_optimizer, saved)):
        opt.zero_grad(set_to_none=True)
        with precision_context("cpu", "bf16"):
            value = subject(x, carry).readout()
            loss = value.float().square().mean()
        loss.backward()
        opt.step()
        assert all(torch.isfinite(p).all() for p in subject.parameters())
        assert all(p.dtype == torch.float32 for p in subject.parameters())
        assert all(slot.dtype == torch.float32 for entry in opt.state.values()
                   for name, slot in entry.items() if name in {"exp_avg", "exp_avg_sq"})


def test_precision_policy_meta_and_invalid_requests():
    assert precision_policy("cpu", "bf16")["parameters"] == "float32"
    with pytest.raises(ValueError, match="precision"):
        precision_policy("cpu", "fp16")
    with torch.device("meta"):
        model = TideModel(ModelConfig(topology=TopologyConfig(n=8, layers=4), d=64))
    assert sum(p.numel() for p in model.parameters()) > 0
    assert all(p.device.type == "meta" for p in model.parameters())


def test_qualification_comparison_checks_clock_none_and_actual_support():
    from scripts.qualify_precision import comparison
    tolerance = {"atol": 1e-6, "rtol": 2e-3}
    result = comparison({"position": 2, "gradient": None, "weight": torch.tensor([-1.]),
                         "state": {"weight": torch.tensor([0.])}},
                        {"position": 1, "gradient": torch.tensor([0.]), "weight": torch.tensor([1.]),
                         "state": {"weight": torch.tensor([1.])}}, tolerance)
    assert "root.position" in result["metadata_failures"]
    assert "root.gradient.type" in result["metadata_failures"]
    assert [x["tensor"] for x in result["support_differences"]] == ["root.state.weight"]
