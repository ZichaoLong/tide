"""Independent recurrence and full BPTT witnesses for optional training paths."""
from dataclasses import replace

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig, detach_state, load_checkpoint, save_checkpoint
from tide.precision import precision_context
from tide.ssd import affine_prefix
from benchmarks.continuous import nested_comparison


@pytest.mark.parametrize("length", [0, 1, 67])
def test_affine_prefix_against_sequential_formula_and_all_input_derivatives(length):
    torch.manual_seed(553)
    a = torch.rand(2, length, 3, dtype=torch.float64)
    a[:, ::7] = 0
    a[:, 1::7] = 1
    a.requires_grad_()
    b = torch.randn(2, length, 3, 5, dtype=torch.float64, requires_grad=True)
    initial = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    state, rows = initial, []
    for t in range(length):
        state = a[:, t, :, None] * state + b[:, t]
        rows.append(state)
    expected = torch.stack(rows, 1) if rows else b
    actual, final = affine_prefix(a, b, initial)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(final, state, atol=2e-14, rtol=2e-14)
    probe = torch.randn_like(actual)
    tail_probe = torch.randn_like(final)
    grad = torch.autograd.grad((actual * probe).sum() + (final * tail_probe).sum(), (a, b, initial),
                               retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad((expected * probe).sum() + (state * tail_probe).sum(), (a, b, initial),
                                    allow_unused=True)
    for x, y in zip(grad, reference):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, atol=3e-13, rtol=3e-13)


def update(model, optimizer, state, inputs, precision="fp32"):
    optimizer.zero_grad(set_to_none=True)
    x = inputs.clone().requires_grad_()
    with precision_context(x.device, precision):
        first = model(x[:, :2], state)
        second = model(x[:, 2:], first.state)
        results = [first, second]
        if not model.settle:
            results.append(model.flush(second.state))
        values = torch.cat([r.readout() for r in results], 1)
        probe = torch.sin(torch.arange(values.numel(), dtype=values.dtype)).reshape_as(values)
        loss = (values * probe).mean()
    loss.backward()
    grads = [x.grad] + [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    optimizer.step()
    return detach_state(results[-1].state), grads, detach_state(results)


@pytest.mark.parametrize("timing,aggregation,whole_region", [
    ("settle", "source_residual", False), ("timed", "source_residual", False),
    ("settle", "incoming_light_64", False), ("timed", "incoming_wide", True),
])
def test_partial_recompute_and_prefix_preserve_shared_bptt_updates(timing, aggregation, whole_region):
    torch.manual_seed(554)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                         n=2, hubs=1, timing=timing), d=64, memory="X", sharing="layer_shared",
                         region_batch_size=4, attention_backend="math", aggregation=aggregation,
                         checkpoint_regions=whole_region)
    reference = TideModel(config).double()
    candidate = TideModel(replace(config, checkpoint_reception=True, ssd_summary_backend="prefix")).double()
    candidate.load_state_dict(reference.state_dict())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (reference, candidate)]
    states = [None, None]
    for _ in range(3):
        x = torch.randn(1, 5, 64, dtype=torch.float64)
        observations = []
        for i, (m, opt) in enumerate(zip((reference, candidate), optimizers)):
            states[i], grads, results = update(m, opt, states[i], x)
            observations.append(dict(state=states[i], gradients=grads, results=results,
                                     parameters=m.state_dict(), optimizer=opt.state_dict()))
        report = nested_comparison(*observations, atol=2e-8, rtol=2e-7)
        assert report["within_threshold"], report
        assert not report["support_mismatches"], report


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_partial_recompute_prefix_checkpoint_restores_real_update(tmp_path, precision):
    torch.manual_seed(555)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                         n=2, hubs=1, timing="timed"), d=64, memory="X", sharing="region_shared",
                         region_batch_size=4, checkpoint_reception=True, ssd_summary_backend="prefix")
    model = TideModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state, _, _ = update(model, optimizer, None, torch.randn(1, 4, 64), precision)
    path = tmp_path / "training.pt"
    save_checkpoint(path, model, state=state, optimizer=optimizer)
    restored, rs, slots = load_checkpoint(path)
    assert restored.config == config
    ropt = torch.optim.AdamW(restored.parameters()); ropt.load_state_dict(slots)
    inputs = torch.randn(1, 4, 64)
    observations = []
    for m, opt, s in ((model, optimizer, state), (restored, ropt, rs)):
        following, grads, results = update(m, opt, s, inputs, precision)
        observations.append(dict(state=following, gradients=grads, results=results,
                                 parameters=m.state_dict(), optimizer=opt.state_dict()))
    assert nested_comparison(*observations, atol=0, rtol=0)["exact"]
