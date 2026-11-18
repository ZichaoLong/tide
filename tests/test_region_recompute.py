"""Activation recomputation retains stateful gradient/update semantics."""
from dataclasses import replace

import pytest
import torch

from tide import ModelConfig, TopologyConfig, TideModel, detach_state, save_checkpoint, load_checkpoint
from tide.precision import precision_context


def test_detached_timed_pending_owns_only_last_tick_storage():
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                                 n=2, hubs=1, timing="timed"), d=64, memory="X"))
    with torch.no_grad():
        state = detach_state(model(torch.randn(2, 13, 64)).state)
    assert state.pending
    for layer in state.pending:
        for tensor in (layer.values, layer.strengths):
            assert tensor.shape[2] == 1
            assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()


@pytest.mark.parametrize("memory,timing,precision", [
    ("K", "settle", "fp32"), ("S", "settle", "bf16"),
    ("X", "timed", "fp32"), ("X", "timed", "bf16"),
])
def test_recomputed_regions_continue_backward_and_restore(memory, timing, precision, tmp_path):
    torch.manual_seed(491)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                         n=2, hubs=1, timing=timing), d=64, memory=memory,
                         sharing="region_shared", attention_backend="math")
    ordinary = TideModel(config)
    recomputed = TideModel(replace(config, checkpoint_regions=True))
    recomputed.load_state_dict(ordinary.state_dict())
    models = [ordinary, recomputed]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in models]
    states = [None, None]
    for step in range(3):
        x = torch.randn(1, 4, 64)
        outputs, gradients = [], []
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            input = x.clone().requires_grad_()
            with precision_context("cpu", precision):
                first = model(input[:, :2], states[i], record_work=True)
                second = model(input[:, 2:], first.state, record_work=True)
                values = [first.readout(), second.readout()]
                end = second
                if timing == "timed":
                    end = model.flush(second.state)
                    values.append(end.readout())
                loss = sum(value.float().square().sum() for value in values)
            loss.backward()
            outputs.append((values, first.work, second.work))
            gradients.append([None if p.grad is None else p.grad.clone() for p in model.parameters()])
            gradients[-1].append(input.grad)
            optimizer.step()
            states[i] = detach_state(end.state)
        for a, b in zip(outputs[0][0], outputs[1][0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert outputs[0][1:] == outputs[1][1:]
        for a, b in zip(gradients[0], gradients[1]):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(ordinary.parameters(), recomputed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if step == 1:
            path = tmp_path / "recomputed.pt"
            save_checkpoint(path, recomputed, state=states[1], optimizer=optimizers[1])
            restored, states[1], slots = load_checkpoint(path)
            assert restored.config.checkpoint_regions
            models[1] = recomputed = restored
            optimizers[1] = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            optimizers[1].load_state_dict(slots)
