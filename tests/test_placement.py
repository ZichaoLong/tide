"""Two logical HB partitions before admitting actual multi-device execution."""
import copy

import pytest
import torch

from tide import (ModelConfig, TopologyConfig, TideModel, canonical_state_dict, detach_state,
                  load_checkpoint, place_model, placement_adamw, placement_report, save_checkpoint)
from tide.precision import precision_context
from benchmarks.continuous import nested_comparison


def _config(kind="exchange", sharing="layer_shared", memory="X", timing="settle"):
    topology = (TopologyConfig(kind="multiscale", n=2, profile="compact-5-dd4", hubs=1, timing=timing)
                if kind == "multiscale" else TopologyConfig(kind=kind, n=2, depth=1, hubs=1))
    return ModelConfig(topology=topology, d=64, memory=memory, sharing=sharing,
                       aggregation="incoming_light_64")


def _objective(result):
    y = result.readout()
    return (y*torch.cos(torch.arange(y.numel(), dtype=y.dtype, device=y.device)).reshape_as(y)).mean()


@pytest.mark.parametrize("sharing", ["independent", "region_shared", "layer_shared"])
@pytest.mark.parametrize("kind,memory,timing", [("exchange", "X", "settle"), ("multiscale", "X", "timed")])
def test_two_cpu_partitions_preserve_values_sharing_all_gradients_state(sharing, kind, memory, timing):
    torch.manual_seed(405)
    reference = TideModel(_config(kind, sharing, memory, timing)).double()
    placed = place_model(copy.deepcopy(reference), ["cpu", "cpu"])
    torch.testing.assert_close(canonical_state_dict(placed), reference.state_dict(), atol=0, rtol=0)
    assert placement_report(placed)["total_unique_parameters"] == sum(p.numel() for p in reference.parameters())
    x = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = reference(x, record_work=True)
    first = placed(y[:, :2]); actual = placed(y[:, 2:], first.state, record_work=True)
    torch.testing.assert_close(torch.cat((first.readout(), actual.readout()), 1), expected.readout(), atol=2e-9, rtol=2e-9)
    comparison = nested_comparison(actual.state, expected.state, atol=2e-9, rtol=2e-9)
    assert comparison["within_threshold"] and not comparison["support_mismatches"]
    whole = placed(y, record_work=True)
    assert whole.work == expected.work
    _objective(expected).backward()
    _objective(whole).backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-10, rtol=3e-8)
    original_grads = {name: p.grad for name, p in reference.named_parameters()}
    torch.testing.assert_close(canonical_state_dict(placed, gradients=True), original_grads, atol=3e-10, rtol=3e-8)
    if timing == "timed":
        expected_flush = reference.flush(expected.state)
        actual_flush = placed.flush(actual.state)
        torch.testing.assert_close(actual_flush.readout(), expected_flush.readout(), atol=2e-9, rtol=2e-9)
        assert nested_comparison(actual_flush.state, expected_flush.state, atol=2e-9, rtol=2e-9)["within_threshold"]


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_partitioned_updates_and_checkpoint_restore_actual_update(tmp_path, precision):
    torch.manual_seed(63)
    reference = TideModel(_config())
    placed = place_model(copy.deepcopy(reference), ["cpu", "cpu"])
    a = torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=True)
    b = placement_adamw(placed, lr=1e-4, foreach=True)
    state_a = state_b = None
    x = torch.randn(2, 3, 64)
    for _ in range(3):
        for current, optimizer, state in ((reference, a, state_a), (placed, b, state_b)):
            optimizer.zero_grad(set_to_none=True)
            with precision_context("cpu", precision):
                result = current(x, state)
                loss = _objective(result)
            loss.backward(); optimizer.step()
            if current is reference:
                state_a = detach_state(result.state)
            else:
                state_b = detach_state(result.state)
        comparison = nested_comparison(canonical_state_dict(placed), reference.state_dict(), atol=1e-6, rtol=2e-3)
        assert comparison["within_threshold"]
    checkpoint = tmp_path/"partitioned.pt"
    save_checkpoint(checkpoint, placed, state=state_b, optimizer=b)
    restored, restored_state, slots = load_checkpoint(checkpoint, placement_devices=["cpu", "cpu"])
    resumed = placement_adamw(restored, lr=1e-4, foreach=True)
    resumed.load_state_dict(slots)
    assert nested_comparison(restored.state_dict(), placed.state_dict(), atol=0, rtol=0)["exact"]
    assert nested_comparison(restored_state, state_b, atol=0, rtol=0)["exact"]
    assert nested_comparison(resumed.state_dict(), b.state_dict(), atol=0, rtol=0)["exact"]
    for current, optimizer, state in ((placed, b, state_b), (restored, resumed, restored_state)):
        optimizer.zero_grad(set_to_none=True)
        with precision_context("cpu", precision):
            result = current(x, state)
            loss = _objective(result)
        loss.backward(); optimizer.step()
    assert nested_comparison(restored.state_dict(), placed.state_dict())["within_threshold"]
    assert nested_comparison(resumed.state_dict(), b.state_dict())["within_threshold"]


def test_sharded_adamw_preserves_original_parameter_none_zero_and_decay():
    torch.manual_seed(31)
    original = TideModel(_config(sharing="independent")).double()
    placed = place_model(copy.deepcopy(original), ["cpu", "cpu"])
    opt_a = torch.optim.AdamW(original.parameters(), lr=.01, weight_decay=.2)
    opt_b = placement_adamw(placed, lr=.01, weight_decay=.2)
    path = "memories.K.kv"
    original_parameter = dict(original.named_parameters())[path+".weight"]
    wrapper = placed._shard_registry[path]
    assert len(wrapper.shards) == 2
    for selected in (0, 1, 0):
        opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
        original_parameter.grad = torch.zeros_like(original_parameter)
        rows = wrapper._maps["weight"][selected]
        original_parameter.grad[list(rows)] = .7
        wrapper.shards[selected].weight.grad = torch.full_like(wrapper.shards[selected].weight, .7)
        other = wrapper.shards[1-selected].weight
        assert other.grad is None
        opt_a.step(); opt_b.step()
        assert other.grad is not None and not bool(other.grad.any())
        torch.testing.assert_close(canonical_state_dict(placed), original.state_dict(), atol=1e-15, rtol=1e-15)
    # Entirely disconnected logical parameters remain None and have no slots.
    for shard in placed._shard_registry["memories.K.summary"].shards:
        assert shard.weight.grad is None and shard.weight not in opt_b.state


def test_placement_validation_is_explicit(tmp_path):
    model = TideModel(_config())
    with pytest.raises(ValueError, match="mixed"):
        place_model(model, ["cpu", "cuda:0"])
    with pytest.raises(ValueError, match="contiguous"):
        place_model(model, ["cpu", "cpu"], layer_partitions=[1]*len(model.layer_nodes))
    placed = place_model(model, ["cpu", "cpu"])
    with pytest.raises(ValueError, match="already"):
        place_model(placed, ["cpu", "cpu"])
    with pytest.raises(ValueError, match="placement_adamw"):
        save_checkpoint(tmp_path/"wrong.pt", placed, optimizer=torch.optim.AdamW(placed.parameters()))
    save_checkpoint(tmp_path/"right.pt", placed)
    with pytest.raises(ValueError, match="number of partitions"):
        load_checkpoint(tmp_path/"right.pt", placement_devices=["cpu"])
