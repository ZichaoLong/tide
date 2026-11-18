"""Placed batches preserve private state, owner gradients and resumed updates."""
from dataclasses import replace
import threading

import pytest
import torch

from benchmarks.continuous import nested_comparison
from tide import (TideModel, canonical_state, canonical_state_dict, detach_state,
                  load_checkpoint, place_model, placement_adamw, save_checkpoint)
from test_region_placement import config, layout, objective


def models(timing, concurrency, recompute=False):
    cfg = config(timing=timing, recompute=recompute)
    cfg = replace(cfg, topology=replace(cfg.topology, profile="compact-9"))
    reference = TideModel(cfg).double()
    batched = TideModel(replace(cfg, region_batch_size=4)).double()
    batched.load_state_dict(reference.state_dict())
    owners = layout(reference)
    return reference, place_model(batched, ["cpu:0", "cpu:1"], region_partitions=owners,
                                  region_concurrency=concurrency)


@pytest.mark.parametrize("timing", ["settle", "timed"])
@pytest.mark.parametrize("concurrency", ["serial", "per_device"])
def test_placed_batches_values_all_gradients_continuation(timing, concurrency, monkeypatch):
    torch.manual_seed(814)
    reference, placed = models(timing, concurrency)
    import tide.region_batch as batching
    original = batching.evaluate_regions
    seen = []

    def observe(model, regions, *args, **kwargs):
        assert len({model._region_partitions[region] for region in regions}) == 1
        seen.append(tuple(regions))
        return original(model, regions, *args, **kwargs)

    monkeypatch.setattr(batching, "evaluate_regions", observe)
    caller = threading.get_ident()
    for wrapper in placed._shard_registry.values():
        prepare = wrapper._prepare_dispatch

        def guarded(possible, wrapper=wrapper, prepare=prepare):
            if possible not in wrapper._dispatch:
                assert threading.get_ident() == caller
            return prepare(possible)

        monkeypatch.setattr(wrapper, "_prepare_dispatch", guarded)
    x = torch.randn(1, 3, 64, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x, record_work=True), placed(y, record_work=True)
    assert seen and max(map(len, seen)) > 1
    assert expected.work == actual.work
    torch.testing.assert_close(actual.values, expected.values, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(actual.strengths, expected.strengths, atol=2e-9, rtol=2e-9)
    assert nested_comparison(canonical_state(placed, actual.state), expected.state,
                             atol=2e-9, rtol=2e-9)["within_threshold"]
    objective(expected).backward(); objective(actual).backward()
    torch.testing.assert_close(y.grad, x.grad, atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(canonical_state_dict(placed, gradients=True),
                               {name: value.grad for name, value in reference.named_parameters()},
                               atol=3e-10, rtol=3e-8)
    first = placed(y[:, :1]); second = placed(y[:, 1:], first.state)
    torch.testing.assert_close(torch.cat((first.values, second.values), 2), expected.values,
                               atol=2e-9, rtol=2e-9)
    if timing == "timed":
        wanted, tail = reference.flush(expected.state), placed.flush(second.state)
        torch.testing.assert_close(tail.values, wanted.values, atol=2e-9, rtol=2e-9)
        assert nested_comparison(canonical_state(placed, tail.state), wanted.state,
                                 atol=2e-9, rtol=2e-9)["within_threshold"]
        for bank in tail.state.pending:
            for chunk in bank.regions:
                for tensor in (chunk.values, chunk.strengths):
                    assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()


def update(model, optimizer, x, state):
    optimizer.zero_grad(set_to_none=True)
    result = model(x, state)
    loss = objective(result)
    if not model.settle:
        result = model.flush(result.state)
        loss = loss+objective(result)
    loss.backward(); optimizer.step()
    return detach_state(result.state)


@pytest.mark.parametrize("timing", ["settle", "timed"])
def test_placed_batch_recompute_three_updates_and_restored_update(tmp_path, timing):
    torch.manual_seed(116)
    reference, placed = models(timing, "per_device", recompute=True)
    optimizers = [torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=False),
                  placement_adamw(placed, lr=1e-4, foreach=False)]
    x = torch.randn(1, 2, 64, dtype=torch.float64)
    states = [None, None]
    for _ in range(3):
        states = [update(model, optimizer, x, state)
                  for model, optimizer, state in zip((reference, placed), optimizers, states)]
        torch.testing.assert_close(canonical_state_dict(placed), reference.state_dict(),
                                   atol=2e-9, rtol=2e-7)
    path = tmp_path/"batched-placed.pt"
    save_checkpoint(path, placed, state=states[1], optimizer=optimizers[1])
    restored, state, slots = load_checkpoint(path, placement_devices=["cpu:1", "cpu:0"])
    resumed = placement_adamw(restored, lr=1e-4, foreach=False)
    resumed.load_state_dict(slots)
    assert restored.config.region_batch_size == 4
    assert nested_comparison(restored.state_dict(), placed.state_dict(), atol=0, rtol=0)["exact"]
    assert nested_comparison(state, states[1], atol=0, rtol=0)["exact"]
    a = update(placed, optimizers[1], x, states[1])
    b = update(restored, resumed, x, state)
    for actual, expected in ((b, a), (restored.state_dict(), placed.state_dict()),
                             (resumed.state_dict(), optimizers[1].state_dict())):
        assert nested_comparison(actual, expected, atol=2e-9, rtol=2e-7)["within_threshold"]


def test_repeated_device_partitions_remain_separate_batches():
    from tide.region_placement import region_batches
    cfg = replace(config(), region_batch_size=4)
    cfg = replace(cfg, topology=replace(cfg.topology, profile="compact-9"))
    model = TideModel(cfg)
    place_model(model, ["cpu", "cpu"], region_partitions=layout(model))
    batches = [group for regions in model.layer_regions for group in region_batches(model, regions, None)]
    assert any(len(group) > 1 for group in batches)
    assert all(len({model._region_partitions[r] for r in group}) == 1 for group in batches)
