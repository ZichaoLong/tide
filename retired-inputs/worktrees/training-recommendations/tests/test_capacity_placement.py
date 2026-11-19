"""Meta-only parameter ownership budgets versus actual small shard tensors."""
import pytest
import torch

from benchmarks.capacity import placement_budget, scale_configs
from benchmarks.continuous import state_memory
from tide import ModelConfig, TopologyConfig, TideModel, place_model, placement_report
from tide.precision import precision_context


def _actual_counts(model):
    # Enumerate the actual constructed shard Parameters, independently of the
    # planner's original owner/group/source-class row formulas.
    parts = [dict() for _ in model._placement_devices]
    for wrapper in model._shard_registry.values():
        for partition, shard in zip(wrapper._live_partitions, wrapper.shards):
            parts[partition].update((id(p), p) for p in shard.parameters())
    for region, group in enumerate(model.region_group):
        if group >= 0:
            layer = model.graph.nodes[model.graph.regions[region][0]].layer
            parts[model._layer_partitions[layer]].update((id(p), p) for p in model.regional[group].parameters())
    parts[model._layer_partitions[0]].update((id(p), p) for p in model.inputs.parameters())
    ids = [identifier for part in parts for identifier in part]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return [sum(p.numel() for p in part.values()) for part in parts]


@pytest.mark.parametrize("sharing", ["independent", "region_shared", "layer_shared"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_exact_meta_partition_count_and_cold_private_state(sharing, precision):
    config = ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1),
                         d=64, memory="X", sharing=sharing, aggregation="incoming_wide")
    planned = placement_budget(config, batch=1, length=8, precision=precision)
    model = place_model(TideModel(config), ["cpu", "cpu"])
    assert planned["layer_partitions"] == list(model._layer_partitions)
    assert [row["unique_parameters"] for row in planned["partitions"]] == _actual_counts(model)
    assert planned["unique_parameters"] == placement_report(model)["total_unique_parameters"]
    assert sum(row["parameter_bytes"] for row in planned["partitions"]) == placement_report(model)["actual_devices"]["cpu"]["parameter_bytes"]
    with torch.no_grad(), precision_context("cpu", precision):
        result = model(torch.randn(1, 8, 64))
    assert sum(row["cold_state_payload_bytes"] for row in planned["partitions"]) == state_memory(result.state)["payload_bytes"]
    # The current Settle convolution representation retains padded logical slots,
    # not only four positive events; this must be counted at its actual capacity.
    assert planned["state_layout"]["cold"]["convolution_slots"] == 8


def test_timed_meta_state_and_singleton_shared_rows_are_accounted():
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4",
                         timing="timed", partition="singleton"), d=64, memory="X", sharing="layer_shared")
    planned = placement_budget(config, batch=2, length=3, precision="bf16")
    model = place_model(TideModel(config), ["cpu", "cpu"])
    assert [row["unique_parameters"] for row in planned["partitions"]] == _actual_counts(model)
    with torch.no_grad(), precision_context("cpu", "bf16"):
        result = model(torch.randn(2, 3, 64))
    assert sum(row["cold_state_payload_bytes"] for row in planned["partitions"]) == state_memory(result.state)["payload_bytes"]
    assert sum(row["steady_state_retained_upper_bytes"] for row in planned["partitions"]) >= state_memory(result.state)["retained_storage_bytes"]


def test_17b_budget_materializes_no_model_payload(monkeypatch):
    from benchmarks import capacity
    original = capacity.TideModel
    def meta_only(config):
        model = original(config)
        assert all(parameter.is_meta for parameter in model.parameters())
        return model
    monkeypatch.setattr(capacity, "TideModel", meta_only)
    planned = placement_budget(scale_configs()["hb-17b"], batch=1, length=8, precision="bf16")
    assert 16e9 < planned["unique_parameters"] < 18e9
    assert sum(row["unique_parameters"] for row in planned["partitions"]) == planned["unique_parameters"]
    assert all(row["headroom_after_parameter_floor_and_steady_state_bytes"] > 0 for row in planned["partitions"])


@pytest.mark.parametrize("timing", ["settle", "timed"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_explicit_clocks_and_compact_pending_budget_match_live_state(timing, precision):
    from benchmarks.capacity import budget
    from tide import ClockConfig
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                         profile="compact-5-dd4", timing=timing), d=64, memory="X",
                         clock=ClockConfig(kv_window=7, ssd_stride=2))
    planned = budget(config, 2, 9, precision, "inference")
    model = TideModel(config)
    expected_kv_slots = (7+model.graph.period-1)//model.graph.period if timing == "settle" else 7
    assert planned["state_layout"]["kv_slots"] == expected_kv_slots
    assert planned["state_layout"]["convolution_slots"] == 7
    assert planned["state_layout"]["pending_retained_storage_upper_per_node"] == (
        0 if timing == "settle" else 2*65*4)
    for placement in (None, "layer", "region"):
        candidate = TideModel(config)
        if placement:
            plan = placement_budget(config, 2, 9, precision, partition_kind=placement)
            layout = {placement+"_partitions": plan[placement+"_partitions"]}
            candidate = place_model(candidate, ["cpu", "cpu"], **layout)
            expected = sum(row["cold_state_payload_bytes"] for row in plan["partitions"])
        else:
            expected = planned["bounded_state_bytes"]
        with torch.no_grad(), precision_context("cpu", precision):
            result = candidate(torch.randn(2, 9, 64))
        measured = state_memory(result.state)
        assert measured["payload_bytes"] == expected
        assert measured["retained_storage_bytes"] <= expected
