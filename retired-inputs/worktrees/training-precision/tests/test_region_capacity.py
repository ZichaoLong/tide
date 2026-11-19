"""Meta region/group counts checked against actual constructed small shards."""
import pytest
import torch

from benchmarks.capacity import placement_budget, region_ownership, scale_configs
from benchmarks.continuous import state_memory
from tide import ModelConfig, TopologyConfig, TideModel, place_model
from tide.precision import precision_context


def actual_counts(model):
    parts = [dict() for _ in model._placement_devices]
    for wrapper in model._shard_registry.values():
        for partition, shard in zip(wrapper._live_partitions, wrapper.shards):
            parts[partition].update((id(parameter), parameter) for parameter in shard.parameters())
    for region, group in enumerate(model.region_group):
        if group >= 0:
            parts[model._region_partitions[region]].update((id(parameter), parameter)
                                                         for parameter in model.regional[group].parameters())
    parts[0].update((id(parameter), parameter) for parameter in model.inputs.parameters())
    ids = [identifier for part in parts for identifier in part]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    return [sum(parameter.numel() for parameter in part.values()) for part in parts]


@pytest.mark.parametrize("sharing", ["independent", "region_shared"])
@pytest.mark.parametrize("timing,precision", [("settle", "fp32"), ("timed", "bf16")])
def test_region_meta_counts_and_state_equal_actual_shards(sharing, timing, precision):
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4",
                                                partition="mixed", timing=timing),
                         d=64, memory="X", sharing=sharing, aggregation="incoming_wide")
    planned = placement_budget(config, batch=2, length=3, precision=precision, mode="train", partition_kind="region")
    model = place_model(TideModel(config), ["cpu", "cpu"], region_partitions=planned["region_partitions"])
    assert [row["unique_parameters"] for row in planned["partitions"]] == actual_counts(model)
    assert sum(row["parameter_tensors"] for row in planned["partitions"]) == len(list(model.parameters()))
    assert sum(row["adamw_cpu_step_scalar_upper_bytes"] for row in planned["partitions"]) == 4*len(list(model.parameters()))
    assert sum(group["unique_parameters"] for group in planned["ownership"]["parameter_groups"]) == planned["unique_parameters"]
    assert all(len(group["regions"]) <= 1 for group in planned["ownership"]["parameter_groups"])
    with torch.no_grad(), precision_context("cpu", precision):
        result = model(torch.randn(2, 3, 64))
    assert sum(row["cold_state_payload_bytes"] for row in planned["partitions"]) == state_memory(result.state)["payload_bytes"]
    assert sum(row["steady_state_retained_upper_bytes"] for row in planned["partitions"]) >= state_memory(result.state)["retained_storage_bytes"]


def test_shared_groups_are_explicit_and_never_copied_for_budget():
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4"),
                         d=64, memory="K", sharing="layer_shared")
    ownership = region_ownership(config)
    shared = [group for group in ownership["parameter_groups"] if len(group["regions"]) > 1]
    assert shared
    assert ownership["unique_parameters"] == sum(group["unique_parameters"] for group in ownership["parameter_groups"])
    proposed = [0]*len(ownership["regions"])
    proposed[shared[0]["regions"][0]] = 1
    with pytest.raises(ValueError, match="shared parameter .* spans region partitions"):
        placement_budget(config, region_partitions=proposed, partition_kind="region")
    valid = placement_budget(config, partition_kind="region")
    for group in shared:
        assert len({valid["region_partitions"][region] for region in group["regions"]}) == 1


@pytest.mark.parametrize("name,parts", [("hb-9b", 4), ("hb-17b", 5)])
def test_large_region_plan_never_materializes_weights(monkeypatch, name, parts):
    from benchmarks import capacity
    original = capacity.TideModel
    def meta_only(config):
        model = original(config)
        assert all(parameter.is_meta for parameter in model.parameters())
        return model
    monkeypatch.setattr(capacity, "TideModel", meta_only)
    planned = placement_budget(scale_configs()[name], mode="train", partition_kind="region", partitions=parts)
    assert sum(row["unique_parameters"] for row in planned["partitions"]) == planned["unique_parameters"]
    assert planned["unique_parameters"] == (8778358224 if name == "hb-9b" else 16707220176)
    # This assertion concerns the static persistent payload only, not feasibility
    # of activations, communication, backward or a real large training run.
    assert max(row["memory"]["persistent_parameter_floor_bytes"] for row in planned["partitions"]) < 60*2**30
