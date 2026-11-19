"""Execution configuration preserves storage; work reports actual FFN rows."""
from dataclasses import replace

import pytest
import torch

from benchmarks.capacity import _private_state_budget
from benchmarks.statistics import audit_model
from tide import ModelConfig, TideModel, TopologyConfig


def test_execution_config_serialization_and_capacity_preserve_parameter_state_budget():
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                        profile="compact-5-dd4"), d=64, memory="X", kv_width=32)
    compact = ModelConfig.from_dict(replace(config, ffn_packing="power2").to_dict())
    assert compact.ffn_packing == "power2"
    with torch.device("meta"):
        original, candidate = TideModel(config), TideModel(compact)
    assert {n: p.shape for n, p in original.named_parameters()} == {
        n: p.shape for n, p in candidate.named_parameters()}
    for steady in (False, True):
        assert _private_state_budget(original, 8, 128, "fp32", steady=steady) == _private_state_budget(
            candidate, 8, 128, "fp32", steady=steady)
    with pytest.raises(ValueError, match="ffn_packing"):
        replace(config, ffn_packing="invalid")


def test_work_audit_separates_attention_rectangle_from_ffn_operands():
    torch.manual_seed(751)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=4, hubs=1,
                        profile="compact-5-dd4"), d=64, memory="X", kv_width=32,
                        region_batch_size=4, ffn_packing="power2")
    model = TideModel(config)
    with torch.no_grad():
        _, _, report = audit_model(model, torch.randn(4, 3, 64))
    for kind, rows in report["ffn_packing"].items():
        query = report["query_packing"][kind]
        assert rows["selected_rows"] == query["selected_rows"]
        assert rows["selected_rows"] <= rows["executed_rows"] <= query["executed_rows"]
        assert rows["executed_rows"] == report["node_matrices"][f"memories.{kind}.ffn.gate"]["rows"]
        for projection in ("up", "down"):
            assert rows["executed_rows"] == report["node_matrices"][f"memories.{kind}.ffn.{projection}"]["rows"]
