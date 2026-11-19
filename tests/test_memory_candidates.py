"""Independent witnesses for the explicitly extended memory bindings."""
from dataclasses import replace
import json

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from scripts.check_memory_candidates import (CANDIDATES, HARD_FAILURES, aggregate,
                                             cache_metadata_witness, candidate_configs, comparison_summary)
from scripts.memory_candidate_formulas import cache_contents, fixture, formula_witness, projected
from scripts.qualify_precision import THRESHOLDS, comparison


@pytest.mark.parametrize("width", [96, 64, 32])
def test_narrow_qkv_keeps_content_summary_ffn_and_independent_value_vjp(width):
    record = formula_witness(width, dtype=torch.float64)
    assert record["state"] == "within_threshold", record["checks"]
    assert record["heads"] == width // 32
    assert record["first_time"] > 2**53
    assert record["zero_mass_gradient_nonzeros"] == [0, 0, 0]
    assert all(len(bank) <= 4 for row in record["retained_times"] for bank in row)
    assert all(record["first_time"] not in bank for row in record["retained_times"] for bank in row)


def test_narrow_fp32_formula_and_private_adapter_gradient_sum():
    record = formula_witness(32)
    assert record["state"] == "within_threshold", record["checks"]


def test_narrow_huge_time_translation_and_zero_write_idle_state():
    args, _ = fixture(64, dtype=torch.float64)
    module, u, mass, times, owners, control = args
    got = projected(*args)
    offset = 2**55 + 19
    translated = projected(module, u, mass, times-offset, owners, control)
    for actual, expected in zip(got["description"] + (got["full"],), translated["description"] + (translated["full"],)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for arow, brow in zip(got["state"], translated["state"]):
        for a, b in zip(arow, brow):
            assert {t-offset for t in a} == set(b)
    _, _, state = module.prepare(u, mass, times, owners)
    idle_mass = torch.zeros_like(mass, requires_grad=True)
    description, _, idle = module.prepare(u, idle_mass, times+10_000, owners, state)
    for arow, brow in zip(cache_contents(idle), cache_contents(state)):
        for a, b in zip(arow, brow):
            assert a.keys() == b.keys()
            for time in a:
                for actual, expected in zip(a[time], b[time]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.count_nonzero(value) == 0 for value in description[1:])


@pytest.mark.parametrize("name", CANDIDATES)
def test_candidate_role_width_binding_and_roundtrip(name):
    config = candidate_configs()[name]
    assert ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    model = TideModel(config)
    pairs = {"x": ("K", "S"), "all_ssd": ("S", "S"), "reverse": ("S", "K")}
    ordinary, hub = pairs.get(name, ("K", "S"))
    for node in model.graph.nodes:
        assert model.kinds[node.id] == (hub if node.role == "H" else ordinary)
    for kind, memory in model.memories.items():
        assert memory.ffn.gate.out_features == 4 * config.d
        assert memory.summary.out_features == config.d // 4
        if kind == "K":
            width = config.kv_width or config.d
            assert memory.heads == width // 32
            assert memory.kv.out_features == 2 * width
            assert memory.output.in_features == width
            assert len(memory.query.heads) == width // 32
            assert all(head.in_features == config.d and head.out_features == 32 for head in memory.query.heads)


@pytest.mark.parametrize("options", [dict(kv_width=0), dict(kv_width=True), dict(kv_width=33),
                                     dict(kv_width=256), dict(ffn_width=7), dict(ffn_width=True),
                                     dict(memory="S", kv_width=32), dict(memory_sharing="layer_shared")])
def test_invalid_extension_dimensions_and_ownership_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(d=192, **options)


def test_no_kv_graph_rejects_unused_kv_width():
    with pytest.raises(ValueError, match="unused"):
        TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=1), d=192, memory="X_reverse", kv_width=32))


def test_fixed_ffn_alternative_width_is_serialized_and_both_memories_use_it():
    config = replace(candidate_configs()["x"], ffn_width=800)
    model = TideModel(ModelConfig.from_dict(config.to_dict()))
    assert all(memory.ffn.gate.out_features == 800 and memory.ffn.down.in_features == 800
               for memory in model.memories.values())


@pytest.mark.parametrize("name", ["reverse", "memory_shared", "all_ssd", "kv_sixth"])
def test_integrated_cache_ledger_checks_real_writes_eviction_and_ssd_clocks(name):
    record = cache_metadata_witness(candidate_configs()[name], "cpu", "fp32")
    assert record["state"] == "within_threshold", record
    if name != "all_ssd":
        assert record["kv_eviction_required"]
        assert record["diagnosis"]["kv_owner_samples_with_verified_eviction"] > 0


def test_larger_window_does_not_claim_eviction_from_a_short_continuation():
    config = candidate_configs()["kv_sixth"]
    config = replace(config, clock=replace(config.clock, kv_window=10000))
    record = cache_metadata_witness(config, "cpu", "fp32")
    assert record["state"] == "within_threshold"
    assert not record["kv_eviction_required"]
    assert record["diagnosis"]["kv_owner_samples_with_verified_eviction"] == 0


@pytest.mark.parametrize("actual,expected,state", [
    ({"strengths": torch.tensor([1.])}, {"strengths": torch.tensor([0.])}, "support_failed"),
    ({"times": torch.tensor([2])}, {"times": torch.tensor([1])}, "metadata_failed"),
    (torch.ones(1, dtype=torch.bfloat16), torch.ones(1), "structural_failed"),
    (torch.tensor([float("nan")]), torch.ones(1), "nonfinite"),
])
def test_diagnostic_aggregate_retains_support_metadata_dtype_and_nan_failures(actual, expected, state):
    result = comparison(actual, expected, THRESHOLDS["fp32"])
    summary = comparison_summary({"old_harness": {"state": "finite_deviation", "checks": [{"comparison": result}]}})
    assert summary["state"] == state
    assert aggregate(["finite_deviation", summary["state"]]) == state
    assert state in HARD_FAILURES


def test_diagnostic_aggregate_rejects_unknown_or_unfinished_states():
    with pytest.raises(ValueError, match="unknown"):
        aggregate(["running"])
