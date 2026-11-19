"""Independent small witnesses for matching, effective touches and timing inputs."""
import math

import pytest
import torch

from tide import ModelConfig, TideModel
from tide.topology import TopologyConfig
from benchmarks.baselines import TransformerBaseline, parameter_count
from benchmarks.matching import match_transformers, match_transformer, parameter_memory_budget
from benchmarks.statistics import ParameterInventory, audit_model
from benchmarks.workloads import activity_fixture, paired_input, parameter_hash


@pytest.mark.parametrize("target", [250000, 988960, 14000000, 9000000000])
def test_four_joint_matches_count_real_parameters(target):
    matches = match_transformers(32, 4, target, .41*target)
    for name, row in matches.items():
        assert row["targets"]["blocks"] == (32 if name in "AB" else 4)
        assert row["targets"]["parameters"] == (target if name in "AC" else .41*target)
        with torch.device("meta"):
            actual = TransformerBaseline(**row["constructor"])
        assert parameter_count(actual) == row["actual"]["parameters"]
        assert row["relative_deviation"]["blocks"] == 0
        assert math.isclose(row["relative_deviation"]["parameters"],
                            (parameter_count(actual)-row["targets"]["parameters"])/row["targets"]["parameters"])


def test_matching_is_nearest_in_declared_small_candidate_grid():
    target, blocks = 988960, 32
    result = match_transformer(blocks, target)
    errors = []
    # Independent exhaustive enumeration, including all interior FFN widths.
    for d in range(32, 160, 16):
        for ffn in range(2*d, 6*d+1, 8):
            count = blocks*(4*d*d+3*d*ffn+2*d)+d
            errors.append(abs(count-target))
    assert abs(result["actual"]["parameters"]-target) == min(errors)


def test_memory_budget_does_not_halve_fp32_optimizer_for_bf16():
    for precision in ("fp32", "bf16"):
        result = parameter_memory_budget(1000, mode="train", precision=precision)
        assert result["persistent_parameter_floor_bytes"] == 16000
        assert result["foreach_scratch_allowance_bytes"] == 4000


def test_width_changed_inputs_share_actual_feature_prefix():
    narrow = paired_input(2, 3, 32, 13)
    wide = paired_input(2, 3, 48, 13)
    assert torch.equal(narrow, wide[..., :32])


@pytest.mark.parametrize("sharing", ["independent", "region_shared", "layer_shared"])
def test_inventory_all_parameters_and_dense_effective_union(sharing):
    torch.manual_seed(17)
    model = TideModel(ModelConfig(d=64, sharing=sharing, allocation="pool"))
    inputs = paired_input(2, 2, 64, 18)
    fingerprint = parameter_hash(model)
    with torch.no_grad():
        expected = model(inputs).readout()
        result, tail, audit = audit_model(model, inputs)
    assert tail is None
    assert torch.equal(expected, result.readout())
    assert parameter_hash(model) == fingerprint
    assert audit["parameters"]["unique_parameters"] == parameter_count(model)
    # Every node and every configured source participates at each FC input.
    assert all(row["parameters"] == parameter_count(model)
               for row in audit["effective_touched"]["by_sample_time"])
    assert audit["effective_touched"]["unique_across_time_and_batch"] == parameter_count(model)
    assert audit["parameters"]["textbook"]["active_proxy"] == parameter_count(model)-64*64
    assert audit["work"]["full_actions"] == 32*2*2
    assert "forward" not in vars(model.receive)
    assert "_run" not in vars(model)


@pytest.mark.parametrize("fixture,selected", [("sparse03", 1), ("sparse10", 3), ("high", 32)])
def test_sparse_actual_touches_match_independent_formula(fixture, selected):
    d, nodes_per_region, depth = 64, 32, 4
    model = TideModel(ModelConfig(topology=TopologyConfig(n=nodes_per_region, layers=depth), d=d, budget=1))
    description = activity_fixture(model, fixture)
    assert description["legal_mass_budget"] == 1
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(1, 2, d, 33))
    n = nodes_per_region*depth
    # Fixed math: Agg A/B + source embedding, KV+summary, descriptor,
    # regional network and ingress all prepare. Full uses Q/O and 3 FFN
    # matrices only at the chosen nodes. Enumerate the mathematical shapes.
    prepare_node = 2*d*(d//8)+(d//8)+2*d*d+d*(d//4)+(2*d+3)*(d//2)+2*(d//2)+1
    region = 4*(d//2)**2+2*d*(d//2)+d+2*(d//2)
    full_node = d*d+d*d+3*d*(4*d)
    expected = d*d+n*prepare_node+depth*region+depth*selected*full_node
    assert audit["effective_touched"]["mean"] == expected
    assert audit["work"]["candidate_updates"] == n*2
    assert audit["work"]["full_actions"] == depth*selected*2
    proxy = depth*region+(1/nodes_per_region)*n*(prepare_node+full_node)
    assert audit["parameters"]["textbook"]["active_proxy"] == proxy
    assert audit["effective_touched"]["mean"] > proxy


def test_shared_private_full_contributions_are_deduplicated_once_per_step():
    d, depth, n = 64, 4, 8
    model = TideModel(ModelConfig(d=d, sharing="region_shared", budget=1))
    activity_fixture(model, "sparse03")
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(2, 3, d, 19))
    # All four regions use each Full base once; one node per region uses its
    # own per-head query and FFN adapters. The private states are not shared.
    rank = 8
    private_full = (d//32)*rank*(32+d)+rank*(d+d)+3*rank*(d+4*d)
    full_base = 14*d*d
    assert audit["stages"]["full"]["mean"] == depth*(full_base+private_full)
    assert audit["stages"]["full"]["sum"] == 6*depth*(full_base+private_full)


def test_timed_work_reports_ticks_and_cumulative_inputs_separately():
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-7", timing="timed"),
                         d=64, sharing="region_shared", memory="X")
    model = TideModel(config)
    with torch.no_grad():
        _, tail, audit = audit_model(model, paired_input(1, 2, 64, 91), flush=True)
    assert tail is not None
    assert audit["input_positions"] == 2
    assert audit["logical_axis"] == "tick"
    assert audit["sample_time_slots"] == tail.state.position*model.graph.period
    assert audit["effective_touched"]["cumulative_per_input_position"] == audit["effective_touched"]["sum"]/2
    assert audit["effective_touched"]["unique_across_time_and_batch"] <= parameter_count(model)


def test_activity_fixture_rejects_nonquad_function():
    model = TideModel(ModelConfig(d=64, allocation="shannon"))
    with pytest.raises(ValueError, match="quad"):
        activity_fixture(model, "sparse03")
