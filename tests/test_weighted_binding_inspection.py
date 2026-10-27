"""Architecture §1/S.1: inspect the installed functions, without execution.

Every configured scalar/vector is compared to its public construction input.
This checks metadata transport and private copies; numerical formulas and graph
schedules retain their independent tests in test_weighted_*.
"""
from dataclasses import asdict, replace

import pytest
import torch

from tidegraph.native_model import encode_model
from tidegraph.weighted import bind
from test_weighted_execution import setup
from test_weighted_recurrent_execution import model as recurrent_model


def plain(value):
    return [plain(x) for x in value] if isinstance(value, (tuple, list)) else value


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_installed_functions_round_trip_all_config_fields(dtype, family):
    import _tide_native as core
    b, _, _, _ = setup(dtype, "pdg") if family == "kv" else recurrent_model(dtype, family)
    if family == "kv":
        memory = replace(b.memories[0], memory_read="qry", read_frequencies=(.17,),
                         position_origin=17., position_scale=.7, distance_slopes=(.13,))
        b = bind(b.graph, memories=memory, regions=b.regions, alpha_agg=.23, dtype=dtype)
    _, weights = encode_model(core, b.graph, b.model)
    for node, weight in zip(b.model.nodes, weights.nodes):
        snapshot = core.weighted_node_binding(weight)
        for name, value in asdict(node.kernel.config).items():
            assert plain(getattr(snapshot["memory"], name)) == plain(value), name
        regional = asdict(node.read_program.config)
        for name, value in regional.items():
            if name != "read_mode":  # This belongs to Graph::Region, not a local function.
                assert getattr(snapshot["regional"], name) == value, name
        aggregate = node.aggregate_program
        assert snapshot["edge_classes"] == list(aggregate.classes.edges)
        assert snapshot["input_classes"] == list(aggregate.classes.inputs)
        assert snapshot["aggregate_alpha"] == aggregate.alpha
        snapshot["memory"].width = 999
        snapshot["regional"].hidden_width = 999
        snapshot["edge_classes"].clear()
        fresh = core.weighted_node_binding(weight)
        assert fresh["memory"].width == node.kernel.config.width
        assert fresh["regional"].hidden_width == node.read_program.config.hidden_width
        assert fresh["edge_classes"] == list(aggregate.classes.edges)
    for program, weight in zip(b.model.regions, weights.regions):
        snapshot = core.weighted_region_binding(weight)
        for name, value in asdict(program.config).items():
            if name != "read_mode":
                assert getattr(snapshot, name) == value, name
        snapshot.hidden_width = 999
        assert core.weighted_region_binding(weight).hidden_width == program.config.hidden_width


def test_uninstalled_functions_are_rejected():
    import _tide_native as core
    x = torch.zeros(1)
    with pytest.raises(ValueError, match="installed weighted"):
        core.weighted_node_binding(core.NodeWeights(x, x, x, x))
    with pytest.raises(ValueError, match="weighted"):
        core.weighted_region_binding(core.RegionWeights())
