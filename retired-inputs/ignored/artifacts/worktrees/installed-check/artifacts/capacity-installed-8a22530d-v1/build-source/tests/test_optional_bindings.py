"""Optional current HB bindings: formulas, ownership and real continuation."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from tide import (ClockConfig, ModelConfig, TideModel, TopologyConfig, build_topology,
                  detach_state, edge_delay_pair, load_checkpoint, save_checkpoint)
from benchmarks.statistics import ParameterInventory
from test_parameter_binding import CAPACITY, independent_inventory
from test_timed_reference import _check_state, _close, _loss
from oracle_timed import evaluate
import test_kv_memory as kv_reference
import test_ssd_memory as ssd_reference


EXPLICIT = ClockConfig(position_period=7, position_origin="absolute", kv_window=11,
                       short_decay=.09, ssd_stride=3)


@pytest.mark.parametrize("kwargs", [
    {"kv_window": 0}, {"kv_window": True}, {"kv_window": 1.5}, {"kv_window": 2**63},
    {"ssd_stride": 0}, {"ssd_stride": 2**62}, {"position_period": 0},
    {"position_period": float("inf")}, {"position_period": True},
    {"short_decay": -.1}, {"short_decay": float("nan")}, {"position_origin": "unknown"},
])
def test_invalid_local_clocks_fail(kwargs):
    with pytest.raises(ValueError):
        ClockConfig(**kwargs)


def test_strict_pair_requires_explicit_identical_local_clocks():
    base = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=2),
                       d=64, memory="X", clock=EXPLICIT)
    settle, timed = edge_delay_pair(base, period=6)
    assert replace(settle.topology, timing="timed") == timed.topology
    assert replace(settle, topology=timed.topology) == timed
    graphs = [build_topology(config.topology) for config in (settle, timed)]
    assert graphs[0].nodes == graphs[1].nodes and graphs[0].period == graphs[1].period == 6
    assert [(e.src, e.dst, e.source_class) for e in graphs[0].edges] == [
        (e.src, e.dst, e.source_class) for e in graphs[1].edges]
    assert {e.delay for e in graphs[0].edges} == {1, 2, 4}
    assert {e.delay for e in graphs[1].edges} == {1}
    models = []
    for config in (settle, timed):
        torch.manual_seed(712)
        model = TideModel(config)
        models.append(model)
        assert model.clock == EXPLICIT
        assert model.memories["K"].position_period == 7
        assert model.memories["K"].window == 11
        assert model.memories["K"].short_decay == model.memories["S"].short_decay == .09
        assert model.memories["S"].stride == 3
    assert models[0].memories["K"].slots == 2  # ceil(11 / actual input interval 6).
    assert models[1].memories["K"].slots == 11
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert ModelConfig.from_dict(json.loads(json.dumps(timed.to_dict()))) == timed
    for invalid in (replace(base, clock=ClockConfig()), replace(base, clock=replace(EXPLICIT, position_origin="binding")),
                    replace(base, topology=TopologyConfig())):
        with pytest.raises(ValueError):
            edge_delay_pair(invalid, period=6)
    with pytest.raises(ValueError):
        edge_delay_pair(base, period=5)


@pytest.mark.parametrize("kind,partition,mask", [
    ("exchange", "group", "all"), ("multiscale", "group", "all"),
    ("multiscale", "singleton", "all"), ("multiscale", "mixed", "all"),
    ("multiscale", "group", "checkerboard"),
])
def test_terminal_hubs_match_current_upstream_constructor(kind, partition, mask):
    topology = TopologyConfig(kind=kind, n=8, hubs=4, partition=partition, hub_mask=mask,
                              terminal_hubs=True, **({"depth": 1} if kind == "exchange" else {"profile": "compact-5-dd4"}))
    graph = build_topology(topology)
    reference = (CAPACITY.st.construct(graph.widths, n=8, h=4, terminal_hubs=True) if kind == "exchange" else
                 CAPACITY.ms.construct(graph.widths, n=8, h=4, placement=mask, add_dd4=True, terminal_hubs=True))
    nodes, _, outgoing = reference[:3]
    addresses = [(v.layer, v.position, v.role.lower(), v.local) for v in graph.nodes]
    assert addresses == nodes
    assert {(e.src, e.dst, e.source_class) for e in graph.edges} == {
        (source, destination, label) for source, edges in enumerate(outgoing) for destination, label in edges}
    assert all(graph.nodes[v].role == "D" for v in graph.output_nodes)
    assert any(v.role == "H" and v.layer == len(graph.widths)-1 for v in graph.nodes)
    without = build_topology(replace(topology, terminal_hubs=False))
    kept = {(v.layer, v.position, v.role, v.local) for v in without.nodes}
    def edge_addresses(g):
        address = [(v.layer, v.position, v.role, v.local) for v in g.nodes]
        return {(address[e.src], address[e.dst], e.source_class) for e in g.edges}
    assert edge_addresses(without) == {e for e in edge_addresses(graph) if e[0] in kept and e[1] in kept}


@pytest.mark.parametrize("sharing", ["independent", "region_shared", "layer_shared"])
@pytest.mark.parametrize("memory", ["K", "X"])
def test_group_fixed_pass_has_independent_parameter_count_and_no_unused_scores(sharing, memory):
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=8, hubs=4,
                                                terminal_hubs=True),
                         d=64, memory=memory, sharing=sharing, aggregation="incoming_light_64", hub_allocation="fixed_pass")
    model = TideModel(config)
    expected = independent_inventory("multiscale-compact-5-dd4", config)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert ParameterInventory(model).report()["unique_parameters"] == expected
    assert all(model.graph.nodes[node].role == "D" for node in model.description_ids)
    assert all((model.region_group[r] < 0) == all(model.graph.nodes[v].role == "H" for v in members)
               for r, members in enumerate(model.graph.regions))
    bounded = replace(config, hub_allocation="bounded")
    assert expected < independent_inventory("multiscale-compact-5-dd4", bounded)


@pytest.mark.parametrize("partition,fixed", [("group", True), ("group", False), ("singleton", False), ("mixed", False)])
def test_terminal_hubs_and_optional_clocks_against_chronological_values_state_vjp(partition, fixed):
    torch.manual_seed(9983)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=2,
                         timing="timed", period=6, partition=partition, terminal_hubs=True),
                         d=64, clock=EXPLICIT, hub_allocation="fixed_pass" if fixed else "bounded",
                         attention_backend="math", attention_chunk=3)
    model = TideModel(config).double()
    inputs = torch.randn(2, 2, 64, dtype=torch.float64, requires_grad=True)
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    reference_inputs = inputs.detach().clone().requires_grad_()
    expected = evaluate(parameters, reference_inputs, model.graph, fixed_hubs=fixed, clock=asdict(EXPLICIT))
    actual = model(inputs)
    assert actual.values.shape[0] == len(model.graph.output_nodes)
    _close(actual.values, expected.values, "terminal-H port values")
    _close(actual.strengths, expected.strengths, "terminal-H port support")
    _check_state(actual.state, expected.state, model.graph)
    actual_gradients = torch.autograd.grad(_loss(actual), (inputs, *model.parameters()), allow_unused=True)
    expected_gradients = torch.autograd.grad(_loss(expected), (reference_inputs, *parameters.values()), allow_unused=True)
    for a, b in zip(actual_gradients, expected_gradients):
        assert (a is None) == (b is None)
        if a is not None:
            _close(a, b, "terminal-H independent VJP")


def test_custom_kv_clock_literal_values_gradients_and_window_continuation():
    args, roots = kv_reference.fixture(position_period=7, window=11, short_decay=.09)
    module, u, mass, times, owners, initial, control = args
    assert module.slots == 3
    actual, expected = kv_reference.actual(*args), kv_reference.formula(*args)
    kv_reference.assert_result(actual, expected)
    ag = torch.autograd.grad(kv_reference.loss(actual), roots, retain_graph=True)
    eg = torch.autograd.grad(kv_reference.loss(expected), roots)
    for a, b in zip(ag, eg):
        torch.testing.assert_close(a, b, atol=3e-9, rtol=3e-10)
    left, context, state = module.prepare(u[:, :, :3], mass[:, :, :3], times[:, :, :3], owners, initial)
    first = module.finish(context, control[:, :, :3])
    right, context, state = module.prepare(u[:, :, 3:], mass[:, :, 3:], times[:, :, 3:], owners, state)
    split = (tuple(torch.cat((a, b), 2) for a, b in zip(left, right)),
             torch.cat((first, module.finish(context, control[:, :, 3:])), 2), kv_reference.decoded(state))
    kv_reference.assert_result(split, expected)


@pytest.mark.parametrize("settle", [False, True])
def test_custom_ssd_clock_literal_values_state_and_all_gradients(settle):
    args = ssd_reference.fixture(shared=True, settle=settle, short_decay=.09, conv_stride=3)
    module, u, mass, _, _, initial, _ = args
    actual, expected = ssd_reference.evaluated(*args), ssd_reference.literal(*args)
    ssd_reference.assert_result(actual, expected)
    roots = ssd_reference.inputs(module, u, mass, initial)
    ag = torch.autograd.grad(ssd_reference.loss(actual), roots, retain_graph=True)
    eg = torch.autograd.grad(ssd_reference.loss(expected), roots)
    for a, b in zip(ag, eg):
        torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-10)


@pytest.mark.parametrize("timing", ["settle", "timed"])
def test_optional_binding_three_updates_restore_and_real_next_update(tmp_path, timing):
    torch.manual_seed(291)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4", n=2, hubs=2,
                         timing=timing, period=6, terminal_hubs=True), d=64, memory="X",
                         hub_allocation="fixed_pass", sharing="region_shared", clock=EXPLICIT,
                         attention_backend="math", attention_chunk=2)
    model = TideModel(config).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = model(torch.randn(1, 2, 64, dtype=torch.float64), state)
        result.readout().square().mean().backward()
        optimizer.step()
        state = detach_state(result.state)
    path = tmp_path/"optional.pt"
    save_checkpoint(path, model, state=state, optimizer=optimizer)
    restored, saved, slots = load_checkpoint(path)
    assert restored.config == config and restored.clock == model.clock
    other = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    other.load_state_dict(slots)
    x = torch.randn(1, 2, 64, dtype=torch.float64)
    torch.testing.assert_close(model(x, state).readout(), restored(x, saved).readout(), atol=0, rtol=0)
    before = [p.detach().clone() for p in model.parameters()]
    for m, o, s in ((model, optimizer, state), (restored, other, saved)):
        o.zero_grad(set_to_none=True)
        result = m(x, s)
        loss = result.readout().square().mean()
        if timing == "timed":
            loss = loss + m.flush(result.state).readout().square().mean()
        loss.backward(); o.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state_dict(), other.state_dict(), atol=0, rtol=0)


def test_optional_binding_rejections_and_old_config_defaults():
    for topology in (TopologyConfig(), TopologyConfig(kind="multiscale", hubs=4, partition="mixed"),
                     TopologyConfig(kind="multiscale", hubs=4, hub_mask="none")):
        with pytest.raises(ValueError):
            ModelConfig(topology=topology, hub_allocation="fixed_pass")
    for topology in (TopologyConfig(terminal_hubs=True), TopologyConfig(kind="multiscale", terminal_hubs=True),
                     TopologyConfig(kind="exchange", hubs=4, terminal_hubs=1)):
        with pytest.raises(ValueError):
            build_topology(topology)
    old = ModelConfig().to_dict()
    del old["clock"], old["hub_allocation"], old["topology"]["terminal_hubs"]
    assert ModelConfig.from_dict(old) == ModelConfig()
