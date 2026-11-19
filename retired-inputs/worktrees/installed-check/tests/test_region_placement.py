"""Whole-region ownership, causal banks, gradients and restored real updates."""
import copy

import pytest
import torch

from benchmarks.continuous import nested_comparison
from tide import (ModelConfig, TopologyConfig, TideModel, canonical_state, canonical_state_dict,
                  detach_state, load_checkpoint, place_model, placement_adamw, placement_report, save_checkpoint)
from tide.precision import precision_context
from tide.region_placement import RegionLayerOutput


def config(sharing="region_shared", timing="settle", partition="group", recompute=False):
    return ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4",
                                              timing=timing, partition=partition),
                       d=64, memory="X", sharing=sharing, aggregation="incoming_light_64",
                       checkpoint_regions=recompute)


def layout(model):
    # Place spatial positions alternately, including two regions of the same
    # role/kind in the wide layer. Both cards retain some complete regions.
    return [(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer) % 2
            for nodes in model.graph.regions]


def objective(result):
    value = result.readout()
    weight = torch.cos(torch.arange(value.numel(), device=value.device, dtype=value.dtype)).reshape_as(value)
    return (value*weight).mean()


def full(model, x, state=None):
    result = model(x, state)
    tail = None if model.settle else model.flush(result.state)
    return result, tail


@pytest.mark.parametrize("sharing,timing,partition", [
    ("independent", "settle", "group"), ("region_shared", "settle", "mixed"),
    ("independent", "timed", "singleton"), ("region_shared", "timed", "group")])
@pytest.mark.parametrize("concurrency", ["serial", "per_device"])
def test_region_placement_whole_split_vjp_state_and_flush(sharing, timing, partition, concurrency):
    torch.manual_seed(1405)
    reference = TideModel(config(sharing, timing, partition)).double()
    placed = place_model(copy.deepcopy(reference), ["cpu:0", "cpu:1"], region_partitions=layout(reference),
                         region_concurrency=concurrency)
    assert placement_report(placed)["kind"] == "hb-region-shards-1"
    torch.testing.assert_close(canonical_state_dict(placed), reference.state_dict(), atol=0, rtol=0)
    x = torch.randn(2, 4, 64, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = reference(x, record_work=True)
    actual = placed(y, record_work=True)
    torch.testing.assert_close(actual.values, expected.values, atol=0, rtol=0)
    torch.testing.assert_close(actual.strengths, expected.strengths, atol=0, rtol=0)
    assert actual.work == expected.work
    assert nested_comparison(canonical_state(placed, actual.state), expected.state, atol=0, rtol=0)["exact"]
    objective(expected).backward(); objective(actual).backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(canonical_state_dict(placed, gradients=True),
                               {name: parameter.grad for name, parameter in reference.named_parameters()},
                               atol=3e-10, rtol=3e-8)
    first = placed(y[:, :2]); second = placed(y[:, 2:], first.state)
    torch.testing.assert_close(torch.cat((first.values, second.values), 2), expected.values, atol=2e-9, rtol=2e-9)
    assert nested_comparison(canonical_state(placed, second.state), expected.state, atol=2e-9, rtol=2e-9)["within_threshold"]
    if timing == "timed":
        assert all(isinstance(bank, RegionLayerOutput) for bank in actual.state.pending)
        for bank in actual.state.pending:
            for chunk in bank.regions:
                for tensor in (chunk.values, chunk.strengths):
                    assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()
        a, b = reference.flush(expected.state), placed.flush(second.state)
        torch.testing.assert_close(a.values, b.values, atol=2e-9, rtol=2e-9)
        assert nested_comparison(canonical_state(placed, b.state), a.state, atol=2e-9, rtol=2e-9)["within_threshold"]


def test_regions_share_one_input_projection_and_deduplicate_source_nodes():
    torch.manual_seed(812)
    c = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=0, profile="compact-9-two-inputs"),
                    d=64, memory="K", sharing="independent")
    reference = TideModel(c).double()
    owners = layout(reference)
    placed = place_model(copy.deepcopy(reference), ["cpu", "cpu"], region_partitions=owners)
    assert len(list(placed.inputs.parameters())) == len(list(reference.inputs.parameters())) == 1
    assert len({owners[region] for region in placed.layer_regions[0]}) == 2
    assert len({owners[region] for region in placed.layer_regions[-1]}) == 2
    x = torch.randn(1, 2, 64, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected, actual = reference(x), placed(y)
    torch.testing.assert_close(expected.values, actual.values, atol=0, rtol=0)
    objective(expected).backward(); objective(actual).backward()
    torch.testing.assert_close(x.grad, y.grad, atol=3e-10, rtol=3e-8)
    torch.testing.assert_close(reference.inputs["D"].weight.grad, placed.inputs["D"].weight.grad,
                               atol=3e-10, rtol=3e-8)
    duplicate_consumers = False
    for layer, destinations in enumerate(placed._region_transfers):
        for destination, origins in destinations.items():
            for origin, plan in origins.items():
                sources = [placed.graph.edges[edge].src
                           for region in placed.layer_regions[layer] if owners[region] == destination
                           for node in placed.graph.regions[region] for edge in placed.graph.incoming[node]
                           if placed.graph.nodes[placed.graph.edges[edge].src].layer == origin]
                assert plan["nodes"] == tuple(sorted(set(sources)))
                duplicate_consumers |= len(sources) > len(plan["nodes"])
    assert duplicate_consumers


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("concurrency", ["serial", "per_device"])
def test_region_three_updates_checkpoint_and_actual_restored_timed_update(tmp_path, precision, concurrency):
    torch.manual_seed(63)
    reference = TideModel(config("region_shared", "timed", recompute=True))
    placed = place_model(copy.deepcopy(reference), ["cpu:0", "cpu:1"], region_partitions=layout(reference),
                         region_concurrency=concurrency)
    optimizers = [torch.optim.AdamW(reference.parameters(), lr=1e-4, foreach=False),
                  placement_adamw(placed, lr=1e-4, foreach=False)]
    states = [None, None]
    trajectory = []
    x = torch.randn(1, 2, 64)
    for _ in range(3):
        for index, (model, optimizer) in enumerate(zip((reference, placed), optimizers)):
            optimizer.zero_grad(set_to_none=True)
            with precision_context("cpu", precision):
                result, tail = full(model, x, states[index])
                loss = objective(result)+objective(tail)
            loss.backward(); optimizer.step()
            states[index] = detach_state(tail.state)
        comparison = nested_comparison(canonical_state_dict(placed), reference.state_dict(), atol=1e-6, rtol=2e-3)
        trajectory.append(comparison)
        # The exact thresholds and finite deviations belong to numerical
        # qualification. This mechanism test rejects structural/nonfinite
        # errors and separately requires a real restored update below.
        assert comparison["finite"] and not comparison["metadata_mismatches"]
        assert states[0].position == states[1].position
    from scripts.job import atomic
    atomic(tmp_path/"trajectory.json", dict(seed=63, precision=precision, parameters=trajectory,
           state="within_threshold" if all(row["within_threshold"] for row in trajectory) else "finite_deviation"))
    path = tmp_path/"regions.pt"
    save_checkpoint(path, placed, state=states[1], optimizer=optimizers[1])
    restored, state, slots = load_checkpoint(path, placement_devices=["cpu:1", "cpu:0"])
    resumed = placement_adamw(restored, lr=1e-4, foreach=False); resumed.load_state_dict(slots)
    assert placement_report(restored)["region_partitions"] == layout(reference)
    assert placement_report(restored)["region_concurrency"] == concurrency
    assert placement_report(restored)["devices"] == ["cpu:1", "cpu:0"]
    assert nested_comparison(restored.state_dict(), placed.state_dict(), atol=0, rtol=0)["exact"]
    assert nested_comparison(state, states[1], atol=0, rtol=0)["exact"]
    assert nested_comparison(resumed.state_dict(), optimizers[1].state_dict(), atol=0, rtol=0)["exact"]
    updated_states = []
    for model, optimizer, current in ((placed, optimizers[1], states[1]), (restored, resumed, state)):
        optimizer.zero_grad(set_to_none=True)
        with precision_context("cpu", precision):
            result, tail = full(model, x, current)
            loss = objective(result)+objective(tail)
        loss.backward(); optimizer.step()
        updated_states.append(detach_state(tail.state))
    assert nested_comparison(updated_states[1], updated_states[0], atol=1e-6, rtol=2e-3)["within_threshold"]
    assert nested_comparison(restored.state_dict(), placed.state_dict(), atol=1e-6, rtol=2e-3)["within_threshold"]
    assert nested_comparison(resumed.state_dict(), optimizers[1].state_dict(), atol=1e-6, rtol=2e-3)["within_threshold"]


def test_region_shared_crossing_rejected_before_any_mutation():
    model = TideModel(config("layer_shared"))
    identities = [id(parameter) for parameter in model.parameters()]
    with pytest.raises(ValueError, match="shared parameter .* spans region partitions"):
        place_model(model, ["cpu", "cpu"], region_partitions=layout(model))
    assert identities == [id(parameter) for parameter in model.parameters()]
    assert not hasattr(model, "_placement_spec")
    with pytest.raises(ValueError, match="mutually exclusive"):
        place_model(model, ["cpu", "cpu"], region_partitions=layout(model), layer_partitions=[0, 0, 0, 1, 1])


def test_region_adamw_original_parameter_none_zero_decay_semantics():
    torch.manual_seed(31)
    reference = TideModel(config("independent")).double()
    placed = place_model(copy.deepcopy(reference), ["cpu", "cpu"], region_partitions=layout(reference))
    a = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.2, foreach=False)
    b = placement_adamw(placed, lr=.01, weight_decay=.2, foreach=False)
    name = "memories.K.kv"
    parameter = dict(reference.named_parameters())[name+".weight"]
    wrapper = placed._shard_registry[name]
    for selected in (0, 1, 0):
        a.zero_grad(set_to_none=True); b.zero_grad(set_to_none=True)
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[list(wrapper._maps["weight"][selected])] = .7
        wrapper.shards[selected].weight.grad = torch.full_like(wrapper.shards[selected].weight, .7)
        assert wrapper.shards[1-selected].weight.grad is None
        a.step(); b.step()
        assert not bool(wrapper.shards[1-selected].weight.grad.any())
        torch.testing.assert_close(canonical_state_dict(placed), reference.state_dict(), atol=1e-15, rtol=1e-15)
    for shard in placed._shard_registry["memories.K.summary"].shards:
        assert shard.weight.grad is None and shard.weight not in b.state


@pytest.mark.parametrize("mode", ["qualify", "inference", "train"])
@pytest.mark.parametrize("concurrency", ["serial", "per_device"])
def test_region_cli_uses_existing_record_and_execution_modes(tmp_path, monkeypatch, mode, concurrency):
    import json
    import sys
    from scripts import check_placement
    c = ModelConfig(topology=TopologyConfig(kind="multiscale", n=1, hubs=0, profile="compact-5-dd4"),
                    d=64, memory="K", sharing="independent")
    config_file = tmp_path/"config.json"
    config_file.write_text(json.dumps(c.to_dict()))
    output = tmp_path/"observation.json"
    monkeypatch.setattr(sys, "argv", ["check_placement", "--partition-kind", "region", "--mode", mode,
        "--region-concurrency", concurrency, "--devices", "cpu:0", "cpu:1",
        "--config", str(config_file), "--output", str(output), "--batch", "1", "--length", "2",
        "--steps", "1", "--warmup", "0", "--threads", "1"])
    assert check_placement.main() == 0
    result = json.loads(output.read_text())
    assert result["state"] == "completed"
    row = result["observations"][0]
    assert row["placement"]["kind"] == "hb-region-shards-1"
    assert row["state"] in {"completed", "within_threshold", "finite_deviation"}
    assert row["placement"]["predecessor_transport"]
    assert row["placement"]["region_concurrency"] == concurrency
    assert row["operator_witness"]["execution"]["requested_region_concurrency"] == concurrency
    assert row["operator_witness"]["execution"]["effective_region_concurrency"] == "serial"
    if mode == "qualify":
        assert row["comparisons"]["checkpoint_data"]["exact"]
        assert not row["comparisons"]["whole_result"]["metadata_mismatches"]


def test_terminal_nonoutput_region_stays_disconnected_from_readout():
    from types import SimpleNamespace
    from tide.model import LayerOutput
    from tide.region_placement import _gather_outputs
    model = SimpleNamespace(layer_regions=[(0, 1)],
                            graph=SimpleNamespace(regions=((0, 1), (2,)), output_nodes=(0, 1)))
    data = torch.randn(2, 1, 2, 4, requires_grad=True)
    terminal = torch.randn(1, 1, 2, 4, requires_grad=True)
    bank = RegionLayerOutput((LayerOutput(data, torch.ones(2, 1, 2)),
                              LayerOutput(terminal, torch.ones(1, 1, 2))))
    result = _gather_outputs(model, bank, torch.device("cpu"), "direct")
    torch.testing.assert_close(result.values, data, atol=0, rtol=0)
    result.values.sum().backward()
    torch.testing.assert_close(data.grad, torch.ones_like(data), atol=0, rtol=0)
    assert terminal.grad is None
