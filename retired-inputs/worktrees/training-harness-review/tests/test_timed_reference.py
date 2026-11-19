"""HB-Timed unequal paths, H partitions and continuation against an event queue."""
from dataclasses import replace

import pytest
import torch

from tide.config import ModelConfig
from tide.model import TideModel
from tide.topology import TopologyConfig

from oracle_timed import evaluate


def _close(actual, expected, label):
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-8, msg=label)


def _check_state(actual, expected, graph):
    assert actual.position == expected.position
    for region, members in enumerate(graph.regions):
        memory = actual.memories[region]["K"]
        for row, node in enumerate(members):
            for batch in range(2):
                entries = expected.caches[batch][node]
                positive = memory.weight[row, batch] > 0
                torch.testing.assert_close(memory.times[row, batch, positive],
                                           torch.tensor([entry.time for entry in entries], dtype=torch.int64))
                for name, field in (("keys", "key"), ("values", "value"),
                                    ("features", "feature"), ("weight", "weight")):
                    reference = (torch.stack([getattr(entry, field) for entry in entries]) if entries else
                                 getattr(memory, name)[row, batch, :0])
                    _close(getattr(memory, name)[row, batch, positive], reference,
                           f"private effective state node={node}, batch={batch}: {name}")
    layer_nodes = [[v.id for v in graph.nodes if v.layer == layer] for layer in range(len(graph.widths))]
    for batch in range(2):
        pending = {delivery.edge: delivery for delivery in expected.pending[batch]}
        expected_edges = set()
        for layer, members in enumerate(layer_nodes):
            for row, node in enumerate(members):
                mass = actual.pending[layer].strengths[row, batch, 0]
                for eid in graph.outgoing[node]:
                    if bool(mass > 0):
                        expected_edges.add(eid)
                        assert eid in pending
                        assert pending[eid].time == expected.position * graph.period + graph.edges[eid].delay
                        _close(mass, pending[eid].mass, f"pending edge {eid}: strength")
                        _close(actual.pending[layer].values[row, batch, 0], pending[eid].content,
                               f"pending edge {eid}: content")
        assert set(pending) == expected_edges


def _loss(result):
    probe = torch.sin(torch.arange(result.values.numel(), dtype=torch.float64).reshape_as(result.values) * 0.17)
    return ((result.values * probe).sum() / 37 + result.strengths.square().sum() / 11
            + result.readout().square().sum() / 53)


def _case(partition, period=None):
    torch.manual_seed(9983)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="compact-5-dd4",
                                                n=2, hubs=2, partition=partition, timing="timed", period=period),
                         d=64, memory="K", allocation="quad", sharing="independent",
                         aggregation="source_residual", attention_backend="math", attention_chunk=3)
    model = TideModel(config).double()
    with torch.no_grad():
        model.receive.embedding.weight.normal_(std=0.12)
        model.describe.encode.bias.normal_(std=0.06)
        model.describe.offset.value.normal_(std=0.15)
    inputs = torch.randn(2, 2, 64, dtype=torch.float64, requires_grad=True)
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    reference_inputs = inputs.detach().clone().requires_grad_()
    return model, inputs, parameters, reference_inputs


@pytest.mark.parametrize("partition", ["singleton", "mixed"])
def test_multiscale_timed_chronological_values_state_gradients_and_continuation(partition):
    model, inputs, parameters, reference_inputs = _case(partition)
    graph = model.graph
    assert graph.widths == (1, 1, 2, 1, 1)
    assert {edge.delay for edge in graph.edges} == {1}
    assert {graph.nodes[e.dst].layer - graph.nodes[e.src].layer for e in graph.edges} == {1, 2, 4}
    expected = evaluate(parameters, reference_inputs, graph)
    actual = model(inputs)
    first = model(inputs[:, :1])
    last = model(inputs[:, 1:], first.state)
    continued = replace(last, values=torch.cat((first.values, last.values), 2),
                        strengths=torch.cat((first.strengths, last.strengths), 2),
                        times=torch.cat((first.times, last.times)))
    # One input's lengths 1,2,3,4 give outputs at logical times 2,3,4,5.
    assert bool((expected.strengths[:, :, 1:5].sum(0) > 0).all())
    assert all([entry.time for entry in expected.state.caches[b][graph.input_nodes[0]]] == [1, 5]
               for b in range(2))
    assert any(expected.state.pending), "the fixture must retain messages past the input boundary"
    reference_gradients = torch.autograd.grad(_loss(expected), (reference_inputs, *parameters.values()), allow_unused=True)
    for label, result in (("window", actual), ("continued", continued)):
        _close(result.values, expected.values, f"{label}: raw Full values")
        _close(result.strengths, expected.strengths, f"{label}: controls")
        _close(result.readout(), expected.readout(), f"{label}: output readout")
        torch.testing.assert_close(result.times, expected.times)
        _check_state(result.state, expected.state, graph)
        gradients = torch.autograd.grad(_loss(result), (inputs, *model.parameters()), allow_unused=True)
        for name, grad, reference in zip(("inputs", *parameters), gradients, reference_gradients):
            assert (grad is None) == (reference is None), f"{label}: gradient connectivity {name}"
            if reference is not None:
                _close(grad, reference, f"{label}: gradient {name}")


@pytest.mark.parametrize("partition", ["singleton", "mixed"])
def test_multiscale_timed_flush_drains_without_new_ingress(partition):
    # Four edges at most, period three: draining needs two complete periods.
    model, inputs, parameters, reference_inputs = _case(partition, period=3)
    graph = model.graph
    expected_prefix = evaluate(parameters, reference_inputs, graph)
    actual_prefix = model(inputs)
    drain_periods = 2
    expected = evaluate(parameters, reference_inputs.new_zeros((2, drain_periods, 64)), graph,
                        expected_prefix.state, inject=False)
    actual = model.flush(actual_prefix.state)
    assert actual.values.shape[2] == 6
    assert bool(expected.strengths.sum() > 0), "the finite stream must have a real tail"
    assert not any(expected.state.pending)
    assert all(bool((layer.strengths == 0).all()) for layer in actual.state.pending)
    _close(actual.values, expected.values, "flush: pending Full contents")
    _close(actual.strengths, expected.strengths, "flush: pending strengths")
    _close(actual.readout(), expected.readout(), "flush: pending readout")
    torch.testing.assert_close(actual.times, expected.times)
    _check_state(actual.state, expected.state, graph)
    for batch in range(2):
        for node in graph.input_nodes:
            assert [entry.time for entry in expected.state.caches[batch][node]] == [1, 4]
    actual_complete = replace(actual, values=torch.cat((actual_prefix.values, actual.values), 2),
                              strengths=torch.cat((actual_prefix.strengths, actual.strengths), 2))
    expected_complete = replace(expected, values=torch.cat((expected_prefix.values, expected.values), 2),
                                strengths=torch.cat((expected_prefix.strengths, expected.strengths), 2))
    actual_gradients = torch.autograd.grad(_loss(actual_complete), (inputs, *model.parameters()), allow_unused=True)
    reference_gradients = torch.autograd.grad(_loss(expected_complete),
                                             (reference_inputs, *parameters.values()), allow_unused=True)
    for name, grad, reference in zip(("inputs", *parameters), actual_gradients, reference_gradients):
        assert (grad is None) == (reference is None), f"flush: gradient connectivity {name}"
        if reference is not None:
            _close(grad, reference, f"flush: gradient {name}")
