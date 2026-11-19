"""Complete FC/K forward and VJP against a time-major mathematical witness."""
import pytest
import torch

from tide.config import ModelConfig
from tide.model import TideModel
from tide.topology import TopologyConfig

from oracle_fc import evaluate


def _case(allocation="pool", backend="math"):
    torch.manual_seed(7251)
    config = ModelConfig(topology=TopologyConfig(kind="fc", layers=2, n=2), d=64,
                         memory="K", aggregation="source_residual", allocation=allocation,
                         sharing="independent", attention_backend=backend, attention_chunk=2)
    model = TideModel(config).double()
    with torch.no_grad():
        model.receive.embedding.weight.normal_(std=0.15)
        model.describe.encode.bias.normal_(std=0.1)
        model.describe.offset.value.normal_(std=0.2)
    inputs = (0.6 * torch.randn(2, 3, 64, dtype=torch.float64)).requires_grad_()
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    reference_inputs = inputs.detach().clone().requires_grad_()
    return model, inputs, parameters, reference_inputs


def _close(actual, expected, label):
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-9, msg=label)


def _check_state(actual, expected):
    assert actual.position == expected.position
    assert actual.pending == ()
    for layer, region in enumerate(actual.memories):
        memory = region["K"]
        for node in range(2):
            for batch in range(2):
                entries = expected.caches[batch][2 * layer + node]
                assert len(entries) == expected.position
                torch.testing.assert_close(memory.times[node, batch],
                                           torch.tensor([entry.time for entry in entries], dtype=torch.int64))
                for name, field in (("keys", "key"), ("values", "value"),
                                    ("features", "feature"), ("weight", "weight")):
                    _close(getattr(memory, name)[node, batch],
                           torch.stack([getattr(entry, field) for entry in entries]),
                           f"private cache layer={layer}, node={node}, batch={batch}: {name}")


def _loss(result):
    values, strengths = result.values, result.strengths
    value_probe = torch.sin(torch.arange(values.numel(), dtype=values.dtype).reshape_as(values) * 0.173)
    mass_probe = torch.cos(torch.arange(strengths.numel(), dtype=values.dtype).reshape_as(strengths) * 0.47)
    return ((values * value_probe).sum() / 13 + 0.2 * (strengths * mass_probe).sum()
            + 0.03 * result.readout().square().sum())


@pytest.mark.parametrize("allocation", ["pool", "quad"])
@pytest.mark.parametrize("backend", ["math", "auto"])
def test_fc_complete_outputs_private_state_and_all_parameter_gradients(allocation, backend):
    model, inputs, parameters, reference_inputs = _case(allocation, backend)
    actual = model(inputs)
    expected = evaluate(parameters, reference_inputs, allocation=allocation)
    _close(actual.values, expected.values, "unscaled Full contents")
    _close(actual.strengths, expected.strengths, "separate sending strengths")
    _close(actual.readout(), expected.readout(), "port strengths used once by ReadOut")
    torch.testing.assert_close(actual.times, expected.times)
    _check_state(actual.state, expected.state)
    actual_roots = (inputs, *model.parameters())
    expected_roots = (reference_inputs, *parameters.values())
    actual_gradients = torch.autograd.grad(_loss(actual), actual_roots, allow_unused=True)
    expected_gradients = torch.autograd.grad(_loss(expected), expected_roots, allow_unused=True)
    for name, grad, reference in zip(("inputs", *parameters), actual_gradients, expected_gradients):
        assert (grad is None) == (reference is None), f"gradient connectivity: {name}"
        if reference is not None:
            _close(grad, reference, f"gradient: {name}")
    if allocation == "quad":
        assert bool((actual.strengths == 0).any()), "the fixture must exercise skipped Full calls"
    else:
        assert bool(((actual.strengths > 0) & (actual.strengths < 1)).all())


@pytest.mark.parametrize("allocation", ["pool", "quad"])
def test_fc_split_continuation_preserves_original_time_and_private_cache(allocation):
    model, inputs, parameters, reference_inputs = _case(allocation)
    first = model(inputs[:, :1])
    second = model(inputs[:, 1:2], first.state)
    last = model(inputs[:, 2:], second.state)
    expected = evaluate(parameters, reference_inputs, allocation=allocation)
    _close(torch.cat((first.values, second.values, last.values), 2), expected.values, "continued Full contents")
    _close(torch.cat((first.strengths, second.strengths, last.strengths), 2), expected.strengths, "continued strengths")
    torch.testing.assert_close(torch.cat((first.times, second.times, last.times)), expected.times)
    _check_state(last.state, expected.state)
    # Future output gradients must still reach the first input and its KV writes.
    actual_gradients = torch.autograd.grad(last.readout().square().sum(), (inputs, *model.parameters()), allow_unused=True)
    expected_gradients = torch.autograd.grad(expected.readout()[:, -1:].square().sum(),
                                           (reference_inputs, *parameters.values()), allow_unused=True)
    assert bool(expected_gradients[0][:, 0].abs().sum() > 0)
    for name, grad, reference in zip(("inputs", *parameters), actual_gradients, expected_gradients):
        assert (grad is None) == (reference is None), f"continued gradient connectivity: {name}"
        if reference is not None:
            _close(grad, reference, f"continued gradient: {name}")
