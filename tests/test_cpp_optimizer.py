"""Direct LibTorch optimizer parity and named-owner contract checks."""

import pytest
import torch

import _tide_native as core


def _group(names, **values):
    group = core.OptimizerGroup()
    group.parameters = list(names)
    for name, value in values.items():
        setattr(group, name, value)
    return group


def _parameters(dtype):
    values = [
        torch.tensor([1.0, -2.0], dtype=dtype).requires_grad_(),
        torch.tensor([0.5], dtype=dtype).requires_grad_(),
        torch.tensor([0.25], dtype=dtype).requires_grad_(),
    ]
    return values


def _set_grads(parameters, values):
    for parameter, value in zip(parameters, values):
        parameter.grad = None if value is None else torch.as_tensor(value, dtype=parameter.dtype).clone()


def _assert_same(cpp, reference, dtype):
    tolerance = dict(rtol=1e-8, atol=1e-10) if dtype is torch.float64 else dict(rtol=1e-5, atol=1e-6)
    for actual, expected in zip(cpp, reference):
        torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
def test_libtorch_named_optimizer_matches_pytorch(dtype, kind):
    reference = _parameters(dtype)
    native = [value.detach().clone().requires_grad_() for value in reference]
    registry = core.ParameterRegistry()
    registry.add("body.nodes.0.weight", native[0])
    registry.add("readout.nodes.0.weight", native[0])  # Cross-graph alias.
    registry.add("body.input_scale.0", native[1])
    registry.add("body.unused", native[2])
    assert registry.canonical_name("readout.nodes.0.weight") == "body.nodes.0.weight"
    assert ["body.nodes.0.weight", "readout.nodes.0.weight"] in registry.alias_partitions()

    if kind == "sgd":
        values = dict(lr=.1, momentum=.8, weight_decay=.02)
        optimizer = core.SGD(registry, [_group(
            ["readout.nodes.0.weight", "body.input_scale.0", "body.unused"], **values)])
        reference_optimizer = torch.optim.SGD(reference, **values)
    else:
        values = dict(lr=.01, weight_decay=.1, beta1=.9, beta2=.999, eps=1e-5, amsgrad=True)
        optimizer = core.AdamW(registry, [_group(
            ["readout.nodes.0.weight", "body.input_scale.0", "body.unused"], **values)])
        reference_optimizer = torch.optim.AdamW(
            reference, lr=values["lr"], weight_decay=values["weight_decay"],
            betas=(values["beta1"], values["beta2"]), eps=values["eps"], amsgrad=True)

    assert optimizer.layout().groups == [[
        "body.nodes.0.weight", "body.input_scale.0", "body.unused"]]
    gradients = [
        ([1.0, 1.0], [0.0], None),
        ([-.4, .2], [1.0], None),
        ([0.0, 0.0], None, [0.0]),
    ]
    for step, values_for_step in enumerate(gradients):
        _set_grads(reference, values_for_step)
        _set_grads(native, values_for_step)
        reference_optimizer.step()
        optimizer.step()
        _assert_same(native, reference, dtype)
        if step == 0:
            assert "body.unused" not in optimizer.state()
        optimizer.zero_grad()
        reference_optimizer.zero_grad(set_to_none=True)

    # The owner is absent for the first two objectives, then connected by a zero
    # gradient and therefore acquires the same state as PyTorch.
    state = optimizer.state()
    assert "body.unused" in state if kind == "sgd" else "body.unused" in state
    if kind == "sgd":
        assert state["body.unused"].momentum_buffer is not None
    else:
        assert state["body.unused"].exp_avg is not None


def test_libtorch_registry_rejects_duplicate_owner_and_foreign_optimizer(dtype):
    value = torch.ones(2, dtype=dtype).requires_grad_()
    outside = torch.ones(2, dtype=dtype).requires_grad_()
    registry = core.ParameterRegistry()
    registry.add("a", value)
    registry.add("alias", value)
    registry.add("outside", outside)
    with pytest.raises((RuntimeError, ValueError), match="duplicate"):
        registry.add("a", value)
    group = _group(["a", "foreign"], lr=.1)
    with pytest.raises((RuntimeError, ValueError), match="unknown|parameter"):
        core.SGD(registry, [group])
    optimizer = core.SGD(registry, [_group(["a"], lr=.1)])
    value.grad = torch.ones_like(value)
    outside.grad = torch.ones_like(outside)
    optimizer.zero_grad()
    assert value.grad is None and outside.grad is not None
