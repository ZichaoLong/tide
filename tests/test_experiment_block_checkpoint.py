"""Opaque block archive decoding and complete canonical optimizer layout."""
from types import SimpleNamespace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.experiment.block_checkpoint import record
from tidegraph.experiment.training_layout import SGDLayout
from tidegraph.resident_training_options import GROUP_FIELDS


def actual_checkpoint(momentum=.2):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([.3, -.2]))
    model.alias = model.weight
    model.fixed = torch.nn.Parameter(torch.tensor(.7), requires_grad=False)
    options = dict(lr=.03, momentum=momentum, weight_decay=.04)
    layout = SGDLayout(model, **options)
    optimizer = torch.optim.SGD(model.parameters(), **options)
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model.weight*(i+1)+model.alias.square()).square().sum().backward()
        optimizer.step()
    params = dict(alias=model.weight.detach().clone(), fixed=model.fixed.detach().clone(), inert=torch.tensor([8., 9., 10.]))
    values = torch.cat([params[n].reshape(-1) for n in sorted(params)])
    gradient = SimpleNamespace(names=sorted(params), aliases=[["alias", "weight"], ["fixed"], ["inert"]],
        offsets=[0, 2, 3], values=torch.cat((model.weight.grad.clone(), torch.zeros(4))), connected=torch.tensor([True, False, False]))
    first = torch.cat((optimizer.state[model.weight]["momentum_buffer"].clone(), torch.zeros(4))) if momentum else torch.zeros(1)
    slots = SimpleNamespace(values=values, first=first, second=torch.zeros(6), maximum=torch.zeros(6),
                            steps=torch.tensor([2, 0, 0]), corrections=torch.zeros((3, 2)))
    opts = dict(parameters=["alias"], **options, dampening=0., beta1=.9, beta2=.999, eps=1e-8,
                nesterov=False, amsgrad=False, maximize=False)
    c = SimpleNamespace(schema=2, generation=2, next_token=2, identity="actual-fixed-owner",
        parameters=params, aliases=[["alias", "weight"], ["fixed"], ["inert"]], optimizer=0,
        groups=[SimpleNamespace(**opts)], slots=slots,
        state=[SimpleNamespace(floating=torch.tensor([[2., 17., 33.]]), integer=torch.tensor([[7, 5, 1, 0, 7]]))],
        bound_initial=[SimpleNamespace(floating=torch.tensor([[1., 2., 3.]]), integer=torch.tensor([[-1, 0, 0, 0, 0]]))],
        last_positions=torch.tensor([3]), node_present=torch.tensor([[True]]), region_times=torch.tensor([[7]]),
        input_counts=torch.tensor([[2]]), input_times=torch.tensor([[7]]))
    client = SimpleNamespace(model=model, checkpoint=lambda: c,
                              module=SimpleNamespace(OptimizerKind=SimpleNamespace(sgd=0)))
    return client, c, model, optimizer, layout, gradient


@pytest.mark.parametrize("momentum", [0., .2])
def test_every_checkpoint_field_is_copied_and_inert_owners_do_not_shift_sgd_slots(momentum):
    client, c, model, optimizer, layout, gradient = actual_checkpoint(momentum)
    saved = record(client)
    assert saved["owner_names"] == ["alias", "fixed", "inert"]
    assert saved["offsets"] == [0, 2, 3] and set(saved["trainable"]) == {"alias", "weight"}
    assert set(saved["groups"][0]) == set(GROUP_FIELDS)
    decoded = layout.decode(gradient, saved)
    equivalent(decoded["gradients"], dict(weight=model.weight.grad, fixed=None), atol=0, rtol=0)
    equivalent(decoded["optimizer"], optimizer.state_dict(), atol=0, rtol=0)
    assert decoded["update_counts"] == dict(weight=2, fixed=0)
    equivalent(layout.parameters(saved), dict(model.named_parameters()), atol=0, rtol=0)
    for name in ("last_positions", "node_present", "region_times", "input_counts", "input_times"):
        equivalent(saved[name], getattr(c, name), atol=0, rtol=0)
    # Exact restart sees padding and bound state too, not just effective state.
    c.state[0].floating[0, -1] = -123.
    c.bound_initial[0].integer[0, -1] = 12
    c.slots.first.zero_()
    assert saved["raw"][0]["floating"][0, -1] == 33.
    assert saved["bound_initial"][0]["integer"][0, -1] == 0
    assert bool(saved["state"]["first"].count_nonzero()) == bool(momentum)


def test_slot_extent_and_undeclared_gradient_cannot_be_hidden():
    client, c, model, optimizer, layout, gradient = actual_checkpoint()
    saved = record(client)
    gradient.connected[-1] = True
    with pytest.raises(ValueError, match="unregistered resident parameter"):
        layout.decode(gradient, saved)
    gradient.connected[-1] = False
    saved["state"]["steps"][-1] = 1
    with pytest.raises(ValueError, match="unregistered block owner"):
        layout.decode(gradient, saved)
    c.slots.values = c.slots.values[:-1]
    with pytest.raises(ValueError, match="owner/slot extent"):
        record(client)
