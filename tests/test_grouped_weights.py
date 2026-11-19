"""Direct weight-bank values, shared adjoints, materialization and real resume."""
import copy

import pytest
import torch

from benchmarks.parameter_gathers import ParameterGatherAudit
from tide import parameters
from tide.parameters import FeedForward, NodeLinear, NodeOwners


GROUPS = (3, 1, 3, 0, 3, 2)


def literal(layer, x, ids):
    rows = []
    for row, owner in zip(x, ids.tolist()):
        group = layer._host_groups[owner]
        matrix = layer.weight[group]
        if layer.shared:
            matrix = matrix + layer.left[owner] @ layer.right[owner]
        value = row @ matrix.T
        rows.append(value if layer.bias is None else value + layer.bias[group])
    return torch.stack(rows)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("groups,ids,possible", [
    (None, [4, 1, 4, 0], (0, 1, 3, 4)),
    (GROUPS, [4, 1, 4, 0], (0, 1, 3, 4)),
    (GROUPS, [4, 0, 4, 2], (0, 2, 4)),
])
def test_direct_bank_independent_values_and_shared_private_gradients(dtype, groups, ids, possible):
    torch.manual_seed(581)
    layer = NodeLinear(6, 5, 7, groups, rank=3, bias=True, weight_access="grouped").to(dtype)
    with torch.no_grad():
        layer.bias.normal_()
        if layer.shared:
            layer.left.uniform_(-.3, .3)
            layer.right.uniform_(-.4, .4)
    ids = torch.tensor(ids, dtype=torch.int64)
    x = torch.randn(len(ids), 3, 2, 5, dtype=dtype).transpose(1, 2).requires_grad_()
    actual = layer(x, NodeOwners(ids, possible))
    expected = literal(layer, x, ids)
    value_tol = 2e-6 if dtype == torch.float32 else 2e-12
    grad_tol = 6e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, atol=value_tol, rtol=value_tol)
    seed = torch.cos(torch.arange(actual.numel(), dtype=dtype)).reshape_as(actual)
    roots = (x, *layer.parameters())
    actual_grad = torch.autograd.grad(actual, roots, seed, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, roots, seed)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, atol=grad_tol, rtol=grad_tol)
    used_groups = {layer._host_groups[owner] for owner in ids.tolist()}
    for group in set(range(layer.weight.shape[0])) - used_groups:
        assert torch.count_nonzero(actual_grad[1][group]) == 0


@pytest.mark.parametrize("groups", [None, GROUPS])
def test_ffn_reuses_mapping_and_eliminates_parameter_gather(groups, monkeypatch):
    torch.manual_seed(166)
    layer = FeedForward(6, 16, groups, rank=3, hidden_features=32, weight_access="grouped")
    owners = NodeOwners(torch.tensor([4, 1, 4, 0]), (0, 1, 3, 4))
    x = torch.randn(4, 2, 3, 16, requires_grad=True)
    real_plan, plans, arguments = parameters.make_grouped_plan, [], []
    def counted_plan(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        plans.append(plan)
        return plan
    monkeypatch.setattr(parameters, "make_grouped_plan", counted_plan)
    handles = [part.register_forward_pre_hook(lambda _, args: arguments.append(args[1]))
               for part in (layer.gate, layer.up, layer.down)]
    with ParameterGatherAudit(layer) as audit:
        output = layer(x, owners)
        output.square().mean().backward()
    for handle in handles:
        handle.remove()
    assert len(plans) == (1 if groups is None else 2)
    assert len(arguments) == 3 and arguments[0] is arguments[1] is arguments[2]
    assert audit.report()["output_bytes"] == 0
    assert x.grad is not None and all(p.grad is not None for p in layer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_plan_reads_no_dynamic_owner_values_on_host(monkeypatch):
    layer = NodeLinear(6, 5, 7, GROUPS, rank=3, weight_access="grouped")
    owners = NodeOwners(torch.tensor([4, 1, 4, 0]), (0, 1, 3, 4))
    x = torch.randn(4, 2, 5)
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not inspect dynamic owner tensor values")
    with monkeypatch.context() as patch:
        for name in ("item", "tolist", "cpu", "__int__", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        plan = layer._grouped_owners(x, owners)
    assert plan.base_plan.rows_per_owner == 2
    assert plan.private_plan.rows_per_owner == 2


@pytest.mark.parametrize("groups", [None, GROUPS])
def test_empty_projection_has_connected_zero_gradients(groups):
    layer = NodeLinear(6, 5, 7, groups, rank=3, bias=True, weight_access="grouped")
    x = torch.empty(0, 2, 5, requires_grad=True)
    result = layer(x, NodeOwners(torch.empty(0, dtype=torch.int64), ()))
    assert result.shape == (0, 2, 7)
    result.sum().backward()
    assert x.grad.shape == x.shape
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in layer.parameters())


def test_grouped_policy_rejects_autocast_and_preserves_parameter_identity():
    grouped = FeedForward(3, 16, hidden_features=32, weight_access="grouped")
    gather = FeedForward(3, 16, hidden_features=32)
    assert list(grouped.state_dict()) == list(gather.state_dict())
    grouped.load_state_dict(gather.state_dict())
    with pytest.raises(ValueError, match="autocast"):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            grouped(torch.randn(3, 2, 16))
    with pytest.raises(ValueError, match="weight_access"):
        FeedForward(3, 16, hidden_features=32, weight_access="hidden_gather")


def test_changed_ownership_does_not_reuse_an_old_plan():
    layer = NodeLinear(6, 5, 7, GROUPS, rank=3, weight_access="grouped")
    changed = NodeLinear(6, 5, 7, (2, 0, 2, 1, 0, 3), rank=3, weight_access="grouped")
    owners = NodeOwners(torch.tensor([4, 1, 4, 0]), (0, 1, 3, 4))
    x = torch.randn(4, 2, 5)
    plan = layer._grouped_owners(x, owners)
    layer.load_state_dict(changed.state_dict())
    assert layer._grouped_owners(x, plan) is not plan
    torch.testing.assert_close(layer(x, plan), literal(layer, x, owners.ids), atol=2e-6, rtol=2e-6)


def test_same_group_map_different_adapter_ownership_rebuilds_plan():
    independent = NodeLinear(4, 5, 7, weight_access="grouped")
    shared = NodeLinear(4, 5, 7, list(range(4)), rank=3, weight_access="grouped")
    owners = NodeOwners(torch.tensor([3, 0, 3]), (0, 1, 2, 3))
    x = torch.randn(3, 2, 5)
    plan = independent._grouped_owners(x, owners)
    assert plan.private_plan is None
    replacement = shared._grouped_owners(x, plan)
    assert replacement is not plan and replacement.private_plan is not None
    torch.testing.assert_close(shared(x, plan), literal(shared, x, owners.ids), atol=2e-6, rtol=2e-6)
    assert independent._grouped_owners(x, replacement).private_plan is None


def test_grouped_adamw_checkpoint_really_restores_next_update(tmp_path):
    torch.manual_seed(221)
    model = FeedForward(6, 16, GROUPS, rank=3, hidden_features=32, weight_access="grouped")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    owners = NodeOwners(torch.tensor([4, 1, 4, 0]), (0, 1, 3, 4))
    inputs = [torch.randn(4, 2, 3, 16) for _ in range(4)]
    def step(module, opt, x):
        opt.zero_grad(set_to_none=True)
        output = module(x, owners)
        loss = (output.sin() * torch.cos(x)).mean()
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in module.named_parameters()}
        opt.step()
        return output.detach(), loss.detach(), gradients
    for x in inputs[:3]:
        step(model, optimizer, x)
    path = tmp_path / "grouped.pt"
    torch.save({"parameters": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    reference = step(model, optimizer, inputs[3])
    fresh = FeedForward(6, 16, GROUPS, rank=3, hidden_features=32, weight_access="grouped")
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3, foreach=False)
    checkpoint = torch.load(path, weights_only=True)
    fresh.load_state_dict(checkpoint["parameters"])
    fresh_opt.load_state_dict(checkpoint["optimizer"])
    restored = step(fresh, fresh_opt, inputs[3])
    for actual, expected in zip(restored[:2], reference[:2]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in restored[2]:
        torch.testing.assert_close(restored[2][name], reference[2][name], atol=0, rtol=0)
    for a, b in zip(fresh.parameters(), model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(fresh_opt.state.values(), optimizer.state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_grouped_and_gather_ffn_input_and_parameter_vjp():
    torch.manual_seed(904)
    grouped = FeedForward(6, 16, GROUPS, rank=3, hidden_features=32, weight_access="grouped")
    gather = FeedForward(6, 16, GROUPS, rank=3, hidden_features=32)
    gather.load_state_dict(copy.deepcopy(grouped.state_dict()))
    ids = torch.tensor([4, 1, 4, 0])
    x = torch.randn(4, 2, 3, 16, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    actual, expected = grouped(x, ids), gather(x2, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    root = torch.sin(torch.arange(actual.numel())).reshape_as(actual)
    left = torch.autograd.grad(actual, (x, *grouped.parameters()), root)
    right = torch.autograd.grad(expected, (x2, *gather.parameters()), root)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=6e-6, rtol=6e-6)
