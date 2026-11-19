"""Memory-only ownership, summed use derivatives and private continuation state."""
import copy
from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from benchmarks.continuous import nested_comparison
from tide import (ClockConfig, ModelConfig, TideModel, TopologyConfig, canonical_state,
                  canonical_state_dict, detach_state, load_checkpoint, place_model,
                  placement_adamw, save_checkpoint)
from tide.parameters import NodeLinear, NodeOwners, NodeVector


def config(**changes):
    base = ModelConfig(topology=TopologyConfig(kind="multiscale", n=4, hubs=2,
                                               profile="compact-5-dd4"),
                       d=64, memory="X", memory_sharing="region_shared",
                       attention_backend="math", clock=ClockConfig(kv_window=3))
    return replace(base, **changes)


def memory_parameter(name):
    return name.startswith("memories.") and ".ffn." not in name


@pytest.mark.parametrize("memory,partition", [("X", "group"), ("S", "group"),
                                               ("X_reverse", "group"), ("K", "mixed")])
@pytest.mark.parametrize("sharing", ["independent", "layer_shared"])
def test_memory_override_leaves_every_other_parameter_owner_unchanged(memory, partition, sharing):
    c = config(memory=memory, sharing=sharing)
    c = replace(c, topology=replace(c.topology, partition=partition))
    candidate = TideModel(c)
    original = TideModel(replace(c, memory_sharing=None))
    assert ModelConfig.from_dict(c.to_dict()) == c
    expected = {name: tuple(value.shape) for name, value in original.named_parameters()
                if not memory_parameter(name)}
    observed = {name: tuple(value.shape) for name, value in candidate.named_parameters()
                if not memory_parameter(name)}
    assert observed == expected
    previous = dict(original.named_modules())
    for name, module in candidate.named_modules():
        if isinstance(module, (NodeLinear, NodeVector)) and not memory_parameter(name+".weight"):
            assert module._host_groups == previous[name]._host_groups
            assert module.shared == previous[name].shared
    for kind, module in candidate.memories.items():
        nodes = [node for node, _ in sorted(candidate.memory_ids[kind].items(), key=lambda row: row[1])]
        for name, child in module.named_modules():
            if not isinstance(child, (NodeLinear, NodeVector)) or name.startswith("ffn."):
                continue
            assert child.shared
            for left, a in enumerate(nodes):
                for right, b in enumerate(nodes):
                    av, bv = candidate.graph.nodes[a], candidate.graph.nodes[b]
                    same = av.region == bv.region and (name == "kv" or av.role == bv.role)
                    assert (child._host_groups[left] == child._host_groups[right]) == same
            if isinstance(child, NodeLinear):
                rank = min(c.adapter_rank, child.in_features, child.out_features)
                assert child.left.shape == (len(nodes), child.out_features, rank)
                assert child.right.shape == (len(nodes), rank, child.in_features)
        if kind == "S":
            # The input includes the gate projection. All small SSD vectors and
            # convolution coefficients follow memory groups without adapters.
            for name in ("a_log", "dt_bias", "skip", "rho", "conv_kernel", "conv_bias"):
                child = getattr(module, name)
                assert child._host_groups == module.input._host_groups
                assert not hasattr(child, "left") and not hasattr(child, "right")


@pytest.mark.parametrize("value", ["independent", "layer_shared", "global", False])
def test_unknown_memory_override_is_not_silently_accepted(value):
    with pytest.raises(ValueError, match="memory_sharing"):
        config(memory_sharing=value)


@pytest.mark.parametrize("kind", ["K", "S"])
def test_memory_projection_values_and_vjp_sum_every_shared_and_repeated_owner_use(kind):
    torch.manual_seed(411)
    model = TideModel(config(memory=kind))
    module = model.memories[kind]
    for name, layer in module.named_modules():
        if not isinstance(layer, NodeLinear) or name.startswith("ffn."):
            continue
        # Two uses of node 1, another node sharing its base, and another region.
        other = next(i for i, group in enumerate(layer._host_groups) if group != layer._host_groups[0])
        owners = (1, 0, other, 1)
        ids = NodeOwners.from_static(owners, device="cpu")
        x = torch.randn(4, 2, layer.in_features, requires_grad=True)
        seed = torch.randn(4, 2, layer.out_features)
        actual = layer(x, ids)
        # Independent dense effective matrices are leaves. Their gradients are
        # later assembled by the written W_g + L_v R_v parameterization.
        dense = torch.stack([layer.weight[layer._host_groups[v]] + layer.left[v] @ layer.right[v]
                             for v in owners]).detach().requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        expected = torch.einsum("nti,noi->nto", reference_x, dense)
        reference_dx, per_use = torch.autograd.grad(expected, (reference_x, dense), seed)
        dx, dw, dl, dr = torch.autograd.grad(actual, (x, layer.weight, layer.left, layer.right), seed)
        expected_w, expected_l, expected_r = torch.zeros_like(dw), torch.zeros_like(dl), torch.zeros_like(dr)
        for v, contribution in zip(owners, per_use):
            expected_w[layer._host_groups[v]] += contribution
            expected_l[v] += contribution @ layer.right[v].detach().T
            expected_r[v] += layer.left[v].detach().T @ contribution
        for a, b in ((actual, expected), (dx, reference_dx), (dw, expected_w),
                     (dl, expected_l), (dr, expected_r)):
            torch.testing.assert_close(a, b, atol=6e-6, rtol=6e-6)
        assert dw[layer._host_groups[0]].abs().sum() > 0
        assert dl[1].abs().sum() > 0 and dr[1].abs().sum() > 0
    # Shared scalar/vector gradients also accumulate every owner use.
    if kind == "S":
        for name in ("a_log", "dt_bias", "skip", "rho", "conv_kernel", "conv_bias"):
            vector = getattr(module, name)
            owners = (1, 0, 1)
            seed = torch.randn(3, vector.value.shape[-1])
            actual = vector(NodeOwners.from_static(owners, device="cpu"))
            derivative, = torch.autograd.grad(actual, (vector.value,), seed)
            expected = torch.zeros_like(derivative)
            for v, contribution in zip(owners, seed):
                expected[vector._host_groups[v]] += contribution
            torch.testing.assert_close(derivative, expected, atol=0, rtol=0)


def state_row(value, row):
    if isinstance(value, torch.Tensor):
        return value[row:row+1]
    if is_dataclass(value):
        return type(value)(**{field.name: state_row(getattr(value, field.name), row) for field in fields(value)})
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(state_row(item, row) for item in value))
    raise TypeError(type(value))


@pytest.mark.parametrize("kind", ["K", "S"])
def test_shared_memory_never_shares_history_and_zero_writes_have_no_cross_node_derivative(kind):
    torch.manual_seed(417)
    module = TideModel(config(memory=kind)).memories[kind]
    module.short_decay = 0.03
    owners = NodeOwners.from_static((0, 1), device="cpu")
    assert (module.kv if kind == "K" else module.input)._host_groups[0] == (
        module.kv if kind == "K" else module.input)._host_groups[1]
    x = torch.randn(2, 1, 6, 64, requires_grad=True)
    mass = torch.tensor([[[1., .3, 0., 1., 0., .6]], [[.2, 0., 1., 0., .4, 0.]]], requires_grad=True)
    times = (2**53+13+torch.arange(6, dtype=torch.int64)).reshape(1, 1, -1).expand(2, 1, -1)
    description, _, combined = module.prepare(x, mass, times, owners)
    for row in (0, 1):
        separate_description, _, separate = module.prepare(x[row:row+1], mass[row:row+1], times[row:row+1],
                                                            NodeOwners.from_static((row,), device="cpu"))
        for a, b in zip(description, separate_description):
            torch.testing.assert_close(a[row:row+1], b, atol=1e-6, rtol=2e-5)
        assert nested_comparison(state_row(combined, row), separate, atol=1e-6, rtol=2e-5)["within_threshold"]
    # Perturb a different owner's entire history; the first owner is unchanged.
    altered = x.detach().clone()
    altered[1] *= 3
    _, _, changed = module.prepare(altered, mass.detach(), times, owners)
    assert nested_comparison(state_row(combined, 0), state_row(changed, 0), atol=0, rtol=0)["exact"]
    assert not nested_comparison(state_row(combined, 1), state_row(changed, 1), atol=0, rtol=0)["exact"]
    payload = (combined.features[0] * combined.weight[0, ..., None]).sum() if kind == "K" else combined.long[0].sum()+combined.matrix[0].sum()
    dx, dm = torch.autograd.grad(payload, (x, mass))
    assert torch.count_nonzero(dx[1]) == 0 and torch.count_nonzero(dm[1]) == 0
    assert torch.count_nonzero(dm[mass == 0]) == 0
    assert dx[0].abs().sum() > 0 and dm[0].abs().sum() > 0
    assert combined.times.dtype == torch.int64
    if kind == "S":
        assert combined.conv.times.dtype == torch.int64


def training_step(model, optimizer, data, state):
    optimizer.zero_grad(set_to_none=True)
    x = data.detach().clone().requires_grad_()
    result = model(x, state)
    value = result.readout()
    root = torch.cos(torch.arange(value.numel(), dtype=value.dtype)).reshape_as(value)
    (value*root).square().mean().backward()
    gradients = canonical_state_dict(model, gradients=True)
    optimizer.step()
    return result, detach_state(result.state), x.grad, gradients


def test_memory_only_sharing_bounded_fp64_placement_three_updates_checkpoint_and_actual_restore(tmp_path):
    torch.manual_seed(83)
    c = config(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4"),
               region_batch_size=2, checkpoint_regions=True, parameter_indexing="static_view")
    ordinary = TideModel(c).double()
    partitions = [(ordinary.graph.nodes[nodes[0]].position+ordinary.graph.nodes[nodes[0]].layer) % 2
                  for nodes in ordinary.graph.regions]
    placed = place_model(copy.deepcopy(ordinary), ["cpu:0", "cpu:1"], region_partitions=partitions)
    models = [ordinary, placed]
    optimizers = [torch.optim.AdamW(ordinary.parameters(), lr=1e-4), placement_adamw(placed, lr=1e-4)]
    states = [None, None]
    tolerance = dict(atol=2e-9, rtol=2e-8)
    with torch.inference_mode():
        placed(torch.randn(1, 1, 64, dtype=torch.float64))
    for step in range(3):
        data = torch.randn(1, 2, 64, dtype=torch.float64)
        results = []
        for index in range(2):
            result, states[index], dx, gradients = training_step(models[index], optimizers[index], data, states[index])
            results.append((result, dx, gradients))
        a, b = results
        torch.testing.assert_close(a[0].values, b[0].values, **tolerance)
        assert torch.equal(a[0].strengths > 0, b[0].strengths > 0)
        torch.testing.assert_close(a[1], b[1], **tolerance)
        torch.testing.assert_close(a[2], b[2], **tolerance)
        torch.testing.assert_close(canonical_state_dict(models[0]), canonical_state_dict(models[1]), **tolerance)
        assert nested_comparison(states[0], canonical_state(models[1], states[1]), **tolerance)["within_threshold"]
        if step == 1:
            path = tmp_path/"memory-only-region.pt"
            save_checkpoint(path, models[1], state=states[1], optimizer=optimizers[1])
            restored, state, slots = load_checkpoint(path, placement_devices=["cpu:0", "cpu:1"])
            resumed_optimizer = placement_adamw(restored, lr=1e-4)
            resumed_optimizer.load_state_dict(slots)
            assert restored.config.memory_sharing == "region_shared" and restored.config.sharing == "independent"
            assert nested_comparison(resumed_optimizer.state_dict(), optimizers[1].state_dict(), atol=0, rtol=0)["exact"]
            models[1], states[1], optimizers[1] = restored, state, resumed_optimizer
    assert all(float(slot["step"]) == 3 for slot in optimizers[1].state.values())


def test_memory_only_sharing_fp32_cpu_placement_value_and_vjp():
    torch.manual_seed(83)
    c = config(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4"),
               region_batch_size=2, checkpoint_regions=True, parameter_indexing="static_view")
    ordinary = TideModel(c)
    partitions = [(ordinary.graph.nodes[nodes[0]].position+ordinary.graph.nodes[nodes[0]].layer) % 2
                  for nodes in ordinary.graph.regions]
    placed = place_model(copy.deepcopy(ordinary), ["cpu:0", "cpu:1"], region_partitions=partitions)
    x = torch.randn(1, 2, 64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = ordinary(x), placed(y)
    root = torch.randn_like(a.values)
    (a.values*root).mean().backward()
    (b.values*root).mean().backward()
    tolerance = dict(atol=1e-6, rtol=2e-3)
    torch.testing.assert_close(a.values, b.values, **tolerance)
    assert torch.equal(a.strengths > 0, b.strengths > 0)
    torch.testing.assert_close(x.grad, y.grad, **tolerance)
    torch.testing.assert_close(canonical_state_dict(ordinary, gradients=True),
                               canonical_state_dict(placed, gradients=True), **tolerance)
    assert nested_comparison(a.state, canonical_state(placed, b.state), **tolerance)["within_threshold"]


@pytest.mark.parametrize("placed", [False, True])
def test_memory_only_sharing_fp32_checkpoint_restores_optimizer_and_actual_next_update(tmp_path, placed):
    torch.manual_seed(59)
    c = config(topology=TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4"),
               region_batch_size=2, checkpoint_regions=True, parameter_indexing="static_view")
    original = TideModel(c)
    if placed:
        partitions = [(original.graph.nodes[nodes[0]].position+original.graph.nodes[nodes[0]].layer) % 2
                      for nodes in original.graph.regions]
        original = place_model(original, ["cpu:0", "cpu:1"], region_partitions=partitions)
    optimizer = placement_adamw(original, lr=1e-4)
    state = None
    for _ in range(2):
        _, state, _, _ = training_step(original, optimizer, torch.randn(1, 2, 64), state)
    path = tmp_path/"memory-sharing-fp32.pt"
    save_checkpoint(path, original, state=state, optimizer=optimizer)
    restored, resumed_state, slots = load_checkpoint(path, placement_devices=["cpu:0", "cpu:1"] if placed else None)
    resumed_optimizer = placement_adamw(restored, lr=1e-4)
    resumed_optimizer.load_state_dict(slots)
    assert restored.config == c
    assert nested_comparison(restored.state_dict(), original.state_dict(), atol=0, rtol=0)["exact"]
    assert nested_comparison(resumed_state, state, atol=0, rtol=0)["exact"]
    assert nested_comparison(resumed_optimizer.state_dict(), optimizer.state_dict(), atol=0, rtol=0)["exact"]
    data = torch.randn(1, 2, 64)
    a = training_step(original, optimizer, data, state)
    b = training_step(restored, resumed_optimizer, data, resumed_state)
    tolerance = dict(atol=1e-6, rtol=2e-3)
    torch.testing.assert_close(a[0].values, b[0].values, **tolerance)
    assert torch.equal(a[0].strengths > 0, b[0].strengths > 0)
    torch.testing.assert_close(a[2], b[2], **tolerance)
    torch.testing.assert_close(a[3], b[3], **tolerance)
    assert nested_comparison(a[1], b[1], **tolerance)["within_threshold"]
    assert nested_comparison(restored.state_dict(), original.state_dict(), **tolerance)["within_threshold"]
    assert nested_comparison(resumed_optimizer.state_dict(), optimizer.state_dict(), **tolerance)["within_threshold"]
    assert all(float(slot["step"]) == 3 for slot in resumed_optimizer.state.values())
