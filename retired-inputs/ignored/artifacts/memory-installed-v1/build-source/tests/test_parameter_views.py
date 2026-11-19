"""Trusted exact ownership versus gather, dynamic selection and actual updates."""
import copy
from dataclasses import replace

import pytest
import torch

from benchmarks.continuous import nested_comparison
from benchmarks.parameter_gathers import ParameterGatherAudit
from tide import (ModelConfig, TideModel, TopologyConfig, detach_state, load_checkpoint,
                  place_model, placement_adamw, save_checkpoint)
from tide.parameters import NodeLinear, NodeOwners, NodeVector, fused_node_linears, select_owners


def literal(layer, x, ids):
    rows = []
    for row, owner in zip(x, ids):
        group = layer._host_groups[owner]
        base = row @ layer.weight[group].T
        if layer.shared:
            base = base + (row @ layer.right[owner].T) @ layer.left[owner].T
        rows.append(base if layer.bias is None else base+layer.bias[group])
    return torch.stack(rows)


@pytest.mark.parametrize("groups,ids", [
    (None, (1, 2, 3)), ((0, 1, 2, 3, 4, 5), (1, 2, 3)),
    ((0, 0, 0, 0, 0, 0), (1, 2, 3)), ((0, 0, 1, 1, 2, 2), (1, 2, 3)),
    ((0, 1, 2, 3, 4, 5), (4, 2, 3)), ((0, 1, 2, 3, 4, 5), (2, 2, 3)),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_row_views_match_independent_formula_and_all_derivatives(groups, ids, dtype):
    torch.manual_seed(147)
    layers = torch.nn.ModuleList(NodeLinear(6, 5, 7, groups, rank=3, bias=True).to(dtype) for _ in range(2))
    owners = NodeOwners.from_static(ids, device="cpu", possible_ids=tuple(range(6)))
    dynamic = NodeOwners(owners.ids, owners.possible_ids)
    x = torch.randn(len(ids), 2, 3, 5, dtype=dtype, requires_grad=True)
    expected = torch.cat([literal(layer, x, ids) for layer in layers], -1)
    actual = torch.cat([layer(x, owners) for layer in layers], -1)
    gathered = torch.cat([layer(x, dynamic) for layer in layers], -1)
    fused = fused_node_linears(layers, x, owners)
    root = torch.randn_like(expected)
    values = (x, *layers.parameters())
    expected_grads = torch.autograd.grad(expected, values, root, retain_graph=True)
    tol = dict(atol=6e-6, rtol=6e-6) if dtype == torch.float32 else dict(atol=3e-12, rtol=3e-12)
    for result in (actual, gathered, fused):
        torch.testing.assert_close(result, expected, **tol)
        for a, b in zip(torch.autograd.grad(result, values, root, retain_graph=True), expected_grads):
            torch.testing.assert_close(a, b, **tol)
    vector = NodeVector(6, 9, groups).to(dtype)
    with torch.no_grad():
        vector.value.normal_()
    reference = torch.stack([vector.value[vector._host_groups[i]] for i in ids])
    observed = vector(owners)
    torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    seed = torch.randn_like(reference)
    torch.testing.assert_close(torch.autograd.grad(observed, vector.value, seed)[0],
                               torch.autograd.grad(reference, vector.value, seed)[0], **tol)


def test_parameter_materializations_only_disappear_for_exact_contiguous_rows():
    layer = NodeLinear(6, 5, 7, list(range(6)), rank=3, bias=True)
    x = torch.randn(3, 1, 1, 5)
    def audit(owners):
        with torch.no_grad(), ParameterGatherAudit(layer) as measured:
            layer(x, owners)
        return measured.report()
    owners = NodeOwners.from_static((1, 2, 3), device="cpu", possible_ids=tuple(range(6)))
    gathered = audit(NodeOwners(owners.ids, owners.possible_ids))
    viewed = audit(owners)
    # Three rows of weight/bias and of both private adapters, all FP32.
    assert gathered["output_bytes"] == 3*(7*5+7+7*3+3*5)*4 == 936
    assert viewed["output_bytes"] == 0
    # Equal row count and superset never imply exact order.
    permutation = NodeOwners(torch.tensor([3, 2, 1]), (1, 2, 3))
    assert audit(permutation)["output_bytes"] == gathered["output_bytes"]
    repeated = NodeOwners.from_static((2, 2, 3), device="cpu", possible_ids=tuple(range(6)))
    assert audit(repeated)["output_bytes"] == gathered["output_bytes"]
    selected = select_owners(owners, torch.tensor([2, 0, 0]))
    assert selected.exact_ids is None and selected.possible_ids is owners.possible_ids
    assert audit(selected)["output_bytes"] == gathered["output_bytes"]


def test_view_selection_never_reads_device_owner_ids_and_reload_invalidates_cache(monkeypatch):
    layer = NodeLinear(6, 5, 7, list(range(6)), rank=3, bias=True).double()
    owners = NodeOwners.from_static((1, 2, 3), device="cpu", possible_ids=tuple(range(6)))
    x = torch.randn(3, 2, 5, dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError("parameter selection must not inspect device owner values")
    with monkeypatch.context() as patch:
        for method in ("item", "tolist", "cpu", "unique", "__int__", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        observed = layer(x, owners)
    torch.testing.assert_close(observed, literal(layer, x, (1, 2, 3)), atol=3e-12, rtol=3e-12)
    assert isinstance(layer._view_indices[(owners.exact_ids, True)], slice)
    replacement = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    replacement["groups"] = torch.tensor([0, 4, 2, 3, 1, 5])
    layer.load_state_dict(replacement)
    assert not layer._view_indices
    torch.testing.assert_close(layer(x, owners), literal(layer, x, (1, 2, 3)), atol=3e-12, rtol=3e-12)
    assert layer._view_indices[(owners.exact_ids, True)] is None


@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("kind", ["K", "S"])
def test_full_dynamic_selection_keeps_gathers_after_static_prepare_and_placement(placed, kind):
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, profile="compact-9-two-inputs"),
                         d=64, memory=kind, parameter_indexing="static_view", region_batch_size=2)
    model = TideModel(config)
    if placed:
        partitions = [(model.graph.nodes[nodes[0]].position+model.graph.nodes[nodes[0]].layer) % 2 for nodes in model.graph.regions]
        model = place_model(model, ["cpu:0", "cpu:1"], region_partitions=partitions)
    prepared, finished = [], []
    prepare_name = "kv" if kind == "K" else "input"
    handles = [getattr(model.memories[kind], prepare_name).register_forward_pre_hook(lambda _, args: prepared.append(args[1])),
               model.memories[kind].output.register_forward_pre_hook(lambda _, args: finished.append(args[1]))]
    try:
        with torch.no_grad(), ParameterGatherAudit(model) as audit:
            model(torch.randn(1, 1, 64))
    finally:
        for handle in handles:
            handle.remove()
    assert prepared and finished
    assert all(owner.exact_ids is not None for owner in prepared)
    assert all(owner.exact_ids is None for owner in finished)
    rows = audit.report()["parameters"]
    assert not any(name.startswith(f"memories.{kind}.{prepare_name}.") for name in rows)
    assert any(name.startswith(f"memories.{kind}.ffn.") for name in rows)


@pytest.mark.parametrize("kind", ["fc", "timed", "placed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_views_three_actual_updates_continuation_and_checkpoint(tmp_path, kind, dtype):
    torch.manual_seed(82)
    topology = (TopologyConfig(n=2, layers=2) if kind == "fc" else
                TopologyConfig(kind="multiscale", n=2, hubs=1, profile="compact-5-dd4", timing="timed"))
    config = ModelConfig(topology=topology, d=64, memory="K" if kind == "fc" else "X",
                         sharing="independent" if kind == "fc" else "region_shared",
                         aggregation="incoming_light_64", region_batch_size=2, checkpoint_reception=True,
                         reception_qkv_backend="fused", reception_mask_padding=True)
    original = TideModel(config).to(dtype=dtype)
    candidate = copy.deepcopy(original)
    candidate.config = replace(config, parameter_indexing="static_view")
    if kind == "placed":
        partitions = [(candidate.graph.nodes[nodes[0]].position+candidate.graph.nodes[nodes[0]].layer) % 2
                      for nodes in candidate.graph.regions]
        original = place_model(original, ["cpu:0", "cpu:1"], region_partitions=partitions, region_concurrency="per_device")
        candidate = place_model(candidate, ["cpu:0", "cpu:1"], region_partitions=partitions, region_concurrency="per_device")
    models = [original, candidate]
    make_optimizer = placement_adamw if kind == "placed" else torch.optim.AdamW
    optimizers = [make_optimizer(model, lr=1e-4) if kind == "placed" else make_optimizer(model.parameters(), lr=1e-4)
                  for model in models]
    # Cached exact metadata must survive inference-mode initialization before
    # checkpointed, grad-enabled training. Private state below starts at zero.
    with torch.inference_mode():
        candidate(torch.randn(1, 1, 64, dtype=dtype))
    states = [None, None]
    for step in range(3):
        data = torch.randn(1, 2, 64, dtype=dtype)
        results, gradients, supports = [], [], []
        for index, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            x = data.clone().requires_grad_()
            result = model(x, states[index])
            value, following = result.readout(), result.state
            support = [result.strengths.detach() > 0]
            if not model.settle:
                tail = model.flush(following)
                value, following = torch.cat((value, tail.readout()), 1), tail.state
                support.append(tail.strengths.detach() > 0)
            loss = (value*torch.cos(torch.arange(value.numel(), dtype=dtype)).reshape_as(value)).square().mean()
            loss.backward()
            gradients.append([x.grad, *(None if p.grad is None else p.grad.clone() for p in model.parameters())])
            results.append(value.detach())
            supports.append(support)
            optimizer.step()
            states[index] = detach_state(following)
        ordinary = dict(atol=1e-6, rtol=2e-3) if dtype == torch.float32 else dict(atol=1e-10, rtol=1e-8)
        derivative = dict(atol=1e-6, rtol=2e-3) if dtype == torch.float32 else dict(atol=2e-9, rtol=2e-8)
        torch.testing.assert_close(results[1], results[0], **ordinary)
        assert all(torch.equal(a, b) for a, b in zip(supports[1], supports[0]))
        for a, b in zip(gradients[1], gradients[0]):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, **derivative)
        for actual, expected in zip(candidate.parameters(), original.parameters()):
            torch.testing.assert_close(actual, expected, **ordinary)
        compared_state = nested_comparison(states[1], states[0], **derivative)
        assert compared_state["within_threshold"] and not compared_state["support_mismatches"]
        assert nested_comparison(optimizers[1].state_dict(), optimizers[0].state_dict(), **derivative)["within_threshold"]
        if step == 1:
            path = tmp_path/f"{kind}.pt"
            save_checkpoint(path, candidate, optimizer=optimizers[1], state=states[1])
            candidate, restored_state, slots = load_checkpoint(path, device="cpu",
                placement_devices=["cpu:0", "cpu:1"] if kind == "placed" else None)
            resumed = (make_optimizer(candidate, lr=1e-4) if kind == "placed" else
                       make_optimizer(candidate.parameters(), lr=1e-4))
            resumed.load_state_dict(slots)
            models[1], optimizers[1], states[1] = candidate, resumed, restored_state
            assert candidate.config.parameter_indexing == "static_view"
    assert all(float(slot["step"]) == 3 for slot in optimizers[1].state.values())
