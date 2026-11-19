"""Static-owner GEMM versus independent matrices and tensor-owner BMM."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tide import ModelConfig, TideModel, TopologyConfig
from tide.kv import KVMemory
from tide.parameters import NodeLinear, NodeOwners, NodeVector, owner_ids, select_owners
from tide.ssd_memory import SSDMemory


GROUPS = [3, 1, 3, 2, 3, 1]
CASES = [
    ([4, 0, 4, 2], (0, 2, 4), True),  # Nonzero group, duplicate private owners.
    ([4, 1, 2, 5], (0, 1, 2, 4, 5), False),
    ([4, 4, 2], (0, 1, 2, 4), False),  # Uniform subset, conservative mixed superset.
]


def literal(layer, x, ids):
    outputs = []
    for row, owner in zip(x, ids.tolist()):
        group = GROUPS[owner]
        weight = layer.weight[group]
        if layer.shared:
            weight = weight + layer.left[owner] @ layer.right[owner]
        value = row @ weight.T
        outputs.append(value if layer.bias is None else value + layer.bias[group])
    return torch.stack(outputs)


@pytest.mark.parametrize("ids,possible,uniform", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_base_values_all_gradients_and_mm_bmm_shapes(ids, possible, uniform, dtype, monkeypatch):
    torch.manual_seed(107)
    layer = NodeLinear(6, 5, 7, GROUPS, rank=3, bias=True).to(dtype)
    with torch.no_grad():
        layer.left.uniform_(-.3, .3)
        layer.right.uniform_(-.4, .4)
        layer.bias.normal_()
    owners = torch.tensor(ids)
    annotated = NodeOwners(owners, possible)
    x = torch.randn(len(ids), 2, 3, 5, dtype=dtype, requires_grad=True)
    calls, mm, bmm = [], torch.mm, torch.bmm

    def observed_mm(a, b):
        calls.append(("mm", tuple(a.shape), tuple(b.shape)))
        return mm(a, b)

    def observed_bmm(a, b):
        calls.append(("bmm", tuple(a.shape), tuple(b.shape)))
        return bmm(a, b)

    monkeypatch.setattr(torch, "mm", observed_mm)
    monkeypatch.setattr(torch, "bmm", observed_bmm)
    actual = layer(x, annotated)
    expected_calls = ([('mm', (len(ids)*6, 5), (5, 7))] if uniform else
                      [('bmm', (len(ids), 6, 5), (len(ids), 5, 7))])
    expected_calls += [('bmm', (len(ids), 6, 5), (len(ids), 5, 3)),
                       ('bmm', (len(ids), 6, 3), (len(ids), 3, 7))]
    assert calls == expected_calls
    calls.clear()
    raw, expected = layer(x, owners), literal(layer, x, owners)
    assert [kind for kind, _, _ in calls] == ["bmm"]*3
    tolerance = dict(atol=2e-6, rtol=2e-6) if dtype == torch.float32 else dict(atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(actual, raw, **tolerance)
    torch.testing.assert_close(actual, expected, **tolerance)
    root = torch.cos(torch.arange(actual.numel(), dtype=dtype)).reshape_as(actual)
    parameters = [x, *layer.parameters()]
    gradients = [torch.autograd.grad(value, parameters, root, retain_graph=True)
                 for value in (actual, raw, expected)]
    tolerance = dict(atol=6e-6, rtol=6e-6) if dtype == torch.float32 else dict(atol=3e-12, rtol=3e-12)
    for actual_grad, raw_grad, expected_grad in zip(*gradients):
        torch.testing.assert_close(actual_grad, raw_grad, **tolerance)
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


@pytest.mark.parametrize("ids,possible,uniform", CASES)
def test_vector_uniform_broadcast_and_duplicate_owner_gradients(ids, possible, uniform):
    torch.manual_seed(154)
    vector = NodeVector(6, 9, GROUPS).double()
    with torch.no_grad():
        vector.value.normal_()
    owners = torch.tensor(ids)
    actual = vector(NodeOwners(owners, possible))
    raw = vector(owners)
    expected = torch.stack([vector.value[GROUPS[i]] for i in ids])
    assert (actual.stride(0) == 0) == uniform
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    root = torch.randn_like(actual)
    gradients = [torch.autograd.grad(value, vector.value, root, retain_graph=True)[0]
                 for value in (actual, raw, expected)]
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, gradients[0], atol=1e-12, rtol=1e-12)


def test_selection_preserves_superset_without_host_tensor_reads(monkeypatch):
    layer = NodeLinear(6, 5, 7, GROUPS, rank=3).double()
    vector = NodeVector(6, 9, GROUPS).double()
    owners = NodeOwners(torch.tensor([4, 0, 4, 2]), (0, 2, 4))
    selection = torch.tensor([3, 0, 0])
    x = torch.randn(3, 2, 5, dtype=torch.float64)

    def forbidden(*args, **kwargs):
        raise AssertionError("uniform-path selection must not read device tensor values")

    with monkeypatch.context() as patch:
        for method in ("item", "tolist", "cpu", "unique", "__int__", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        selected = select_owners(owners, selection)
        actual, vectors = layer(x, selected), vector(selected)
    assert selected.possible_ids is owners.possible_ids
    torch.testing.assert_close(owner_ids(selected), torch.tensor([2, 4, 4]))
    torch.testing.assert_close(actual, layer(x, owner_ids(selected)), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(vectors, vector(owner_ids(selected)), atol=0, rtol=0)


def test_each_module_uses_its_own_groups_and_reload_invalidates_cached_uniformity():
    first = NodeLinear(6, 5, 7, GROUPS, rank=3).double()
    second = NodeLinear(6, 5, 7, [3, 1, 1, 2, 3, 1], rank=3).double()
    owners = NodeOwners(torch.tensor([4, 0, 2]), (0, 2, 4))
    x = torch.randn(3, 2, 5, dtype=torch.float64)
    assert first._uniform_group(owners) == 3
    assert second._uniform_group(owners) is None
    checkpoint = {name: value.clone() for name, value in second.state_dict().items()}
    first.load_state_dict(checkpoint)
    assert first._uniform_group(owners) is None
    torch.testing.assert_close(first(x, owners), first(x, owners.ids), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["K", "S"])
def test_sparse_memory_full_keeps_static_owner_superset(kind):
    torch.manual_seed(93)
    constructor = KVMemory if kind == "K" else SSDMemory
    memory = constructor(3, 64, [2, 1, 2], 4, 4, chunk=4).double()
    owners = NodeOwners(torch.tensor([2, 0]), (0, 2))
    u = torch.randn(2, 1, 3, 64, dtype=torch.float64, requires_grad=True)
    mass = torch.ones(2, 1, 3, dtype=u.dtype)
    times = (1+4*torch.arange(3))[None, None].expand(2, 1, 3)
    control = torch.tensor([[[0., 0., 0.]], [[1., 0., 1.]]], dtype=u.dtype)
    observed = []
    handle = memory.output.register_forward_pre_hook(lambda _, args: observed.append(args[1]))
    desc, context, _ = memory.prepare(u, mass, times, owners)
    output = memory.finish(context, control)
    handle.remove()
    assert len(observed) == 1 and observed[0].possible_ids is owners.possible_ids
    torch.testing.assert_close(observed[0].ids, torch.tensor([0]))
    raw_desc, raw_context, _ = memory.prepare(u, mass, times, owners.ids)
    raw = memory.finish(raw_context, control)
    torch.testing.assert_close(output, raw, atol=3e-11, rtol=3e-11)
    roots = [u, *memory.parameters()]
    actual_loss = output.square().sum() + sum(value.square().sum() for value in desc)
    raw_loss = raw.square().sum() + sum(value.square().sum() for value in raw_desc)
    actual_grad = torch.autograd.grad(actual_loss, roots, retain_graph=True)
    for actual, expected in zip(actual_grad, torch.autograd.grad(raw_loss, roots)):
        torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-10)


@pytest.mark.parametrize("timed", [False, True])
def test_continuation_time_rejects_invalid_position_and_int64_overflow_before_work(timed):
    topology = (TopologyConfig(kind="multiscale", n=1, profile="compact-5-dd4", timing="timed")
                if timed else TopologyConfig(n=1, layers=1))
    model = TideModel(ModelConfig(topology=topology, d=64))
    x = torch.randn(1, 1, 64)
    for position in (-1, True, 1.0, torch.tensor(1), 2**63):
        with pytest.raises(OverflowError):
            model(x, SimpleNamespace(position=position))
    with torch.no_grad():
        state = model(x).state
        largest = ((torch.iinfo(torch.int64).max//model.graph.period)-1 if timed else
                   (torch.iinfo(torch.int64).max-len(model.layer_nodes))//model.graph.period)
        result = model(x, replace(state, position=largest))
        assert result.times[-1] <= torch.iinfo(torch.int64).max
        with pytest.raises(OverflowError):
            model(x, replace(state, position=largest+1))
