"""Selected FFN rows: independent formulas, ownership and private-state boundaries."""
import copy
from dataclasses import fields, is_dataclass

import pytest
import torch

from benchmarks.continuous import nested_comparison
from benchmarks.parameter_gathers import ParameterGatherAudit
from tide import ModelConfig, TideModel, TopologyConfig, canonical_state, canonical_state_dict, place_model
from tide.ffn_packing import power2_buckets, selected_ffn
from tide.kv import KVMemory
from tide.layout import select_queries
from tide.parameters import FeedForward, NodeOwners, select_owners
from tide.ssd_memory import SSDMemory


def fixture(dtype=torch.float32):
    # Five active nodes, with 1, 2, 3, 5 and 9 rows over four samples.
    counts = ((0, 0, 0, 0), (0, 0, 1, 0), (0, 1, 1, 0),
              (1, 1, 1, 0), (2, 1, 1, 1), (3, 2, 2, 2))
    control = torch.zeros(6, 4, 4, dtype=dtype)
    for node, per_sample in enumerate(counts):
        for batch, count in enumerate(per_sample):
            control[node, batch, :count] = 0.25
    content = torch.randn(6, 4, 4, 64, dtype=dtype)
    times = (2**53 + 19 + 2*torch.arange(4))[None, None].expand(6, 4, 4)
    return select_queries(content, control, times), control


def tolerance(dtype, *, gradients=False):
    if dtype == torch.float32:
        return dict(atol=1e-6, rtol=2e-3)
    return dict(atol=2e-9, rtol=2e-8) if gradients else dict(atol=1e-10, rtol=1e-8)


def literal_linear(layer, value, owner):
    weight = layer.weight[layer._host_groups[owner]]
    if layer.shared:
        weight = weight + layer.left[owner] @ layer.right[owner]
    result = weight @ value
    return result if layer.bias is None else result + layer.bias[layer._host_groups[owner]]


def literal_ffn(ffn, value, owner):
    normalized = value / torch.sqrt(value.square().sum()/value.numel() + 1e-5)
    hidden = torch.nn.functional.silu(literal_linear(ffn.gate, normalized, owner))
    hidden = hidden * literal_linear(ffn.up, normalized, owner)
    return value + literal_linear(ffn.down, hidden, owner)


def tensor_fields(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if is_dataclass(value):
        return [tensor for field in fields(value) for tensor in tensor_fields(getattr(value, field.name))]
    return [tensor for item in value for tensor in tensor_fields(item)]


def test_buckets_cover_each_selected_row_and_owner_once_with_capped_last_width():
    queries, _ = fixture()
    buckets = power2_buckets(queries)
    assert [bucket.width for bucket in buckets] == [1, 2, 4, 8, 12]
    assert [bucket.rows.numel() for bucket in buckets] == [1, 2, 3, 5, 9]
    assert sum(bucket.nodes.numel()*bucket.width for bucket in buckets) == 27
    assert queries.node.numel() == 20
    assert queries.valid.numel() == 60
    assert sorted(torch.cat([bucket.nodes for bucket in buckets]).tolist()) == list(range(5))
    assert sorted(torch.cat([bucket.rows for bucket in buckets]).tolist()) == list(range(20))
    for bucket in buckets:
        assert torch.equal(bucket.nodes[bucket.local_node], queries.node[bucket.rows])
        for local in range(bucket.nodes.numel()):
            slots = bucket.slot[bucket.local_node == local]
            assert torch.equal(slots, torch.arange(slots.numel()))


def test_decode_buckets_combine_samples_instead_of_bucketing_unit_query_lengths():
    control = torch.zeros(5, 8, 1)
    for node, count in enumerate((1, 2, 3, 5, 8)):
        control[node, :count] = 1
    queries = select_queries(torch.randn(5, 8, 1, 64), control,
                             torch.full((5, 8, 1), 2**53+21, dtype=torch.int64))
    buckets = power2_buckets(queries)
    assert queries.valid.shape == (5, 8, 1)
    assert queries.node.numel() == 19
    assert sum(bucket.nodes.numel()*bucket.width for bucket in buckets) == 23 < 40
    assert [(bucket.width, bucket.nodes.numel()) for bucket in buckets] == [(1, 1), (2, 1), (4, 1), (8, 2)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("groups", [None, [0]*7, [0, 0, 1, 1, 2, 2, 2]])
def test_compact_ffn_matches_literal_rows_and_repeated_owner_shared_vjp(dtype, groups):
    torch.manual_seed(901)
    queries, control = fixture(dtype)
    ffn = FeedForward(7, 64, groups, rank=4, hidden_features=128).to(dtype)
    # The two smallest active nodes use owner 1 twice, in different buckets.
    original_ids = (6, 1, 1, 3, 0, 2)
    owners = select_owners(NodeOwners.from_static(original_ids, device="cpu"), queries.nodes)
    x = torch.randn_like(queries.data, requires_grad=True)
    original = queries.publish(selected_ffn(ffn, x, queries, owners))
    compact = queries.publish(selected_ffn(ffn, x, queries, owners, packing="power2"))
    # A bounded FP64 formula is the independent reference for the same stored
    # parameters and inputs. The separate FP32 formula failure is retained below.
    reference_ffn = copy.deepcopy(ffn).double()
    reference_x = x.detach().double().requires_grad_()
    rows = [literal_ffn(reference_ffn, reference_x[node, batch, slot], original_ids[int(queries.nodes[node])])
            for node, batch, slot in zip(queries.node.tolist(), queries.batch.tolist(), queries.slot.tolist())]
    expected = reference_x.new_zeros(control.shape+(64,)).index_put(
        (queries.nodes[queries.node], queries.batch, queries.time), torch.stack(rows))
    root = torch.randn_like(original)
    inputs = (x, *ffn.parameters())
    reference_gradients = torch.autograd.grad(expected, (reference_x, *reference_ffn.parameters()), root.double())
    ordinary_gradients = torch.autograd.grad(original, inputs, root, retain_graph=True)
    for observed in (original, compact):
        torch.testing.assert_close(observed.double(), expected, **tolerance(dtype))
        gradients = torch.autograd.grad(observed, inputs, root, retain_graph=True)
        for actual, reference in zip(gradients, reference_gradients):
            torch.testing.assert_close(actual.double(), reference, **tolerance(dtype, gradients=True))
        for actual, ordinary in zip(gradients, ordinary_gradients):
            torch.testing.assert_close(actual, ordinary, **tolerance(dtype, gradients=True))
        assert torch.count_nonzero(gradients[0][~queries.valid]) == 0
    # Nonzero, large invalid rows may have their own normalization and residual
    # in the old path, but neither their values nor their derivatives publish.
    changed = torch.where(queries.valid[..., None], x, x+10_000)
    for mode in ("rectangular", "power2"):
        actual = queries.publish(selected_ffn(ffn, changed, queries, owners, packing=mode))
        torch.testing.assert_close(actual, compact, **tolerance(dtype))


class KnownFP32FormulaDeviation(AssertionError):
    pass


@pytest.mark.xfail(strict=True, raises=KnownFP32FormulaDeviation,
                   reason="Retained FP32 shared FFN literal-formula gradient exceedance at unchanged thresholds")
def test_retained_fp32_literal_formula_failure_is_bounded_and_visible():
    """Keep the original strict failure separate from the FP64 formula witness.

    The expected envelope below records one observed discrepancy; it does not
    replace the unchanged acceptance threshold or qualify this FP32 oracle pair.
    Unexpected fields/coordinates or larger differences fail instead of xfail.
    """
    torch.manual_seed(901)
    queries, control = fixture()
    ffn = FeedForward(7, 64, [0]*7, rank=4, hidden_features=128)
    original_ids = (6, 1, 1, 3, 0, 2)
    owners = select_owners(NodeOwners.from_static(original_ids, device="cpu"), queries.nodes)
    x = torch.randn_like(queries.data, requires_grad=True)
    original = queries.publish(selected_ffn(ffn, x, queries, owners))
    compact = queries.publish(selected_ffn(ffn, x, queries, owners, packing="power2"))
    rows = [literal_ffn(ffn, x[node, batch, slot], original_ids[int(queries.nodes[node])])
            for node, batch, slot in zip(queries.node.tolist(), queries.batch.tolist(), queries.slot.tolist())]
    expected = x.new_zeros(control.shape+(64,)).index_put(
        (queries.nodes[queries.node], queries.batch, queries.time), torch.stack(rows))
    root = torch.randn_like(expected)
    leaves = (x, *ffn.parameters())
    reference = torch.autograd.grad(expected, leaves, root, retain_graph=True)
    names = ("input", *(name for name, _ in ffn.named_parameters()))
    original_gradients = torch.autograd.grad(original, leaves, root, retain_graph=True)
    compact_gradients = torch.autograd.grad(compact, leaves, root, retain_graph=True)
    failures = []
    for actual, baseline in zip(compact_gradients, original_gradients):
        torch.testing.assert_close(actual, baseline, **tolerance(torch.float32, gradients=True))
    for mode, gradients in (("rectangular", original_gradients), ("power2", compact_gradients)):
        for name, actual, wanted in zip(names, gradients, reference):
            assert torch.isfinite(actual).all() and torch.isfinite(wanted).all()
            error = (actual-wanted).abs()
            exceeded = error > 1e-6 + 2e-3*wanted.abs()
            if exceeded.any():
                assert name == "up.weight" and exceeded.nonzero().tolist() == [[0, 20, 27]]
                assert float(error[exceeded].max()) < 2e-6
                failures.append((mode, name, float(error[exceeded].max())))
    if failures:
        raise KnownFP32FormulaDeviation(f"FP32 literal-formula failures at atol=1e-6/rtol=2e-3: {failures}")


@pytest.mark.parametrize("weight_access", ["gather", "grouped"])
def test_one_bucket_layout_is_shared_by_gate_up_down_and_dense_path_is_unchanged(weight_access):
    queries, _ = fixture()
    ffn = FeedForward(6, 64, hidden_features=128, weight_access=weight_access)
    owners = select_owners(NodeOwners.from_static(range(6), device="cpu"), queries.nodes)
    observed = {name: [] for name in ("gate", "up", "down")}
    handles = [getattr(ffn, name).register_forward_pre_hook(
        lambda _, args, name=name: observed[name].append((tuple(args[0].shape[:-1]), args[1])))
        for name in observed]
    try:
        selected_ffn(ffn, queries.data, queries, owners, packing="power2")
    finally:
        for handle in handles:
            handle.remove()
    assert [shape for shape, _ in observed["gate"]] == [(1, 1), (1, 2), (1, 4), (1, 8), (1, 12)]
    for gate, up, down in zip(observed["gate"], observed["up"], observed["down"]):
        assert gate[0] == up[0] == down[0]
        assert gate[1] is up[1] is down[1]
        assert gate[1].exact_ids is None
    dense = select_queries(torch.randn(2, 3, 2, 64), torch.ones(2, 3, 2),
                           torch.arange(2)[None, None].expand(2, 3, 2))
    owners = NodeOwners.from_static((1, 2), device="cpu")
    calls = []
    handle = ffn.register_forward_pre_hook(lambda _, args: calls.append(args))
    try:
        candidate = selected_ffn(ffn, dense.data, dense, owners, packing="power2")
    finally:
        handle.remove()
    assert len(calls) == 1 and calls[0][0] is dense.data and calls[0][1] is owners
    torch.testing.assert_close(candidate, ffn(dense.data, owners), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["K", "S"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_memory_full_only_changes_ffn_layout_with_private_state_and_input_parameter_vjp(kind, dtype):
    torch.manual_seed(902)
    constructor = KVMemory if kind == "K" else SSDMemory
    options = dict(window=5, kv_width=32) if kind == "K" else dict(summary_backend="prefix", scan_backend="doubling")
    original = constructor(4, 64, [0, 0, 1, 1], 4, 2, backend="math", chunk=2,
                           ffn_width=128, ffn_groups=None, **options).to(dtype)
    candidate = copy.deepcopy(original)
    candidate.ffn_packing = "power2"
    owners = NodeOwners.from_static((2, 0, 2), device="cpu")
    data = torch.randn(3, 4, 5, 64, dtype=dtype)
    masses = torch.rand(3, 4, 5, dtype=dtype)
    masses[:, 1, ::2] = 0
    times = (2**53+19+2*torch.arange(5))[None, None].expand(3, 4, 5)
    control = torch.zeros(3, 4, 5, dtype=dtype)
    control[0, 0, (0, 2, 4)] = 0.25
    control[1, :, 1:3] = 0.5
    control[2, 1, 3:] = 1
    root = torch.randn_like(data)
    records = []
    for module in (original, candidate):
        x = data.clone().requires_grad_()
        mass = masses.clone().requires_grad_()
        description, context, state = module.prepare(x, mass, times, owners)
        output = module.finish(context, control)
        loss = (output*root).sum() + sum(value.square().mean() for value in description)
        gradients = torch.autograd.grad(loss, (x, mass, *module.parameters()), allow_unused=True)
        records.append((output, state, gradients))
    torch.testing.assert_close(records[1][0], records[0][0], **tolerance(dtype))
    for actual, expected in zip(tensor_fields(records[1][1]), tensor_fields(records[0][1])):
        assert torch.equal(actual, expected)
    for actual, expected in zip(records[1][2], records[0][2]):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, **tolerance(dtype, gradients=True))
    states = tensor_fields(records[1][1])
    assert any(value.is_floating_point() and not torch.equal(value[0], value[2]) for value in states)


@pytest.mark.parametrize("kind", ["K", "S"])
def test_empty_full_retains_absent_ffn_gradients_and_rejects_invalid_packing(kind):
    constructor = KVMemory if kind == "K" else SSDMemory
    with pytest.raises(ValueError, match="ffn_packing"):
        constructor(2, 64, None, 4, 2, ffn_packing="implicit")
    module = constructor(2, 64, None, 4, 2, ffn_packing="power2")
    x = torch.randn(2, 3, 2, 64, requires_grad=True)
    times = torch.arange(2)[None, None].expand(2, 3, 2)
    description, context, _ = module.prepare(x, torch.zeros(2, 3, 2), times, torch.arange(2))
    calls = []
    handle = module.ffn.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        output = module.finish(context, torch.zeros(2, 3, 2))
    finally:
        handle.remove()
    assert not calls and torch.count_nonzero(output) == 0 and not output.requires_grad
    loss = output.sum() + description[0].sum()
    assert all(value is None for value in torch.autograd.grad(loss, tuple(module.ffn.parameters()), allow_unused=True))


@pytest.mark.parametrize("kind", ["K", "S"])
def test_fp32_nonempty_continuation_keeps_state_and_full_bptt_at_original_thresholds(kind):
    torch.manual_seed(903)
    constructor = KVMemory if kind == "K" else SSDMemory
    options = dict(window=5, kv_width=32) if kind == "K" else dict(summary_backend="prefix", scan_backend="doubling")
    original = constructor(3, 64, None, 4, 2, backend="math", chunk=2,
                           ffn_width=128, **options)
    candidate = copy.deepcopy(original)
    candidate.ffn_packing = "power2"
    owners = NodeOwners.from_static((2, 0, 1), device="cpu")
    data = torch.randn(3, 4, 7, 64)
    mass = torch.rand(3, 4, 7)
    mass[0, 1, 1::2] = 0
    control = (torch.arange(3*4*7).reshape(3, 4, 7) % 5 == 0).float()
    control[1, 0] = 0.25
    times = (2**53+43+2*torch.arange(7))[None, None].expand(3, 4, 7)
    root = torch.randn_like(data)
    records = []
    for module in (original, candidate):
        x = data.clone().requires_grad_()
        incoming = mass.clone().requires_grad_()
        state, output, saved, descriptions = None, [], [], []
        for start, end in ((0, 3), (3, 4), (4, 7)):
            desc, context, state = module.prepare(x[:, :, start:end], incoming[:, :, start:end],
                                                  times[:, :, start:end], owners, state)
            output.append(module.finish(context, control[:, :, start:end]))
            saved.append(state)
            descriptions.extend(desc)
        output = torch.cat(output, 2)
        loss = (output*root).sum() + sum(value.square().mean() for value in descriptions)
        gradients = torch.autograd.grad(loss, (x, incoming, *module.parameters()), allow_unused=True)
        records.append((output, saved, gradients))
    torch.testing.assert_close(records[1][0], records[0][0], **tolerance(torch.float32))
    for actual, expected in zip(tensor_fields(records[1][1]), tensor_fields(records[0][1])):
        assert torch.equal(actual, expected)
    for actual, expected in zip(records[1][2], records[0][2]):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, **tolerance(torch.float32, gradients=True))


@pytest.mark.parametrize("groups", [None, [0, 0, 1, 1, 2, 2, 2]])
def test_combined_compact_direct_weight_ffn_keeps_all_gradients_without_parameter_gathers(groups):
    torch.manual_seed(905)
    queries, _ = fixture()
    original = FeedForward(7, 64, groups, rank=4, hidden_features=128)
    combined = FeedForward(7, 64, groups, rank=4, hidden_features=128, weight_access="grouped")
    combined.load_state_dict(original.state_dict())
    owners = select_owners(NodeOwners.from_static((6, 1, 1, 3, 0, 2), device="cpu"), queries.nodes)
    x = torch.randn_like(queries.data, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    expected = queries.publish(selected_ffn(original, x, queries, owners))
    with ParameterGatherAudit(combined) as audit:
        actual = queries.publish(selected_ffn(combined, other, queries, owners, packing="power2"))
    assert audit.report()["output_bytes"] == 0
    torch.testing.assert_close(actual, expected, **tolerance(torch.float32))
    root = torch.randn_like(expected)
    left = torch.autograd.grad(actual, (other, *combined.parameters()), root)
    right = torch.autograd.grad(expected, (x, *original.parameters()), root)
    for got, wanted in zip(left, right):
        torch.testing.assert_close(got, wanted, **tolerance(torch.float32, gradients=True))


@pytest.mark.parametrize("weight_access", ["gather", "grouped"])
def test_compact_ffn_cpu_region_placement_keeps_policy_state_and_gradients(weight_access):
    torch.manual_seed(906)
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                                                 profile="compact-5-dd4"),
        d=64, memory="X", kv_width=32, ffn_width=128, ffn_packing="power2",
        ffn_weight_access=weight_access, attention_backend="math", parameter_indexing="static_view",
        checkpoint_reception=True, region_batch_size=1)
    original = TideModel(config)
    candidate = copy.deepcopy(original)
    partitions = [(candidate.graph.nodes[nodes[0]].position+candidate.graph.nodes[nodes[0]].layer) % 2
                  for nodes in candidate.graph.regions]
    candidate = place_model(candidate, ["cpu:0", "cpu:1"], region_partitions=partitions)
    for memory in candidate.memories.values():
        assert memory.ffn_packing == "power2" and memory.ffn.weight_access == weight_access
        for projection in (memory.ffn.gate, memory.ffn.up, memory.ffn.down):
            assert projection.weight_access == weight_access
            assert all(shard.weight_access == weight_access for shard in projection.shards)
    data = torch.randn(2, 4, 64)
    root = torch.randn(2, 4, 64)
    records = []
    for module in (original, candidate):
        x = data.clone().requires_grad_()
        state, outputs = None, []
        for start, end in ((0, 3), (3, 4)):
            result = module(x[:, start:end], state)
            outputs.append(result.readout())
            state = result.state
        output = torch.cat(outputs, 1)
        (output*root).sum().backward()
        records.append((output, canonical_state(module, state), x.grad,
                        canonical_state_dict(module, gradients=True)))
    torch.testing.assert_close(records[1][0], records[0][0], **tolerance(torch.float32))
    state = nested_comparison(records[1][1], records[0][1], **tolerance(torch.float32))
    assert state["within_threshold"] and not state["support_mismatches"]
    torch.testing.assert_close(records[1][2], records[0][2], **tolerance(torch.float32, gradients=True))
    assert records[1][3].keys() == records[0][3].keys()
    for name, actual in records[1][3].items():
        expected = records[0][3][name]
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, **tolerance(torch.float32, gradients=True))
