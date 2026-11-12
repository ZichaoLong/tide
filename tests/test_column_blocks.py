"""A real KV producer/region consumer shares storage without bypassing roots.

The independent comparison uses ordinary stack/cat and literal cross-node
polynomials. Regional mathematics and raw continuation have separate oracles.
"""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tidegraph.compare import equivalent
from tidegraph.weighted.column_blocks import concatenate
from tidegraph.weighted.kv_column_groups import read_kv_columns_many
from tidegraph.weighted.regional_packing import RegionalPacking
from tidegraph.weighted.packing_observation import observe
from test_column_query_groups import make_queries


def storage(value):
    return value.untyped_storage()._cdata


class NoNumericRepacking(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.cat.default, torch.ops.aten.stack.default,
                    torch.ops.aten.index_select.default):
            raise AssertionError(f"aligned Read -> region copied its numerical field: {func}")
        return func(*args, **(kwargs or {}))


def gradients(loss, roots):
    return torch.autograd.grad(loss, roots, retain_graph=True, allow_unused=True)


@pytest.mark.parametrize("read", ["sum", "qry"])
def test_actual_read_to_region_owner_and_cross_node_roots(read):
    _, requests, _, inputs = make_queries(read, positional=True)
    with observe() as counts:
        descriptors = read_kv_columns_many(requests, max_workspace_bytes=2**28)
    names = tuple(descriptors[0])
    for name in names:
        assert len({storage(row[name]) for row in descriptors}) == 1
    assert counts.metrics()["packing/projection/column_block_publish/calls"] == len(names)
    sources = dict(enumerate(descriptors))
    packing = RegionalPacking({i: torch.arange(len(request[1].mass)) for i, request in enumerate(requests)})
    with NoNumericRepacking():
        actual = packing.groups[0].pack(sources).fields
    literal = {name: torch.stack([row[name] for row in descriptors], 1) for name in names}
    equivalent(actual, literal)
    for name in names:
        assert storage(actual[name]) == storage(descriptors[0][name])

    # Cross-node coupling deliberately violates the producer's node-diagonal
    # numeric domain. The consumer's true roots must still all participate.
    def loss(fields):
        scores = fields["score"]
        return ((scores[:, 0]-2*scores[:, 1]+scores[:, 2]).square()
                +fields["evidence"].sum((1, 2))*fields["w"].sum(1)).sum()
    field_roots = tuple(t for row in descriptors for t in row.values() if t.requires_grad)
    input_roots = tuple(inputs.values())
    for factor in (1., 0.):
        equivalent(gradients(factor*loss(actual), field_roots), gradients(factor*loss(literal), field_roots))
        equivalent(gradients(factor*loss(actual), input_roots), gradients(factor*loss(literal), input_roots))


def test_ragged_region_borrows_one_bank_and_transpose_keeps_support():
    _, requests, _, _ = make_queries("sum")
    descriptors = read_kv_columns_many(requests, max_workspace_bytes=2**28)
    # Each node still has seven real rows, but the incidence is not aligned.
    domains = {0: torch.tensor([0, 1, 3, 5, 7, 9, 11]),
               1: torch.tensor([0, 2, 3, 6, 7, 10, 11]),
               2: torch.tensor([1, 2, 4, 5, 6, 8, 9])}
    sources = dict(enumerate(descriptors))
    group = next(g for g in RegionalPacking(domains).groups if g.candidates == 2)
    with observe() as counts:
        packed = group.pack(sources)
    assert counts.metrics()["packing/projection/column_block_borrow/calls"] == len(descriptors[0])
    for name, actual in packed.fields.items():
        literal = torch.stack([sources[node][name][row]
                               for node, row in zip(group.node_ids, group.local_rows)]).reshape_as(actual)
        equivalent(actual, literal)
    support = torch.tensor([node != 2 for node in group.node_ids]).reshape(len(group.rows), 2)
    bars = {"score": torch.zeros_like(packed.fields["score"])}
    result = packed.pullback(bars, support)
    assert result[2]["score"] is None
    assert result[0]["score"] is not None and result[1]["score"] is not None
    assert storage(result[0]["score"]) == storage(result[1]["score"])
    assert all(value["evidence"] is None for value in result.values())


def test_borrowed_storage_does_not_collapse_independent_alias_roots(dtype):
    owner = torch.arange(24, dtype=dtype).reshape(6, 4)
    roots = [owner[:2].detach().requires_grad_(), owner[2:5].detach().requires_grad_(),
             owner[5:].detach()]
    unused = owner[:2].detach().requires_grad_()
    actual = concatenate(roots)
    expected = torch.cat(roots)
    assert storage(actual) == storage(owner)
    for factor in (1., 0.):
        equivalent(gradients(factor*actual.square().sum(), (*roots[:2], unused)),
                   gradients(factor*expected.square().sum(), (*roots[:2], unused)))
        assert gradients(factor*actual.sum(), (unused,))[0] is None
    # Reordered/overlapping and equal-but-separate roots require real packing.
    for values in ((roots[1], roots[0]), (roots[0], roots[0]),
                   (roots[0], roots[0].detach().clone().requires_grad_())):
        joined = concatenate(values)
        assert storage(joined) != storage(owner)
        equivalent(joined, torch.cat(values))
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(actual.sum(), roots[:2], create_graph=True)
    with torch.no_grad():
        roots[2].add_(1)  # Even a frozen root shares the guarded owner lifetime.
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(actual.sum(), roots[:2])


@pytest.mark.parametrize("kind", ["empty_first", "single_row_stride", "zero_tail"])
def test_degenerate_contiguous_views_use_logical_strides(dtype, kind):
    owner = torch.arange(24, dtype=dtype).reshape(6, 4)
    if kind == "empty_first":
        pieces = (owner.as_strided((0, 4), (999, 7)), owner)
    elif kind == "single_row_stride":
        pieces = (owner.as_strided((1, 4), (999, 1)), owner[1:])
    else:
        pieces = (owner.as_strided((2, 0), (999, 7)), owner.as_strided((3, 0), (1, 17)))
    roots = tuple(t.detach().requires_grad_() for t in pieces)
    assert all(t.is_contiguous() for t in roots)
    actual, expected = concatenate(roots), torch.cat(roots)
    equivalent(actual, expected)
    assert storage(actual) == storage(owner)
    equivalent(gradients(actual.sum(), roots), gradients(expected.sum(), roots))
    with torch.no_grad():
        inference = concatenate(roots)
    assert not inference.requires_grad
    torch.testing.assert_close(inference, expected.detach())


def test_default_settle_reaches_the_owner_boundary(dtype):
    from test_fc_tensor_baseline import fixture
    from tidegraph.weighted.settle_columns import SettleColumnPlan
    binding, spec = fixture(dtype)
    values = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    with observe() as counts:
        actual = SettleColumnPlan(binding, spec).run(values)
    assert counts.metrics()["packing/projection/column_block_borrow/calls"] > 0
    # Every independent returned descriptor remains a real root in downstream
    # control/message computation; the result is not filled by reference replay.
    roots = tuple(t for node in actual.nodes.values() for t in node.descriptors.values() if t.requires_grad)
    grads = torch.autograd.grad(actual.readout(values).square().sum(), roots, allow_unused=True)
    assert any(grad is not None for grad in grads)
