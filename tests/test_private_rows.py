"""Packed private roots preserve interval transposes, absence and aliases."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.private_rows import PrivateRows, join


def test_ragged_columns_use_one_leaf_and_scatter_only_participating_rows(dtype):
    parameter = torch.tensor(.7, dtype=dtype, requires_grad=True)
    original = [torch.arange(n*2, dtype=dtype).reshape(n, 2).requires_grad_() for n in (1, 3, 2)]
    pack = PrivateRows([parameter, *original], 1, 3)
    assert len(pack.leaves) == 2
    views = pack.views()[1:]
    joined = join(views)
    assert joined.untyped_storage().data_ptr() == pack.leaves[1].untyped_storage().data_ptr()
    grad = torch.autograd.grad(joined.square().sum()*pack.leaves[0], pack.leaves)
    result = [None]*4
    for i, value in enumerate(grad):
        pack.scatter(i, [0, 2], value, result)
    assert result[2] is None
    equivalent(result[1], 2*parameter*original[0])
    equivalent(result[3], 2*parameter*original[2])


@pytest.mark.parametrize("kind", ["adjacent", "partial", "overlap", "reordered", "strided"])
def test_copy_elision_uses_actual_contiguous_partition_not_equal_values(dtype, kind):
    leaf = torch.arange(24, dtype=dtype, requires_grad=True)
    rows = leaf.reshape(6, 4)
    values = {"adjacent": list(rows.unbind()), "partial": list(rows[1:4].unbind()),
              "overlap": [rows[0], rows[0]], "reordered": [rows[2], rows[0]],
              "strided": [rows[0, ::2], rows[1, ::2]]}[kind]
    actual = join(values, stack=True)
    expected = torch.stack(values)
    equivalent(actual, expected, atol=0, rtol=0)
    equivalent(torch.autograd.grad(actual.square().sum(), leaf, retain_graph=True),
               torch.autograd.grad(expected.square().sum(), leaf), atol=0, rtol=0)
    assert (actual.untyped_storage().data_ptr() == leaf.untyped_storage().data_ptr()) == (kind in {"adjacent", "partial"})


def test_ragged_bucket_crossing_owners_has_one_gather_transpose(dtype):
    leaf = torch.arange(80, dtype=dtype, requires_grad=True)
    rows = [leaf[0:8].reshape(2, 4), leaf[32:44].reshape(3, 4), leaf[64:80].reshape(4, 4)]
    actual = join(rows)
    equivalent(actual, torch.cat(rows), atol=0, rtol=0)
    names, pending, seen = [], [actual.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node); names.append(type(node).__name__)
        pending.extend(n for n, _ in node.next_functions)
    assert sum(name.startswith("IndexSelectBackward") for name in names) == 1
    assert not any(name.startswith("SliceBackward") for name in names)
    equivalent(torch.autograd.grad(actual.square().sum(), leaf, retain_graph=True),
               torch.autograd.grad(torch.cat(rows).square().sum(), leaf), atol=0, rtol=0)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_partial_field_roots_and_duplicate_rows(dtype, native):
    from test_weighted_dense_query import independent
    from test_weighted_query_vjp import native_query
    from tidegraph.weighted.query_vjp import evaluate
    from isolated_cases import vjp
    w, requests, roots = independent(dtype)
    rows = [requests[1], requests[2], requests[1]]
    expected = [w.read_program.step(w, r) for r in rows]
    w.read_program.query_execution = "packed-v2"
    actual = (native_query if native else evaluate)(w.read_program, w, rows)
    # Different row/field masks, including a completely unused duplicate row.
    for selected in ([(0, "score"), (1, "w")], [(1, "evidence")], [(0, "score"), (2, "score")]):
        def objective(results):
            return sum(results[i][key].square().sum() for i, key in selected)
        equivalent(vjp(objective(actual), roots), vjp(objective(expected), roots))
