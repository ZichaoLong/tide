"""Compact numerical blocks: ragged node histories and independent boundaries."""
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.kv_column_groups import read_kv_columns_many, full_kv_columns_many
from tidegraph.weighted.kv_columns import read_kv_columns, full_kv_columns
from tidegraph.weighted.kv_columns_state import KVStateColumns, prepare_kv_columns
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.packing_observation import observe
from tidegraph.weighted.kv_tensor_groups import Query, cost, groups
from test_column_query_groups import make_queries, scalar
from isolated_cases import vjp


@pytest.mark.parametrize("read,mode", [("sum", "+"), ("qry", "+"), ("qry", "-"), ("qry", "0")])
def test_unequal_queries_and_histories_share_one_numerical_block(read, mode):
    cases, requests, fulls, roots = make_queries(read, positional=True)
    node, old, mass, content, layout, supplied = cases[1]
    index = torch.tensor([0, 1, 3, 4, 6])
    layout = SequenceLayout(torch.tensor([0, 2, 4, 5]), layout.times[index])
    cases[1] = node, old, mass[index], content[index], layout, supplied
    fulls[1] = node, fulls[1][1], fulls[1][2][index]
    for i, (node, old, mass, content, layout, _) in enumerate(cases):
        left = KVStateColumns.from_states(old)
        window = prepare_kv_columns(node.kernel, layout, mass, content, left)
        requests[i] = node, window, mode, left
        control = fulls[i][2][:len(layout.times)]
        fulls[i] = node, window, control
    with observe() as counts:
        actual = read_kv_columns_many(requests, max_workspace_bytes=2**28)
        full = full_kv_columns_many(fulls, max_workspace_bytes=2**28)
    expected, values = scalar(cases, fulls, mode)
    equivalent(actual, expected); equivalent([f.content for f in full], values)
    metrics = counts.metrics()
    assert metrics["packing/bucket/kv_tensor_read/calls"] == 1
    assert metrics["packing/bucket/kv_tensor_read/rows"] == 3
    assert metrics["packing/bucket/kv_tensor_full/calls"] == 1
    def loss(desc, output):
        return desc[1]["score"].sum()+desc[0]["evidence"].square().sum()+output[1].square().sum()
    # Scalar event roots and whole-column roots have distinct connectivity for
    # an empty old slot joined to later writes. Values use the scalar schedule;
    # exact column-root identity also compares the independent one-node map.
    per_node = [read_kv_columns(n, w, read_mode=m, initial=s) for n, w, m, s in requests]
    single_full = [full_kv_columns(*r).content for r in fulls]
    for zero in (False, True):
        equivalent(vjp(loss(actual, [f.content for f in full]), roots, zero=zero),
                   vjp(loss(per_node, single_full), roots, zero=zero))
        a, e = vjp(loss(actual, [f.content for f in full]), roots, zero=zero), vjp(loss(expected, values), roots, zero=zero)
        # Empty-domain differences are explicitly checked, never applied to a
        # nonempty root, and are not a claim of full event-record equivalence.
        for name, root in roots.items():
            if root.numel() == 0:
                assert a[name] is None or a[name].numel() == 0
                assert e[name] is None or e[name].numel() == 0
            else:
                equivalent(a[name], e[name], path=name)


def test_padding_cost_is_recomputed_for_the_whole_proposed_group():
    _, requests, _, _ = make_queries("qry")
    n, w, mode, left = requests[1]
    short_layout = SequenceLayout(torch.tensor([0, 1, 2, 3]), w.layout.times[[0, 3, 6]])
    short = replace(w, layout=short_layout, mass=w.mass[:3], content=w.content[:3])
    a = Query(n, w, w.bank, torch.arange(len(w.layout.times)))
    b = Query(n, short, short.bank, torch.arange(3))
    # The long node determines padded Q for both, so individual sums undercount.
    individual, joint = cost([a])+cost([b]), cost([a, b])
    assert joint > individual
    assert list(groups([a, b], "read", individual)) == [[0], [1]]
    assert list(groups([a, b], "read", joint)) == [[0, 1]]


def test_new_block_does_not_call_node_numerical_helpers(monkeypatch):
    _, requests, fulls, _ = make_queries("qry")
    import tidegraph.weighted.kv_column_groups as old
    def forbidden(*args, **kwargs):
        raise AssertionError("tensor block split back into per-node arithmetic")
    for name in ("_queries", "describe", "linear", "contract"):
        monkeypatch.setattr(old, name, forbidden)
    read_kv_columns_many(requests, max_workspace_bytes=2**28)
    full_kv_columns_many(fulls, max_workspace_bytes=2**28)
