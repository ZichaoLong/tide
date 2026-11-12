"""Node×B/L contractions: independent formulas, snapshots and root support."""
from dataclasses import replace
import math
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.weighted.kv_columns_state import KVStateColumns, prepare_kv_columns
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.kv_column_groups import read_kv_columns_many, full_kv_columns_many, query_group_bytes
from tidegraph.weighted.kv_columns import read_kv_columns, full_kv_columns
from tidegraph.weighted.column_query_ops import contract
from tidegraph.weighted.packing_observation import observe
from isolated_cases import vjp
from test_kv_columns import fixture


@pytest.mark.parametrize("kind", ["matmul", "attention", "logits"])
def test_contractions_literal_gradients_unused_alias_and_zero(kind, dtype):
    rows, roots, expected = [], {}, []
    torch.manual_seed(87)
    for i in range(3):
        if kind == "matmul":
            shapes = [(2, 3, 4), (2, 4, 2)]
        else:
            shapes = ([(2, 2, 3, 4)] if kind == "logits" else [(2, 2, 3, 2), (2, 2, 4, 2)])+[(2, 2, 4, 2), (2, 3, 4)]
        row = [(torch.rand(s, dtype=dtype)*.3+.2).requires_grad_() for s in shapes]
        if kind != "matmul":
            with torch.no_grad():
                row[-1][0, 0, :3] = 0
                row[-1][1, 1] = 0
        if i == 1:
            row[0] = rows[0][0]  # Real shared root, other fields stay private.
        rows.append(tuple(row))
        for j, x in enumerate(row):
            if all(x is not t for t in roots.values()):
                roots[f"{i}.{j}"] = x
        if kind == "matmul":
            expected.append(torch.stack([row[0][b]@row[1][b] for b in range(2)]))
        else:
            logits = row[0] if kind == "logits" else row[0]@row[1].transpose(-1, -2)/math.sqrt(2)
            v, weights = row[-2:]
            samples = []
            for b in range(2):
                heads = []
                for h in range(2):
                    qs = []
                    for q in range(3):
                        present = weights[b, q] > 0
                        qs.append((torch.softmax(logits[b, h, q, present]+weights[b, q, present].log(), 0)[:, None]
                                   *v[b, h, present]).sum(0) if present.any() else v.new_zeros(2))
                    heads.append(torch.stack(qs))
                samples.append(torch.stack(heads))
            expected.append(torch.stack(samples))
    actual = contract(kind, rows)
    equivalent(actual, expected)
    for zero in (False, True):
        loss = lambda xs: sum((x*torch.linspace(.2, .8, x.numel(), dtype=dtype).reshape_as(x)).sum() for x in xs[:2])
        equivalent(vjp(loss(actual), roots, zero=zero), vjp(loss(expected), roots, zero=zero))
        assert vjp(loss(actual), roots, zero=zero)["2.0"] is None
    with torch.no_grad():
        rows[0][0].add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        actual[0].sum().backward()


def make_queries(read, positional=False):
    cases = [fixture(torch.float64, read, window=17) for _ in range(3)]
    for case in cases[1:]:
        case[0].kernel.summary = cases[0][0].kernel.summary
        case[0].full_program.content.query = cases[0][0].full_program.content.query
    reads, fulls, roots = [], [], {}
    for i, (node, old, mass, content, layout, supplied) in enumerate(cases):
        if positional:
            node.kernel.config = replace(node.kernel.config, read_frequencies=(.17,) if read == "qry" else (),
                                         full_frequencies=((.11,), (.23,)), distance_slopes=(.03, .04), position_scale=5.)
            node.full_program.content.config = node.kernel.config
        left = KVStateColumns.from_states(old)
        window = prepare_kv_columns(node.kernel, layout, mass, content, left)
        controls = torch.tensor([.2, 0., .4, 0., .5, .6, .3], dtype=torch.float64, requires_grad=True)
        reads.append((node, window, "+", left)); fulls.append((node, window, controls))
        for name, value in [*node.named_parameters(), *[(n, v) for n, v in supplied.items() if n.split(".")[0].isdigit() or n in {"content", "mass"}], ("controls", controls)]:
            if value.requires_grad and all(value is not t for t in roots.values()):
                roots[f"{i}.{name}"] = value
    return cases, reads, fulls, roots


def scalar(cases, requests, mode):
    descs, fulls = [], []
    for case, request in zip(cases, requests):
        node, old, mass, content, layout, _ = case
        controls = request[2]
        desc, values = [], []
        for b, (a, z) in enumerate(layout.spans):
            previous = old[b]
            for j in range(a, z):
                h = Content(torch.cat((mass[j:j+1], content[j])))
                proposal = node.kernel.step(node, previous, h, int(layout.times[j]))
                snapshot = old[b] if mode == "0" else previous if mode == "-" else proposal
                desc.append(node.read_program.step(node, ReadInput(snapshot, int(layout.times[j]), h)))
                if controls[j].detach() > 0:
                    values.append(node.full_program.content(node.kernel, proposal, int(layout.times[j]), content[j]))
                previous = proposal
        descs.append({k: torch.stack([d[k] for d in desc]) for k in desc[0]})
        fulls.append(torch.stack(values))
    return descs, fulls


@pytest.mark.parametrize("read,mode,positional", [("sum", "+", False), ("qry", "+", False),
    ("qry", "-", True), ("qry", "0", True)])
def test_grouped_queries_against_independent_scalar_snapshots(read, mode, positional):
    cases, reads, fulls, roots = make_queries(read, positional)
    reads = [(n, w, mode, s) for n, w, _, s in reads]
    with observe() as counts:
        actual = read_kv_columns_many(reads, max_workspace_bytes=2**28)
        fresh = full_kv_columns_many(fulls, max_workspace_bytes=2**28)
    metrics = counts.metrics()
    assert metrics["packing/projection/tensor_block_bmm/calls"] > 0
    assert metrics["packing/projection/shared_block_mm/calls"] > 0
    expected, e_full = scalar(cases, fulls, mode)
    equivalent(actual, expected); equivalent([x.content for x in fresh], e_full)
    def loss(fields, outputs):
        values = [x for row in fields[:2] for x in row.values()]+outputs[:2]
        return sum((x*torch.linspace(.1, .7, x.numel(), dtype=x.dtype).reshape_as(x)).sum() for x in values)
    for zero in (False, True):
        a = vjp(loss(actual, [x.content for x in fresh]), roots, zero=zero)
        e = vjp(loss(expected, e_full), roots, zero=zero)
        equivalent(a, e)
        assert a["2.content"] is None
    for request, full in zip(fulls, fresh):
        equivalent(full.strength, request[2][full.rows])
        assert full.rows.tolist() == [0, 2, 4, 5, 6]


@pytest.mark.parametrize("budget_kind", ["single", "pair", "all"])
def test_query_byte_budget_keeps_private_shapes_and_empty_full(budget_kind):
    _, reads, fulls, _ = make_queries("qry")
    node, window, _, _ = reads[0]
    cost = query_group_bytes(node, window.layout, window.bank.layout, window.content.element_size())
    budget = {"single": 1, "pair": 2*cost, "all": 2**28}[budget_kind]
    with observe() as counts:
        got = read_kv_columns_many(reads, max_workspace_bytes=budget)
    equivalent(got, [read_kv_columns(n, w, read_mode=m, initial=s) for n, w, m, s in reads])
    assert (counts.metrics().get("packing/projection/tensor_block_bmm/calls", 0) == 0) == (budget_kind == "single")
    n, w, control = fulls[-1]
    fulls[-1] = n, w, torch.zeros_like(control, requires_grad=True)
    got = full_kv_columns_many(fulls, max_workspace_bytes=budget)
    expected = [full_kv_columns(*r) for r in fulls]
    equivalent([x.content for x in got], [x.content for x in expected])
    assert len(got[-1].rows) == 0
    for invalid in (0, -1, True, 1.5, 2**63):
        with pytest.raises(ValueError, match="positive int64"):
            read_kv_columns_many([], max_workspace_bytes=invalid)


def test_equal_padded_shapes_keep_different_sample_layouts_and_configurations():
    cases, reads, fulls, roots = make_queries("qry", positional=True)
    node, old, mass, content, _, supplied = cases[1]
    layout = SequenceLayout(torch.tensor([0, 3, 5, 7]),
                            torch.tensor([2**54+t for t in (10, 13, 29, 11, 18, 22, 35)]))
    cases[1] = node, old, mass, content, layout, supplied
    for i, (node, old, mass, content, layout, _) in enumerate(cases):
        if i == 2:
            node.kernel.config = replace(node.kernel.config, beta_short=.03)
            node.full_program.content.config = node.kernel.config
            node.read_program.memory_config = node.kernel.config
        left = KVStateColumns.from_states(old)
        window = prepare_kv_columns(node.kernel, layout, mass, content, left)
        reads[i] = node, window, "+", left
        fulls[i] = node, window, fulls[i][2]
    with observe() as counts:
        actual = read_kv_columns_many(reads, max_workspace_bytes=2**28)
        full = full_kv_columns_many(fulls, max_workspace_bytes=2**28)
    expected, outputs = scalar(cases, fulls, "+")
    equivalent(actual, expected); equivalent([f.content for f in full], outputs)
    metrics = counts.metrics()
    assert metrics["packing/bucket/kv_read_nodes/calls"] == 2
    assert metrics["packing/bucket/kv_full_nodes/calls"] == 2
    def loss(fields, values):
        return sum(d["score"].sum()+v.square().sum() for d, v in zip(fields[:2], values[:2]))
    equivalent(vjp(loss(actual, [f.content for f in full]), roots), vjp(loss(expected, outputs), roots))
