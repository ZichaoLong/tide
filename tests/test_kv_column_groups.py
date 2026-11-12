"""Node-group preparation: actual shapes, bounded packing and independent roots."""
import pytest
import torch
from tidegraph.content import Content
from tidegraph.compare import equivalent
from tidegraph.weighted.kv_columns_state import (
    KVStateColumns, prepare_kv_columns, prepare_kv_columns_many, preparation_group_bytes)
from tidegraph.weighted.packing_observation import observe
from test_kv_columns import fixture


def requests(dtype):
    cases = [fixture(dtype, read="qry", window=7) for _ in range(5)]
    # Sharing parameters is allowed, while each content/left/state root is private.
    cases[1][0].kernel.base = cases[0][0].kernel.base
    cases[1][0].kernel.summary = cases[0][0].kernel.summary
    output, roots = [], {}
    for i, (node, old, mass, value, layout, supplied) in enumerate(cases):
        if i == 3:
            mass = torch.zeros_like(mass, requires_grad=True)
        elif i == 4:
            mass = (mass.detach()+.1).requires_grad_()  # Different actual R, same driving layout.
        for name, p in node.named_parameters():
            if p.requires_grad and all(p is not other for other in roots.values()):
                roots[f"node{i}.{name}"] = p
        for name, leaf in supplied.items():
            if "." in name and name.split(".")[0].isdigit():
                roots[f"left{i}.{name}"] = leaf
        roots[f"mass{i}"], roots[f"content{i}"] = mass, value
        output.append((node.kernel, layout, mass, value, KVStateColumns.from_states(old)))
    return cases, output, roots


def local_scalar(node, old, mass, content, layout):
    result = []
    for b, (a, z) in enumerate(layout.spans):
        previous = old[b]
        for row in range(a, z):
            previous = node.kernel.step(node, previous,
                Content(torch.cat((mass[row:row+1], content[row]))), int(layout.times[row]))
        result.append(previous)
    return result


def loss(windows):
    return sum((getattr(window.bank, name)*torch.linspace(.13, .71, getattr(window.bank, name).numel(),
        dtype=window.content.dtype).reshape_as(getattr(window.bank, name))).sum()
        for window in windows for name in ("keys", "values", "summaries", "weights"))


def test_grouped_raw_state_and_none_connected_zero_roots(dtype):
    cases, rows, roots = requests(dtype)
    with observe() as counts:
        grouped = prepare_kv_columns_many(rows, max_workspace_bytes=2**27)
    # Only the three compatible positive-write operands take the four block maps.
    metrics = counts.metrics()
    assert metrics["packing/projection/private_bmm/calls"] == 4
    assert metrics["packing/projection/private_bmm/rows/3/calls"] == 4
    assert grouped[3].terminal is rows[3][-1]
    separate = [prepare_kv_columns(*row) for row in rows]
    for i, (a, b) in enumerate(zip(grouped, separate)):
        equivalent(a.terminal.states(), b.terminal.states())
        node, old, *_ = cases[i]
        equivalent(a.terminal.states(), local_scalar(node, old, rows[i][2], rows[i][3], rows[i][1]))
    # The third group output is unused. Its private matrix/input/left roots must
    # remain None even though it participates in the same physical BMM.
    for zero in (False, True):
        ga = torch.autograd.grad(loss(grouped[:2])*(0 if zero else 1), list(roots.values()), allow_unused=True, retain_graph=True)
        ge = torch.autograd.grad(loss(separate[:2])*(0 if zero else 1), list(roots.values()), allow_unused=True, retain_graph=True)
        equivalent(dict(zip(roots, ga)), dict(zip(roots, ge)))
        assert ga[list(roots).index("content2")] is None
        assert ga[list(roots).index("mass2")] is None


@pytest.mark.parametrize("limit", ["single", "pair", "all"])
def test_actual_byte_budget_splits_group_before_packing(dtype, limit):
    _, rows, _ = requests(dtype)
    cost = preparation_group_bytes(rows[0][0], int((rows[0][2] > 0).sum()), rows[0][3].element_size())
    budget = {"single": 1, "pair": 2*cost, "all": 2**27}[limit]
    with observe() as counts:
        actual = prepare_kv_columns_many(rows, max_workspace_bytes=budget)
    keys = counts.metrics()
    calls = keys.get("packing/projection/private_bmm/calls", 0)+keys.get("packing/projection/shared_block_mm/calls", 0)
    assert calls == (0 if limit == "single" else 4)
    if limit == "pair":
        assert keys["packing/projection/shared_block_mm/rows/2/calls"] == 2
        assert keys["packing/projection/private_bmm/rows/2/calls"] == 2
    for window, request in zip(actual, rows):
        equivalent(window.terminal.states(), prepare_kv_columns(*request).terminal.states())


def test_group_versions_and_invalid_budget_are_explicit(dtype):
    _, rows, _ = requests(dtype)
    grouped = prepare_kv_columns_many(rows, max_workspace_bytes=2**27)
    with torch.no_grad():
        rows[0][0].base.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        loss(grouped[:1]).backward()
    for value in (0, -1, True, 1.5, 2**63):
        with pytest.raises(ValueError, match="positive int64"):
            prepare_kv_columns_many([], max_workspace_bytes=value)
