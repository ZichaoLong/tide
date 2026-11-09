"""0770593 architecture §3–4 / KV §2–4: dense-v1 independent field roots.

Scalar formulas/schedules remain independent. None is checked as a value, not
coerced to zero. Different cache lengths must share a dense projection batch.
"""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.full import FullInput
from tidegraph.weighted.query_vjp import evaluate
from tidegraph.weighted.query_execution import branches, execution_view
from test_weighted_read_batch import fixture
from test_weighted_query_vjp import native_query
from isolated_cases import vjp

# Register the device before any autograd engine use (CANN 9.2 contract).
from tidegraph.runtime import resolve_device
DEVICE, _ = resolve_device(os.environ.get("TIDE_DENSE_DEVICE", "cpu"))


def independent(dtype):
    w, requests = fixture(dtype, "kv", "qry", False)
    w.to(DEVICE)
    roots = {n: p for n, p in w.named_parameters() if p.requires_grad}
    rows = []
    for i, request in enumerate(requests):
        def leaf(name, t):
            value = t.detach().to(DEVICE).requires_grad_(t.is_floating_point())
            if value.requires_grad:
                roots[f"row.{i}.{name}"] = value
            return value
        state = request.state
        if state is not None:
            state = replace(state, value=leaf("state", state.value),
                            slots={k: leaf(k, t) for k, t in state.slots.items()})
            if i == 3:
                with torch.no_grad():
                    state.slots["weights"].zero_()
        rows.append(replace(request, state=state, content=replace(request.content,
            value=leaf("content", request.content.value))))
    return w, rows, roots


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("kind", ["read", "full"])
def test_dense_ragged_independent_fields(dtype, native, kind, request):
    from dense_checks import Checks
    check = Checks(request.node.nodeid, known_evidence_rounding=DEVICE.type == "cpu" and dtype == torch.float32 and kind == "read")
    w, rows, roots = independent(dtype)
    if kind == "full":
        controls = [r.content.value.new_tensor(.3, requires_grad=True) for r in rows]
        roots.update({f"control.{i}": t for i, t in enumerate(controls)})
        rows = [FullInput(r.state if r.state is not None else w.initial(), r.time, r.content, t)
                for r, t in zip(rows, controls)]
        p = w.full_program
        expected = [[p.step(w, r, 2, "hard", 1.).value] for r in rows]
    else:
        p = w.read_program
        expected = [[v for _, v in sorted(p.step(w, r).items())] for r in rows]
    p.query_execution = "dense-v1"
    query = native_query if native else evaluate
    result = query(p, w, rows, kind=kind, slots=2)
    actual = [[r.value] if kind == "full" else [v for _, v in sorted(r.items())] for r in result]
    check(actual, expected, "forward")
    for a, b in zip(expected, actual):
        for x, y in zip(a, b):
            assert x.requires_grad == y.requires_grad
            if x.requires_grad:
                check(vjp(y, roots), vjp(x, roots), "field_vjp")
    def mixed(values):
        return sum(row[i % len(row)].square().sum() for i, row in enumerate(values))
    check(vjp(mixed(actual), roots), vjp(mixed(expected), roots), "mixed_vjp")
    check(vjp(actual[2][-1], roots, zero=True), vjp(expected[2][-1], roots, zero=True), "zero_vjp")
    assert query(p, w, [], kind=kind, slots=2) == []
    check.finish()


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("mode", ["dense-v1", "packed-v2"])
def test_dense_aliases_versions_and_unused_optimizer(dtype, native, mode):
    w, rows, roots = independent(dtype)
    w.full_program.content.query = w.kernel.read_q
    w.full_program.content.out = w.kernel.read_v
    p = w.full_program
    rows = [rows[1], replace(rows[2], content=rows[1].content), rows[5], rows[1]]
    rows = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.4)) for r in rows]
    roots = {**{n: p for n, p in w.named_parameters() if p.requires_grad},
             **{n: p for n, p in roots.items() if n.startswith("row.")}}
    expected = [p.step(w, r, 1, "hard", 1.).value for r in rows]
    p.query_execution = mode
    query = native_query if native else evaluate
    actual = [r.value for r in query(p, w, rows, kind="full", slots=1)]
    equivalent(vjp(sum(x.square().sum() for x in expected), roots),
               vjp(sum(x.square().sum() for x in actual), roots))
    optimizer = torch.optim.AdamW(w.parameters(), lr=.001)
    unused = w.read_program.score.matrix
    before = unused.detach().clone()
    actual[0].sum().backward()
    assert unused.grad is None
    optimizer.step()
    assert unused not in optimizer.state and torch.equal(before, unused)
    value = query(p, w, rows, kind="full", slots=1)[0].value
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(value.sum(), tuple(roots.values()), create_graph=True, allow_unused=True)
    value = query(p, w, rows, kind="full", slots=1)[0].value
    with torch.no_grad():
        w.full_program.content.up.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        value.sum().backward()


def test_dense_batches_across_lengths_and_keeps_branches(monkeypatch):
    w, rows, _ = independent(torch.float32)
    p = w.full_program
    p.query_execution = "dense-v1"
    selected = [rows[i] for i in (1, 2, 5)]
    requests = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.4)) for r in selected]
    from tidegraph.weighted import full_batch
    original, sizes = full_batch.project, []
    def observe(x, weight):
        if weight.shape == p.content.up.shape:
            sizes.append(len(x))
        return original(x, weight)
    monkeypatch.setattr(full_batch, "project", observe)
    evaluate(p, w, requests, kind="full", slots=1)
    assert sizes and set(sizes) == {3}
    assert branches(w.kernel, w, rows, "read") == [(False, False), (True, True),
        (True, True), (False, False), (False, False), (True, True), (False, False)]


def test_dense_execution_identity_and_rejections():
    from tidegraph.planning.execution import ExecutionSpec
    legacy = ExecutionSpec(query_execution="legacy").record()
    dense = ExecutionSpec(query_execution="dense-v1").record()
    assert "query_execution" not in legacy["requested"]
    assert dense["execution_id"] != legacy["execution_id"]
    assert ExecutionSpec().record()["requested"]["query_execution"] == "auto"
    assert ExecutionSpec(client="libtorch", query_execution="dense-v1")
    for fields in (dict(packing="scalar"), dict(schedule="streaming"), dict(workers=2), dict(dtype="float16"),
                   dict(device="npu:0", placement="resident")):
        with pytest.raises(ValueError, match="dense-v1"):
            ExecutionSpec(query_execution="dense-v1", **fields)
    from tidegraph import Graph, Node, Region
    from tidegraph.weighted import bind, KVConfig, RegionalConfig
    b = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=KVConfig(memory_read="qry"))
    view = execution_view(b.model, "dense-v1")
    assert view.nodes[0].read_program is not b.model.nodes[0].read_program
    assert all(x is y for x, y in zip(view.parameters(), b.model.parameters()))
    assert not hasattr(b.model.nodes[0].read_program, "query_execution")
    bad = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=KVConfig())
    with pytest.raises(ValueError, match="KV/qry"):
        execution_view(bad.model, "dense-v1")


def test_dense_fp64_directional_formula():
    # CPU-only mathematical anchor, also when an NPU module check is selected.
    w, rows = fixture(torch.float64, "kv", "qry", False)
    p = w.read_program
    p.query_execution = "dense-v1"
    parameter = p.evidence_weight
    original = parameter.detach().clone()
    direction = torch.linspace(-.1, .2, parameter.numel(), dtype=torch.float64).reshape_as(parameter)
    def scalar():
        return sum(p.step(w, r)["evidence"].sin().sum() for r in rows)
    result = sum(r["evidence"].sin().sum() for r in evaluate(p, w, rows))
    expected = (torch.autograd.grad(result, parameter)[0]*direction).sum()
    try:
        with torch.no_grad():
            parameter.copy_(original+1e-5*direction)
            plus = scalar()
            parameter.copy_(original-1e-5*direction)
            minus = scalar()
        torch.testing.assert_close((plus-minus)/2e-5, expected, rtol=2e-5, atol=2e-7)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
