"""KV §2–4: packed snapshots vs independent scalar formulas/field roots."""
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.full import FullInput
from tidegraph.weighted.query_vjp import evaluate
from tidegraph.weighted.query_layout import Layout, chunks
from test_weighted_dense_query import independent
from test_weighted_query_vjp import native_query
from isolated_cases import vjp


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("kind", ["read", "full"])
def test_packed_snapshot_independent_fields(dtype, native, kind):
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
    p.query_execution = "packed-v2"
    result = (native_query if native else evaluate)(p, w, rows, kind=kind, slots=2)
    actual = [[r.value] if kind == "full" else [v for _, v in sorted(r.items())] for r in result]
    equivalent(actual, expected)
    for a, b in zip(expected, actual):
        for x, y in zip(a, b):
            assert x.requires_grad == y.requires_grad
            if x.requires_grad:
                equivalent(vjp(y, roots), vjp(x, roots))
    equivalent(vjp(actual[2][-1], roots, zero=True), vjp(expected[2][-1], roots, zero=True))


def test_layout_exact_time_visibility_and_budget():
    w, rows, _ = independent(torch.float64)
    state = rows[1].state
    clock = 2**54+37
    length = state.slots["weights"].numel()
    state = replace(state, slots={**state.slots, "times": torch.arange(length, dtype=torch.int64,
        device=state.value.device)+clock})
    layout = Layout([state, w.initial()], [clock+length-1, clock], 1)
    expected = torch.zeros_like(layout.visible)
    expected[0, length-1] = True
    assert torch.equal(layout.visible, expected)
    assert layout.age.dtype == torch.int64
    equivalent(layout.slot("values")[0, :length], state.slots["values"], atol=0, rtol=0)
    assert list(chunks([state]*4, 2048, budget=1)) == [[0], [1], [2], [3]]


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("family", ["fc", "hb", "pdg"])
def test_default_complete_training_and_exact_resume(dtype, native, family, tmp_path, request, monkeypatch):
    import test_weighted_dense_training as training
    from tidegraph import ExecutionPlacement
    from tidegraph.weighted.execution import eager_runner
    from test_weighted_dense_query import DEVICE
    def runner(b, native):
        execute = eager_runner(b, "native" if native else "python",
            ExecutionPlacement(preset="cpu" if DEVICE.type == "cpu" else "mixed-c",
                               scoring_dtype=str(dtype).split(".")[-1]))
        def run(*args, **kw):
            result = execute(*args, **kw)
            assert result.stats["query_packed-v2_nodes"] == len(b.graph.nodes)
            if native:
                assert result.stats.get("region_time_batch_frames", 0) > 0
            return result
        return run
    monkeypatch.setattr(training, "runner", runner)
    training.test_dense_complete_training_and_cuts(dtype, native, family, tmp_path, request)


def test_default_policy_is_capability_based_and_reproducible():
    from tidegraph.weighted.query_execution import execution_view
    from tidegraph.planning import ExecutionSpec
    from test_weighted_dense_training import fixture
    b, _, _ = fixture(torch.float32, "hb", "cpu")
    view = execution_view(b.model, "auto")
    assert all(n.read_program.query_execution == "packed-v2" for n in view.nodes)
    assert all(not hasattr(n.read_program, "query_execution") for n in b.model.nodes)
    assert list(view.parameters()) == list(b.model.parameters())
    old = ExecutionSpec(query_execution="legacy").record()
    new = ExecutionSpec().record()
    assert old["execution_id"] != new["execution_id"]
    assert new["effective"]["query_execution"] == "auto"
    for kw in (dict(schedule="streaming"), dict(packing="scalar"), dict(dtype="float16")):
        assert ExecutionSpec(**kw).record()["effective"]["query_execution"] == "legacy"


def test_masked_attention_keeps_first_positive_vjp_anchor():
    from tidegraph.weighted.query_packed import read
    from tidegraph.weighted.kv import weighted_read
    # A discarded first value must not destroy the meaningful small differences
    # between the actual positive values in the centered softmax derivative.
    logits = torch.tensor([[.2, .3, -.4]], requires_grad=True)
    values = torch.tensor([[[1e30, -1e30], [1., 1.], [1.01, 1.02]]], requires_grad=True)
    weights = torch.tensor([[0., .3, .4]], requires_grad=True)
    total = weights.sum(-1)
    expected = weighted_read(logits[0], values[0], weights[0], total[0])
    actual = read(logits, values, weights, total)[0]
    roots = dict(logits=logits, values=values, weights=weights)
    equivalent(actual, expected)
    equivalent(vjp(actual, roots), vjp(expected, roots))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_sum_memory_and_other_descriptors(dtype, native):
    from test_weighted_query_vjp import independent as fixture
    w, rows, roots = fixture(dtype, "kv", "sum")
    p = w.read_program
    expected = [p.step(w, r) for r in rows]
    p.query_execution = "packed-v2"
    actual = (native_query if native else evaluate)(p, w, rows)
    equivalent(actual, expected)
    for a, b in zip(actual, expected):
        for name in b:
            if b[name].requires_grad:
                equivalent(vjp(a[name], roots), vjp(b[name], roots))


def test_native_history_dependent_selection_keeps_causal_order(dtype):
    from tidegraph.native import Native
    from tidegraph.reference import run
    from tidegraph.compare import objective
    from region_cases import fixture
    g, m, q, xs, roots = fixture(dtype)
    expected = run(g, m, q, xs, 10, sealed_until=10)
    actual = Native(g, m, packed=True, algorithm="greedy", full_autograd="batched").run(q, xs, 10, sealed_until=10)
    equivalent(actual, expected)
    equivalent(vjp(objective(actual), roots), vjp(objective(expected), roots))
    assert actual.stats.get("region_time_batch_frames", 0) == 0
