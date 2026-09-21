import pytest
import torch
from dataclasses import replace
from tidegraph import Continuation
from tidegraph.reference import run
from single_graph_compare import compare
from single_graph_cases import fixture, PROFILES
from single_graph_checks import every_cut


@pytest.mark.parametrize("pool", PROFILES)
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", ["reference", "native-serial", "native-parallel", "native-packed", "cursor"])
@torch.no_grad()
def test_single_graph_complete_cuts(dtype, pool, clear, implementation):
    body, model, readout, read_model, xs, layers = fixture(dtype, pool, clear)
    adapter, result, bq, rq, outputs, _, cuts = every_cut(
        body, model, readout, read_model, xs, layers, 3, 3*(layers+1), implementation)
    assert any(size for _, _, _, size in cuts)
    assert all(b != 2 for b, _ in result.continuation.states)  # Entirely absent sample.
    assert all(t % (layers+1) == layers for _, t, _, _ in result.outputs)
    assert bq.cut == 3*layers and rq.cut == 3
    assert len(outputs) == len({(b, t) for b, t, _, _ in outputs})
    for e, origin in enumerate(adapter.edge_origin):
        if origin >= 0:
            assert adapter.model.edge_scale[e] is model.edge_scale[origin]
            assert adapter.model.agg_scale[e] is model.agg_scale[origin]
        else:
            assert adapter.model.edge_scale[e] is model.output_scale[0]
    for v, origins in enumerate(adapter.output_origin):
        for slot, origin in enumerate(origins):
            for prefix in ("emit_w_", "emit_b_"):
                assert adapter.model.nodes[v].extra[f"{prefix}{slot}"] is model.nodes[v].extra[f"{prefix}{origin}"]


@pytest.mark.parametrize("field", ["descriptor", "active"])
@torch.no_grad()
def test_single_norm_policy_still_rejects_corrupt_read_or_route(dtype, field):
    body, model, _, _, xs, _ = fixture(dtype, "add")
    result = run(body, model, Continuation(body.identity, 3), [x for x in xs if x.time < 2], 2, sealed_until=2)
    event = next(e for e in result.trace if e["node"] == 0)
    corrupt = dict(event)
    corrupt[field] = event[field]+.001 if field == "descriptor" else not event[field]
    changed = replace(result, trace=[corrupt if e is event else e for e in result.trace])
    with pytest.raises(AssertionError):
        compare(body, changed, result, "intentional corruption")


@torch.no_grad()
def test_single_read_view_does_not_forge_phase_occurrences(dtype):
    body, model, readout, read_model, xs, layers = fixture(dtype, "add")
    adapter, result, _, rq, _, _, _ = every_cut(body, model, readout, read_model, xs, layers, 3, 12)
    assert rq.ledger and any(position != time for position, time in rq.ledger.values())
    assert adapter.read_view(result.continuation).ledger == {}
