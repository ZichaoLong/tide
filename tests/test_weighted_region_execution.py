"""Allocation §2–6 and Batch §2–4: online independent schedules/complete roots."""
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from test_weighted_scoring_execution import configured, CASES
from test_weighted_execution import variables
from test_weighted_query_training import runner
from isolated_cases import vjp


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_online_region_batch_records_support_and_vjp(dtype, case, native, monkeypatch):
    b, xs, stop = configured(dtype, case, "hb" if case % 2 else "pdg")
    xs = [External(i, x.port, x.position, x.time,
                   (x.value.detach()+x.value.new_tensor([0., .04*i, -.02*i])).requires_grad_())
          for i in range(2) for x in xs]
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    if not native:
        def forbidden(*a, **kw):
            raise AssertionError("online region batch replayed scalar region selection")
        for p in b.model.regions:
            monkeypatch.setattr(p, "step", forbidden)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert actual.stats["region_batch_calls"] > 0
    assert actual.stats["max_region_batch"] >= 2
    assert actual.stats["batched_region_frames"] > 0
    assert actual.stats.get("region_scalar_calls", 0) == 0
    roots = variables(b, xs)
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    first = expected.trace[0]
    other = next(e for e in actual.trace if (e["batch"], e["node"], e["time"]) == (first["batch"], first["node"], first["time"]))
    if first["control"].requires_grad:
        equivalent(vjp(first["control"], roots), vjp(other["control"], roots))


def test_generic_history_cannot_be_treated_as_independent():
    from tidegraph.region import RegionProgram, evaluate_batch
    from test_weighted_region_batch import fixture
    from types import SimpleNamespace
    p, rows = fixture(torch.float64, 0)
    p.history_independent = False
    r = rows[0]
    graph = SimpleNamespace(region_layouts={0:r.layout})
    model = SimpleNamespace(regions={0:p})
    frames = [(0, 0, r.time+i, r.candidates) for i in range(2)]
    with pytest.raises(ValueError, match="independent histories"):
        evaluate_batch(graph, model, None, frames)
    assert not RegionProgram.history_independent


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_region_adamw_checkpoint_and_retained_windows(dtype, native, tmp_path, monkeypatch):
    # Reuse the independent training/continuation protocol, with an extra region
    # capability gate on every whole/window result (including after restore).
    import test_weighted_query_training as training
    base = training.stats
    def stats(result):
        base(result)
        assert result.stats["batched_region_frames"] > 0
        assert result.stats.get("region_scalar_calls", 0) == 0
    monkeypatch.setattr(training, "stats", stats)
    training.test_query_adamw_and_pending_checkpoint(dtype, "hb", native, tmp_path)
    training.test_query_retained_cut_and_independent_roots(dtype, native, monkeypatch)
