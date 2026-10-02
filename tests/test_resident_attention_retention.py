"""Immutable attention parameters, independent dynamic windows and exact budgets."""
import json
from functools import partial
import pytest
import torch
from tidegraph import ResidentTrainingLimits
from tidegraph.compare import equivalent
from test_resident_training import target
from resident_training_cases import inputs
from resident_cache_training import training_case
from resident_cache_roots import roots, terms
from resident_event_cases import runtime as event_runtime
from resident_fiber_cases import runtime as fiber_runtime


@pytest.mark.parametrize("family,schedule", [("pdg", "greedy"), ("timed-dag", "streaming"), ("settle", "greedy")])
@pytest.mark.parametrize("memory", ["event", "fiber"])
def test_attention_snapshot_budget_and_versions(target, family, schedule, memory, tmp_path):
    runtime = event_runtime if memory == "event" else partial(fiber_runtime, pooling="all-softmax")
    options = dict(runtime=runtime, roots=roots, terms=terms, root_modes=("all", "all", "all"))
    broad, bounded = tmp_path/"broad", tmp_path/"bounded"
    broad.mkdir(); bounded.mkdir()
    # Both executions independently check full CPU autograd/KV/state, aliases,
    # optimizer publication and disk resume. Every update has nonzero roots.
    records = training_case(target, family, schedule, "adamw", broad, **options)
    peak = records[0]["retained_bytes"]
    for s in records:
        assert s["retained_attention_bytes"] > 0 and s["retained_windows"] == 2
        assert s["retained_bytes"] == s["retained_dense_bytes"] == peak
        assert peak == 2*s["retained_window_bytes"]+s["retained_projection_bytes"]+s["retained_attention_bytes"]
    limits = ResidentTrainingLimits(retained_bytes=peak)
    assert training_case(target, family, schedule, "adamw", bounded, training_limits=limits, **options) == records
    (tmp_path/"retention.json").write_text(json.dumps(dict(memory=memory, family=family, statistics=records)))

    r = runtime(family, target, schedule)
    values = torch.ones(2, 2, 4)*.01
    with torch.no_grad(), r.training_session(2, limits=ResidentTrainingLimits(retained_bytes=peak-1)) as session:
        args, kw = inputs(session, values, 0, 1)
        session.advance_device(args, **kw)
        before, cut = session.result(), session.cut
        args, kw = inputs(session, values, 1, 2)
        with pytest.raises(ValueError, match="retained-window capacity"):
            session.advance_device(args, **kw)
        equivalent(before, session.result())
        assert session.cut == cut and session.retained_windows == 1
        session.detach()
        session.advance_device(args, **kw)
        assert session.cut > cut and session.retained_windows == 1
