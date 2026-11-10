"""Optional native diagnostics observe real work without changing query roots."""
import torch
from tidegraph.compare import equivalent


def test_native_stage_profile_reset_and_query_gradient(dtype):
    from test_weighted_dense_query import independent
    from test_weighted_query_vjp import native_query
    from tidegraph.full import FullInput
    from isolated_cases import vjp
    import _tide_native as core
    w, requests, roots = independent(dtype)
    rows = [FullInput(r.state, r.time, r.content, r.content.value.new_tensor(.3))
            for r in requests[1:3]]
    expected = [w.full_program.step(w, r, 1, "hard", 1.).value for r in rows]
    w.full_program.query_execution = "packed-v2"
    core.reset_stage_profile(True)
    try:
        result = native_query(w.full_program, w, rows, kind="full", slots=1)
        actual = vjp(sum(r.value.square().sum() for r in result), roots)
        metrics = core.stage_profile_metrics()
        assert metrics["detail/query_bind_calls"] > 0
        assert metrics["detail/query_gradient_worker_seconds"] > 0
        equivalent(actual, vjp(sum(t.square().sum() for t in expected), roots))
    finally:
        core.reset_stage_profile(False)
    assert all(value == 0 for value in core.stage_profile_metrics().values())
