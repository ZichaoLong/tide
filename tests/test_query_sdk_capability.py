"""SDK gate must verify head-only recurrent auto and reject scalar fallback."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_query_sdk import check_query_selection, check_selected_state_vjp, check_task_projection


@pytest.mark.parametrize("families,packed", [(('ssd', 'ssd'), 0), (('kv', 'kv'), 2), (('ssd', 'kv'), 1)])
def test_sdk_default_memory_and_head_counts(families, packed):
    material = dict(nodes=[{}, {}], memories=[{} if f == "kv" else dict(memory_family=f) for f in families])
    stats = {"query_packed-v2_nodes": packed, "query_legacy_nodes": 2-packed,
             "describe_batched_nodes": 2, "describe_scalar_nodes": 0}
    assert check_query_selection(material, stats)["query_legacy_nodes"] == 2-packed
    with pytest.raises(AssertionError, match="capability mismatch"):
        check_query_selection(material, {**stats, "describe_batched_nodes": 1, "describe_scalar_nodes": 1})
    with pytest.raises(AssertionError, match="capability mismatch"):
        check_query_selection(material, {**stats, "query_legacy_nodes": 3-packed})
    expected = check_query_selection(material, stats)
    live = {**stats, "batched_state_events": 8, "joint_state_vjp_events": 4 if packed else 0}
    scope = check_selected_state_vjp(expected, live)
    assert scope == ("recurrent-batched-state" if not packed else
                     "kv-hidden-sequence" if packed == 2 else "kv-hidden-and-recurrent-batched")
    with pytest.raises(AssertionError, match="batched state VJP"):
        check_selected_state_vjp(expected, stats)
    with pytest.raises(AssertionError, match="hidden sequence VJP"):
        check_selected_state_vjp(expected, {**live, "joint_state_vjp_events": 0 if packed else 4})


@pytest.mark.parametrize("families", [("kv", "kv"), ("ssd", "gated_delta"), ("affine", "delta")])
def test_sdk_columns_use_real_identity_and_work_not_legacy_vjp_counters(families):
    fixture = dict(nodes=[{}, {}], memories=[dict(memory_family=f) for f in families])
    execution = dict(interface="tensor-window-v1", schedule="settle-columns-v1", roots="columns", fallback_reason=None)
    stats = dict(column_windows=1, recorded_trace=0, state_column_calls=2,
                 region_column_calls=1, candidate_events=8, selected_events=6)
    expected = check_query_selection(fixture, stats, execution)
    assert expected["memory_families"] == list(families)
    assert expected["position_policy"] is None
    assert check_query_selection(fixture, stats, {**execution, "position_policy": "bounded-common-origin-v1"})["position_policy"] == "bounded-common-origin-v1"
    assert check_selected_state_vjp(expected, stats).startswith("column-task-backward;")
    for bad in (dict(state_column_calls=0), dict(state_column_calls=3), dict(region_column_calls=0),
                dict(selected_events=9), dict(column_windows=0), dict(recorded_trace=1)):
        with pytest.raises(AssertionError):
            check_query_selection(fixture, {**stats, **bad}, execution)
    for bad in (dict(interface="invented"), dict(roots="records"), dict(schedule="streaming"), dict(fallback_reason="unsupported"), dict(position_policy="invented")):
        with pytest.raises(AssertionError):
            check_query_selection(fixture, stats, {**execution, **bad})
    # A record run cannot acquire column qualification by reporting the same
    # task counts; it must still exercise its own packed/Describe algorithms.
    with pytest.raises(AssertionError, match="capability mismatch"):
        check_query_selection(fixture, stats, dict(interface="records"))


def test_sdk_column_projection_checks_each_real_update_and_independent_schedule():
    rows = [dict(step=i, loss=.01/(i+1), candidate_updates=8, full_calls=6,
                 messages=12, outputs=4, pending=0) for i in range(3)]
    actual = dict(metrics=rows)
    reference = dict(metrics=[dict(x) for x in rows],
                     effective_execution=dict(interface="records", schedule="streaming"))
    assert check_task_projection(actual, reference, "float64")["windows"] == 3
    for key, value in (("loss", .02), ("loss", float("nan")), ("full_calls", 5), ("pending", 1)):
        changed = [dict(x) for x in rows]; changed[1][key] = value
        with pytest.raises(AssertionError, match="differs"):
            check_task_projection(dict(metrics=changed), reference, "float64")
    with pytest.raises(AssertionError, match="domains differ"):
        check_task_projection(dict(metrics=rows[:1]), reference, "float64")
    with pytest.raises(AssertionError, match="independent record Streaming"):
        check_task_projection(actual, {**reference, "effective_execution": dict(interface="records", schedule="component-window")}, "float64")


@pytest.mark.parametrize("case", ["recurrent-grouped", "kv-context-gain"])
def test_sdk_describe_recipes_resolve_static_modules(case):
    from tidegraph.planning.recipes import prepare
    request = json.loads((Path(__file__).parent/"fixtures"/f"describe-{case}.json").read_text())
    plan, _ = prepare(request, samples=2, positions=2)
    assert plan["graph"]["nodes"] == 4 and plan["graph"]["regions"] == 2
    assert plan["samples"] == plan["positions"] == 2
    for region in plan["regions"]:
        if case == "recurrent-grouped":
            assert (region["local_score"], region["region_score"], region["region_evidence"]) == ("match", "att", "grouped")
        else:
            assert region["selection_mode"] == "direct_attention" and region["send_gain"] == "context"
