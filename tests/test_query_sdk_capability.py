"""SDK gate must verify head-only recurrent auto and reject scalar fallback."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_query_sdk import check_query_selection, check_selected_state_vjp


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
