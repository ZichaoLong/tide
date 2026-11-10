"""The 1B/B32 refusal reserved records the selected inference never retains."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tidegraph.experiment.config import resolve
from tidegraph.experiment.record_policy import needs_trace
from tidegraph.planning.capacity import capacity_budget
from tidegraph.planning.record_capacity import event_reservations


def request():
    return dict(schema="tide-experiment-v1", model=dict(recipe=dict(topology="lh",
        target_parameters=1000000000, region_width=8, depth=3, family="kv")),
        task=dict(samples=32, positions=16, seed=7, input_pattern="staggered-v1"),
        execution=dict(device="npu:0", client="python-native", placement="mixed-c",
                       return_mode="outputs", dtype="float32"),
        measurement=dict(mode="inference", steps=1, warmup=1, validation="none"))


def test_b32_original_material_has_audited_event_reservation_without_allocating():
    r = resolve(request())
    budget = r["capacity_budget"]
    row = budget["per_device"]["npu:0"]
    c = row["components"]
    assert c["complete_record_estimate"] > 90_000_000_000
    assert c["live_record_estimate"] < 12_000_000_000
    assert row["estimated_peak_bytes"] < 32_000_000_000
    assert budget["unknown"] and not budget["automatic_admission"]
    # A caller without the runtime capability still reserves the old complete
    # records. Parameters, raw state, messages, recovery and host/disk stay put.
    old = capacity_budget(r["plan"], r["execution"], placement=r["payload_placement"],
        resources=r["resources"])
    before = old["per_device"]["npu:0"]["components"]
    assert old["per_device"]["npu:0"]["estimated_peak_bytes"] > 100_000_000_000
    assert {k:v for k,v in c.items() if k != "live_record_estimate"} == {
        k:v for k,v in before.items() if k != "live_record_estimate"}
    assert budget["host"] == old["host"] and budget["disk"] == old["disk"]


@pytest.mark.parametrize("change", ["training", "complete", "plugin", "greedy", "streaming"])
def test_unaudited_execution_keeps_full_record_budget(change):
    value = request()
    if change == "training": value["measurement"]["mode"] = "training"
    elif change == "complete": value["execution"]["return_mode"] = "complete"
    elif change == "plugin": value["task"]["plugin"] = dict(module="example_downstream_task")
    elif change == "greedy": value["execution"]["prefill_schedule"] = "greedy"
    else: value["execution"]["schedule"] = "streaming"
    row = resolve(value)["capacity_budget"]["per_device"]["npu:0"]
    assert row["components"]["live_record_estimate"] == row["components"]["complete_record_estimate"]


def test_irregular_region_owners_count_two_live_blocks_and_preserve_fallbacks():
    # Device a owns two rows in region 0, then rows in regions 1 and 2.
    # Its complete records total 170, while two largest blocks reserve 140.
    plan = dict(graph=dict(kind="ranked-local", node_regions=[0, 0, 1, 2, 2]))
    placement = dict(per_device={"a": dict(nodes=[0, 1, 2, 3]), "b": dict(nodes=[4])})
    spec = SimpleNamespace(client="pytorch", placement="cpu", return_mode="metrics",
                           schedule="prefill", prefill_schedule="window")
    rows, selected, _ = event_reservations(plan, spec, placement, [30, 40, 60, 40, 100],
                                          mode="inference", trace_records=False)
    assert selected and rows == {"a": 140, "b": 100}
    for field, value in [("client", "libtorch"), ("placement", "resident")]:
        other = deepcopy(spec); setattr(other, field, value)
        rows, selected, _ = event_reservations(plan, other, placement, [30, 40, 60, 40, 100],
                                              mode="inference", trace_records=False)
        assert not selected and rows == {"a": 170, "b": 100}
    plan["graph"]["kind"] = "positive-delay"
    assert not event_reservations(plan, spec, placement, [30, 40, 60, 40, 100],
                                 mode="inference", trace_records=False)[1]


def test_frozen_pre_policy_reference_and_plugin_keep_intermediate_roots():
    spec = resolve(request())["execution"]["effective"]
    assert not needs_trace(spec, {})
    assert needs_trace(spec, {}, reference=True)
    assert needs_trace(spec, dict(plugin={"module": "example"}))
    spec.pop("record_policy")
    assert needs_trace(spec, {})
