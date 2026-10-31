"""Static placement/capacity contracts, independent of accelerator availability."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest
from tidegraph.planning import ExecutionSpec, capacity_budget, plan_payload_placement
from tidegraph.planning.model import fingerprint, resolve_model, workload
from tidegraph.planning.recipes import prepare
from tidegraph.planning.topology import layered, topology


def fixture(*, regions=3, family="kv", source_mode="region", samples=2):
    return prepare(dict(topology="fc", depth=regions, region_width=2, width=8,
                        family=family, source_mode=source_mode), samples=samples, positions=2)


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_same_model_canonical_storage_and_arbitrary_device_counts(count):
    plan, inv = fixture(regions=7)
    devices = [f"cuda:{i}" for i in range(count)]
    placed = plan_payload_placement(plan, devices)
    assert placed == plan_payload_placement(plan, devices)
    assert placed["node_devices"][0] == devices[0]
    assert set(placed["node_devices"]) == set(devices)
    assert len(placed["co_location_groups"]) == 7
    assert sum(row["parameter_elements"] for row in placed["per_device"].values()) == inv["totals"]["unique"]
    for node in range(0, 14, 2):
        assert placed["node_devices"][node] == placed["node_devices"][node+1]
        assert placed["region_devices"][node//2] == placed["node_devices"][node]
    assert placed["model_id"] == plan["model_id"]
    assert not placed["automatic_admission"]


def test_explicit_map_rejects_shared_splitting_before_any_allocation():
    plan, _ = fixture()
    devices = ["npu:0", "npu:1", "npu:2"]
    with pytest.raises(ValueError, match="shared parameter.*co-locate"):
        plan_payload_placement(plan, devices, strategy="explicit", node_devices=devices*2)
    nodes = {str(v): devices[v//2] for v in range(6)}
    actual = plan_payload_placement(plan, devices, strategy="explicit", node_devices=nodes)
    rr = plan_payload_placement(plan, devices, strategy="region-round-robin")
    assert actual["node_devices"] == rr["node_devices"]
    assert actual["parameter_devices"] == rr["parameter_devices"]
    assert actual["placement_id"] != rr["placement_id"]
    with pytest.raises(ValueError, match="every node"):
        plan_payload_placement(plan, devices, strategy="explicit", node_devices={"0": devices[0]})
    with pytest.raises(ValueError, match="canonical"):
        plan_payload_placement(plan, devices, strategy="explicit", node_devices={"00": devices[0]})


def test_independent_nodes_may_split_a_region_without_splitting_its_attention():
    plan, inv = fixture(regions=1, family="delta", source_mode="node")
    devices = ["cuda:0", "cuda:1"]
    placed = plan_payload_placement(plan, devices, strategy="explicit", node_devices=devices)
    assert placed["region_devices"] == ["cuda:0"]
    for name in inv["owners"]:
        if name.startswith("nodes.1."):
            assert placed["parameter_devices"][name] == "cuda:1"
        elif name.startswith("regions.0."):
            assert placed["parameter_devices"][name] == "cuda:0"
    # Full-region selection remains one region, not two independent heads.
    assert plan["graph"]["node_regions"] == [0, 0]


def test_shared_component_cannot_use_sum_of_free_device_memory():
    plan, _ = fixture(regions=1)
    devices = ["cuda:0", "cuda:1", "cuda:2"]
    placed = plan_payload_placement(plan, devices)
    size = placed["largest_indivisible_group"]["static_bank_bytes"]
    with pytest.raises(ValueError, match="indivisible.*no declared device"):
        plan_payload_placement(plan, devices, device_limits={d: size-1 for d in devices})
    assert sum(size-1 for _ in devices) > size


def test_physical_edge_and_port_parameter_owners_are_not_coalesced():
    from tidegraph.planning.legacy_topology import graph_counts
    graph = layered("fc", depth=2, region_width=2)
    graph["edges"].append(list(graph["edges"][0]))
    graph["inputs"].append(0)
    graph["outputs"].append(2)
    graph["counts"] = graph_counts(graph)
    model, _ = resolve_model(graph, {}, {})
    plan, _ = workload(model)
    placed = plan_payload_placement(plan, ["cuda:0", "cuda:1"], strategy="region-round-robin")
    assert placed["cross_device_edge_ids"] == list(range(5))
    owners = placed["parameter_devices"]
    assert owners["edge_scale.4"] == owners["input_scale.2"] == "cuda:0"
    assert owners["agg_scale.4"] == owners["output_scale.2"] == "cuda:1"


def test_historical_execution_ids_remain_unchanged_and_new_axes_are_explicit():
    assert ExecutionSpec().record()["execution_id"] == "8357bacf48905d72e894302a7b86c53954a9c9d9fcd6bd7c1f99b1ccf34218c9"
    old = dict(client="python-native", device="cuda:0", placement="mixed-c", peer_devices=("cuda:0", "cuda:1"))
    assert ExecutionSpec(**old).record()["execution_id"] == "6f55d0e3a6308ab463ecf4509edec98a22abb1dccbd80e4597f32dd9a6d585d5"
    records = [ExecutionSpec(**old, **extra).record() for extra in (
        dict(node_devices=("cuda:0", "cuda:1")), dict(communication="blocking"),
        dict(communication="cpu_bridge"), dict(return_mode="outputs"), dict(return_mode="metrics"))]
    assert len({r["execution_id"] for r in records}) == len(records)
    assert all(r["schema"] == "tide-weighted-execution-v2" for r in records)
    with pytest.raises(ValueError, match="node_devices"):
        ExecutionSpec(**old, node_devices=("cuda:0", "cuda:2"))
    with pytest.raises(ValueError, match="device peer protocol"):
        ExecutionSpec(client="python-native", device="cuda:0", placement="resident", communication="cpu_bridge")


def test_training_budget_counts_canonical_half_masters_gradients_and_restore_per_card():
    plan, inv = fixture()
    devices = ("npu:0", "npu:1", "npu:2")
    placed = plan_payload_placement(plan, devices, dtype="float16")
    spec = ExecutionSpec(client="python-native", device=devices[0], peer_devices=devices,
                         node_devices=placed["node_devices"], dtype="float16", communication="blocking")
    budget = capacity_budget(plan, spec, placement=placed, mode="training", optimizer="adamw",
                             resources=dict(logical_windows=3, retained_windows=2))
    total = {key: sum(row["components"][key] for row in budget["per_device"].values()) for key in (
        "parameters", "payload_gradients", "fp32_masters", "master_gradients", "optimizer_slots")}
    assert total == dict(parameters=2*inv["totals"]["unique"], payload_gradients=2*inv["totals"]["trainable"],
                        fp32_masters=4*inv["totals"]["trainable"], master_gradients=4*inv["totals"]["trainable"],
                        optimizer_slots=8*inv["totals"]["trainable"])
    assert budget["communication"]["requested"] == "blocking" and budget["communication"]["effective"] is None
    for row in budget["per_device"].values():
        assert row["components"]["checkpoint_optimizer_preflight"] > 0
        assert row["components"]["checkpoint_restore_staging"] >= row["static_bank_bytes"]
        assert row["components"]["retained_state_snapshots"] > 0
        assert "workspace" in row["unknown_components"]
    assert budget["status"] == "requires-measurement" and not budget["automatic_admission"]


def test_finite_unbounded_kv_grows_and_minimal_return_never_hides_internal_cost():
    plan, _ = fixture(regions=4)
    short = capacity_budget(plan, ExecutionSpec(), resources=dict(logical_windows=1))
    complete = capacity_budget(plan, ExecutionSpec(), resources=dict(logical_windows=4))
    small = capacity_budget(plan, ExecutionSpec(return_mode="outputs"), resources=dict(logical_windows=4))
    # Fourth region has literal window=0; original times, not activity counts,
    # govern the other regions' windows. There is no silent cache truncation.
    assert complete["finite_scope"]["node_cache_rows"][-1] == 8
    assert short["finite_scope"]["node_cache_rows"][-1] == 2
    a, b = complete["per_device"]["cpu"]["components"], small["per_device"]["cpu"]["components"]
    assert b["requested_return_estimate"] < a["requested_return_estimate"]
    assert a["complete_record_estimate"] == b["complete_record_estimate"]
    assert a["private_state"] == b["private_state"]


@pytest.mark.parametrize("kind", ["dag", "pdg"])
def test_nonaligned_and_feedback_capacity_uses_finite_clock_not_a_reference_trace(kind):
    graph = topology(kind, layers=3, region_width=2)
    model, _ = resolve_model(graph, {}, {})
    plan, _ = workload(model, samples=2, positions=3)
    budget = capacity_budget(plan, ExecutionSpec(), resources=dict(logical_windows=2))
    scope = budget["finite_scope"]
    assert scope["candidate_upper"] > 0
    assert ("feedback" if kind == "pdg" else "nonaligned") in scope["candidate_count_method"]


def test_resource_refusal_and_tampered_placement_are_distinct_from_qualification():
    plan, _ = fixture(regions=1)
    spec = ExecutionSpec(device="cuda:0", placement="mixed-c")
    budget = capacity_budget(plan, spec, resources=dict(devices={"cuda:0": dict(free_bytes=1, reserve_bytes=0)}))
    assert budget["status"] == "static-refusal" and budget["static_rejections"]
    assert budget["communication"]["effective"] is None
    placed = plan_payload_placement(plan, ["cuda:0"])
    corrupt = deepcopy(placed)
    corrupt["per_device"]["cuda:0"]["parameter_bytes"] = 0
    corrupt["placement_id"] = fingerprint({k: v for k, v in corrupt.items() if k != "placement_id"})
    with pytest.raises(ValueError, match="storage/owner inventory"):
        capacity_budget(plan, spec, placement=corrupt)
    with pytest.raises(ValueError, match="workspace_bytes"):
        capacity_budget(plan, spec, resources=dict(workspace_bytes=True))
    with pytest.raises(ValueError, match="undeclared device"):
        capacity_budget(plan, spec, resources=dict(workspace_bytes={"cuda:1": 100}))
    with pytest.raises(ValueError, match="free_bytes exceeds"):
        capacity_budget(plan, spec, resources=dict(devices={"cuda:0": dict(capacity_bytes=1, free_bytes=2)}))
    with pytest.raises(ValueError, match="int64 time"):
        capacity_budget(plan, spec, resources=dict(logical_windows=2**62))


def test_inference_restore_capacity_distinguishes_cpu_decoded_and_native_staged_banks():
    # A 4.03B FP32 bank fits a 24GiB device with 4GiB reserved. C++ restores
    # a second complete device bank transactionally; Python copies CPU weights
    # into its existing bank. Both still need host/disk and runtime admission.
    plan, _ = prepare(dict(topology="fc", depth=2, region_width=2, width=8192),
                      samples=1, positions=2)
    budgets = {}
    for client in ("pytorch", "python-native", "libtorch"):
        spec = ExecutionSpec(client=client, device="cuda:0", placement="mixed-c")
        budgets[client] = capacity_budget(plan, spec, resources=dict(logical_windows=4,
            devices={"cuda:0": dict(capacity_bytes=24*1024**3, reserve_bytes=4*1024**3)}))
    for client in ("pytorch", "python-native"):
        assert not budgets[client]["per_device"]["cuda:0"]["reservation_exceeds_available"]
        assert budgets[client]["host"]["estimated_tensor_reservation_bytes"] > 4*16*1024**3
        assert not budgets[client]["automatic_admission"]
    assert budgets["libtorch"]["per_device"]["cuda:0"]["reservation_exceeds_available"]


def test_plan_and_capacity_work_without_torch_or_device_plugins():
    root = Path(__file__).resolve().parents[1]
    code = """
import json, sys
from tidegraph.planning import ExecutionSpec, plan_payload_placement, capacity_budget
from tidegraph.planning.recipes import prepare
p, _ = prepare(dict(topology='hb', width=8, region_width=2, depth=1))
devices = tuple('npu:'+str(i) for i in range(7))
placed = plan_payload_placement(p, devices)
e = ExecutionSpec(client='python-native', device=devices[0], peer_devices=devices, node_devices=placed['node_devices'])
b = capacity_budget(p, e, placement=placed)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print(json.dumps({'status': b['status'], 'devices': len(b['per_device'])}))
"""
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root/"python")), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(status="requires-measurement", devices=7)


def test_batch_128_and_512_scale_state_return_restore_without_changing_parameters():
    budgets = []
    for batch in (1, 128, 512):
        plan, _ = fixture(samples=batch)
        spec = ExecutionSpec(client="python-native", device="cuda:0", peer_devices=("cuda:0", "cuda:1", "cuda:2"))
        budgets.append(capacity_budget(plan, spec, resources=dict(logical_windows=3)))
    for device in budgets[0]["per_device"]:
        rows = [b["per_device"][device]["components"] for b in budgets]
        assert len({row["parameters"] for row in rows}) == 1
        for key in ("private_state", "complete_record_estimate", "pending_messages"):
            assert rows[1][key] == 128*rows[0][key]
            assert rows[2][key] == 512*rows[0][key]
        assert rows[2]["checkpoint_restore_staging"] >= rows[1]["checkpoint_restore_staging"]
        assert all(type(value) is int or value is None for row in rows for value in row.values())


def test_resident_physical_journal_and_coordinator_optimizer_are_explicit():
    model, _ = resolve_model(layered("fc", depth=1, region_width=1),
                             dict(width=2, summary_width=1, adapter_rank=1, ffn_width=4), dict(k=1))
    plan, inv = workload(model, samples=2, positions=1)
    spec = ExecutionSpec(client="python-native", device="npu:0", peer_devices=("npu:0", "npu:1", "npu:2"), placement="resident")
    limits = dict(queue=7, stages=4, cache=5, tensor_bytes=11209)
    result = capacity_budget(plan, spec, mode="training", resources=dict(resident_limits=limits))
    layout = result["resident_layout"]
    # One raw row: 33 FP32 + 9 int64; B2 -> 408 bytes. Persistent node
    # 3*408+16=1240, queue/control=930, four journal stages+boundary=9039.
    assert layout["node_private_preflight_bytes"]["npu:0"] == 1240
    assert layout["coordinator_queue_control_bytes"] == 930
    assert layout["coordinator_journal_bytes"] == 9039
    assert layout["preflight_total_bytes"] == 11209
    assert result["effective_owner_contract"]["regions"] == ["npu:0"]
    assert result["effective_owner_contract"]["optimizer"] == "npu:0"
    assert result["per_device"]["npu:0"]["components"]["fp32_masters"] == 4*inv["totals"]["trainable"]
    assert result["per_device"]["npu:1"]["components"]["fp32_masters"] == 0
    assert "resident_physical_role_banks" in result["per_device"]["npu:0"]["unknown_components"]
    refused = capacity_budget(plan, spec, resources=dict(resident_limits=dict(limits, tensor_bytes=11208)))
    assert refused["status"] == "static-refusal"
    assert "resident_limits.tensor_bytes" in refused["static_rejections"][0]
    assert not result["automatic_admission"]
    with pytest.raises(ValueError, match="prefill must be bool"):
        capacity_budget(plan, spec, resources=dict(resident_limits=dict(limits, prefill=1)))
