"""Independent storage alias/lifetime witnesses and small real update ledgers."""
import gc
import json
import subprocess
import sys

import pytest
import torch

from tide import ModelConfig, TideModel
from tide.topology import TopologyConfig
from benchmarks.memory_ledger import SavedStorageAudit, diagnose, snapshot, storage_inventory
from benchmarks.workloads import paired_input


def test_views_count_their_whole_retained_storage_once():
    base = torch.arange(16, dtype=torch.float32)
    stores, row = storage_inventory([base[:2], base[8:10]])
    assert len(stores) == 1
    assert row["payload_bytes"] == 16
    assert row["unique_storage_bytes"] == 64


def test_saved_objects_survive_unpack_with_retain_graph_and_release_after_backward():
    x = torch.arange(4, dtype=torch.float32, requires_grad=True)
    with SavedStorageAudit(existing=[x]) as audit:
        y = (x*x).sum()
        assert audit.report()["peak_live_saved_payload_bytes"] == 32
        assert audit.report()["peak_live_unique_storage_bytes"] == 16
        assert audit.report()["peak_live_additional_storage_bytes"] == 0
        y.backward(retain_graph=True)
        assert audit.report()["current_live_unique_storage_bytes"] == 16
        y.backward()
        assert audit.report()["current_live_unique_storage_bytes"] == 0
    assert torch.equal(x.grad, 4*x.detach())
    assert audit.report()["unpack_calls"] == 4


def test_saved_view_lifetime_is_released_when_an_unused_graph_is_dropped():
    x = torch.arange(8, dtype=torch.float32, requires_grad=True)
    with SavedStorageAudit() as audit:
        y = (x[:2]*x[2:4]).sum()
        assert audit.report()["current_live_unique_storage_bytes"] == 32
        del y
        gc.collect()
        assert audit.report()["current_live_unique_storage_bytes"] == 0


def test_snapshot_disjoint_ownership_does_not_sum_aliases():
    model = torch.nn.Linear(4, 2, bias=False)
    row = snapshot(model, None, model.weight[:1], dict(alias=model.weight[1:]), [], None, torch.device("cpu"))
    assert row["unique_storage_union_bytes"] == 32
    assert row["categories"]["parameters"]["unique_storage_bytes"] == 32
    assert row["categories"]["private_state"]["payload_bytes"] == 16
    assert row["categories"]["private_state"]["exclusive_unique_storage_bytes"] == 0
    assert row["categories"]["inputs"]["exclusive_unique_storage_bytes"] == 0
    assert row["live_allocator_minus_visible_union_bytes"] is None


@pytest.mark.parametrize("precision,mode", [("fp32", "grad_forward"), ("bf16", "train")])
def test_small_real_ledger_keeps_optimizer_state_and_saved_graph_separate(precision, mode):
    model = TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=2), d=64, sharing="region_shared"))
    x = paired_input(1, 2, 64, 1108)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True) if mode == "train" else None
    state, row = diagnose(model, x, mode=mode, precision=precision, optimizer=optimizer)
    assert state.position == 2
    assert row["saved_for_backward"]["peak_live_unique_storage_bytes"] > 0
    assert row["saved_for_backward"]["current_live_unique_storage_bytes"] == 0
    assert row["input_positions"] == 2
    assert row["actual_work"] is None
    assert row["finite"]
    released = row["phases"]["released_outputs_and_graph"]["categories"]
    parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    assert released["parameters"]["unique_storage_bytes"] == parameter_bytes
    if mode == "train":
        assert row["runtime"]["bf16_matrix_observed"]
        assert released["gradients"]["unique_storage_bytes"] == parameter_bytes
        assert released["optimizer"]["unique_storage_bytes"] == 2*parameter_bytes+4*len(list(model.parameters()))
        assert all(state["step"] == 1 for state in optimizer.state.values())
    else:
        assert released["gradients"]["unique_storage_bytes"] == 0
        assert released["optimizer"]["unique_storage_bytes"] == 0


def test_timed_diagnostic_continues_detached_state_and_retains_view_storage():
    config = ModelConfig(topology=TopologyConfig(kind="multiscale", n=2, hubs=1,
                         profile="compact-5-dd4", timing="timed"), d=64, memory="X", sharing="region_shared")
    model = TideModel(config)
    inputs = paired_input(1, 3, 64, 92)
    state, first = diagnose(model, inputs, mode="grad_forward", precision="fp32", chunk=1, decode=1)
    assert state.position == 4
    state, second = diagnose(model, inputs, mode="grad_forward", precision="fp32", state=state, chunk=1, decode=1)
    assert state.position == 8
    for row in (first, second):
        assert row["input_positions"] == 3
        assert row["actual_work"] is None
        assert row["saved_for_backward"]["current_live_unique_storage_bytes"] == 0
        private = row["phases"]["released_outputs_and_graph"]["categories"]["private_state"]
        assert private["unique_storage_bytes"] >= private["payload_bytes"]


def test_ledger_entry_records_two_real_bf16_timed_updates(tmp_path):
    config = tmp_path/"timed.json"
    config.write_text(json.dumps(dict(topology=dict(kind="multiscale", n=2, hubs=1,
                                    profile="compact-5-dd4", timing="timed"),
                                    d=64, memory="X", sharing="region_shared")))
    output = tmp_path/"memory.json"
    result = subprocess.run([sys.executable, "-m", "benchmarks.memory_ledger", "--config", str(config),
                             "--device", "cpu", "--precision", "bf16", "--mode", "train", "--length", "2",
                             "--prefill-chunk", "1", "--decode", "1", "--continue-state", "--segments", "2",
                             "--warmup", "0", "--threads", "4", "--output", str(output)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr+result.stdout
    record = json.loads(output.read_text())
    assert record["state"] == "completed_finite"
    assert [r["state_position"] for r in record["observations"]] == [4, 8]
    assert all(r["runtime"]["bf16_matrix_observed"] for r in record["observations"])
    assert all(r["saved_for_backward"]["current_live_unique_storage_bytes"] == 0 for r in record["observations"])
    assert record["scratch"]["measured_operator_scratch_bytes"] is None
