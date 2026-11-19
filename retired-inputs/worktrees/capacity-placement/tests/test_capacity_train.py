"""Real small capacity updates without the duplicate qualification model."""
import json
import sys

import pytest
import torch

from scripts import check_placement
from tide import ModelConfig, TopologyConfig, canonical_state_dict


@pytest.mark.parametrize("kind,precision", [("exchange", "fp32"), ("multiscale", "bf16")])
def test_capacity_train_updates_detaches_and_counts_actual_storage(monkeypatch, kind, precision):
    topology = (TopologyConfig(kind="exchange", n=2, depth=1, hubs=1) if kind == "exchange" else
                TopologyConfig(kind="multiscale", n=2, profile="compact-5-dd4", hubs=1, timing="timed"))
    config = ModelConfig(topology=topology, d=64, memory="X", sharing="layer_shared")
    observed = {}
    place, adamw = check_placement.place_model, check_placement.placement_adamw
    def capture_model(*args, **kwargs):
        model = place(*args, **kwargs)
        observed.update(model=model, initial=canonical_state_dict(model))
        return model
    def capture_optimizer(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        observed["optimizer"] = optimizer
        return optimizer
    monkeypatch.setattr(check_placement, "place_model", capture_model)
    monkeypatch.setattr(check_placement, "placement_adamw", capture_optimizer)
    record = check_placement.capacity_train(config, ["cpu", "cpu"], precision,
                                           batch=1, length=3, steps=2, foreach=False)
    assert record["state"] == "completed" and record["completed_updates"] == 2
    assert record["optimizer"]["foreach"] is False
    assert record["initial_parameter_sha256"] and record["input_sha256"]
    if precision == "bf16":
        assert record["operator_witness"]["bf16_matrix_observed"]
    model, optimizer = observed["model"], observed["optimizer"]
    current = canonical_state_dict(model)
    assert any(not torch.equal(value, observed["initial"][name]) for name, value in current.items())
    expected_parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    expected_gradient_bytes = sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
    expected_slot_bytes = sum(value.numel()*value.element_size() for state in optimizer.state.values()
                              for value in state.values() if isinstance(value, torch.Tensor))
    previous_position = 0
    flush_periods = 0 if model.settle else (len(model.layer_nodes)-1+model.graph.period-1)//model.graph.period
    for index, row in enumerate(record["updates"]):
        assert row["finite"] and all(check["finite"] for check in row["finite_checks"].values())
        assert row["state_before_position"] == previous_position
        assert row["state_position"] == previous_position+3+flush_periods
        previous_position = row["state_position"]
        assert row["private_state_requires_grad_tensors"] == 0
        assert row["optimizer_step_counters"]["maximum"] == index+1
        assert row["parameter_update_witness"]["changed_elements"] > 0
        assert row["complete_seconds"] >= sum(row["phase_seconds"].values())
        assert row["work"]["input"]["candidates"] > 0
        assert row["actual_storage"]["parameters"]["payload_bytes"] == expected_parameter_bytes
        assert row["actual_storage"]["private_state"]["retained_storage_bytes"] > 0
        assert set(row["actual_storage"]["gradients"]["by_device"]["cpu"]["dtypes"]) == {"torch.float32"}
        if flush_periods:
            assert row["work"]["flush"]["positive_messages"] > 0
            assert row["flush_output_positive_strengths"] > 0
        else:
            assert row["work"]["flush"] is None
    storage = record["updates"][-1]["actual_storage"]
    assert storage["gradients"]["payload_bytes"] == expected_gradient_bytes > 0
    assert storage["optimizer_slots"]["payload_bytes"] == expected_slot_bytes > 2*expected_gradient_bytes


def test_capacity_training_defaults_and_nonforeach_scope(tmp_path, monkeypatch):
    captured = []
    def training(config, devices, precision, batch, length, steps, **kwargs):
        captured.append((batch, length, steps, kwargs))
        return dict(state="completed")
    monkeypatch.setattr(check_placement, "capacity_train", training)
    output = tmp_path/"train.json"
    monkeypatch.setattr(sys, "argv", ["check_placement", "--mode", "train", "--no-foreach", "--output", str(output)])
    assert check_placement.main() == 0
    assert len(captured) == 2
    assert all(row[:3] == (1, 8, 2) and row[3]["foreach"] is False for row in captured)
    assert json.loads(output.read_text())["arguments"]["warmup"] == 0
    monkeypatch.setattr(sys, "argv", ["check_placement", "--no-foreach", "--output", str(tmp_path/"invalid.json")])
    with pytest.raises(SystemExit):
        check_placement.main()


def test_capacity_stops_on_nonfinite_and_keeps_partial_record(monkeypatch):
    config = ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1), d=64, memory="X")
    original = check_placement._capacity_update
    def inject(*args, **kwargs):
        returned = original(*args, **kwargs)
        # A finite loss alone must not hide an invalid optimizer slot.
        optimizer = args[1]
        next(iter(optimizer.state.values()))["exp_avg"].reshape(-1)[0] = float("nan")
        return returned
    monkeypatch.setattr(check_placement, "_capacity_update", inject)
    progress = []
    record = check_placement.capacity_train(config, ["cpu", "cpu"], "fp32", batch=1, length=2,
                                           steps=2, foreach=False, progress=lambda row: progress.append(row["last_stage"]))
    assert record["state"] == "nonfinite" and len(record["updates"]) == 1
    assert record["not_run_updates"] == [1]
    assert record["updates"][0]["finite_checks"]["optimizer_slots"]["nonfinite_tensors"]
    assert progress[-1] == "stopped.nonfinite"


def test_storage_counts_aliases_and_chunked_finite_checks():
    value = torch.arange(17, dtype=torch.float32)
    items = [("prefix", value[:3]), ("tail", value[-2:])]
    storage = check_placement._storage_record(items)
    assert storage["payload_bytes"] == 5*4 and storage["retained_storage_bytes"] == 17*4
    value[-1] = float("inf")
    checked = check_placement._finite_record(items, chunk_elements=1)
    assert checked["nonfinite_tensors"] == ["tail"]


def test_qualification_complete_time_charges_stage_io_and_state_detach(monkeypatch):
    config = ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1), d=64, memory="X")
    model = check_placement.place_model(check_placement.TideModel(config), ["cpu", "cpu"])
    optimizer = check_placement.placement_adamw(model, lr=1e-4, foreach=False)
    clock = [0.0]
    original_detach = check_placement.detach_state
    def detach(state):
        clock[0] += 3
        return original_detach(state)
    def stage(name):
        clock[0] += 2
    monkeypatch.setattr(check_placement.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(check_placement, "detach_state", detach)
    state, result = check_placement._train(model, optimizer, torch.randn(1, 2, 64), None,
                                           "fp32", ["cpu", "cpu"], progress=stage)
    assert result["complete_seconds"] == 7
    assert result["between_phase_host_seconds"] == 4
    assert result["phase_seconds"]["optimizer_detach_sync"] == 3
    assert state.position == 2
    assert not any(value.requires_grad for _, value in check_placement._named_tensors(state))
