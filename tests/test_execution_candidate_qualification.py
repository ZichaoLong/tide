"""Public W/P paths genuinely update and recover; integer/support errors survive."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.check_execution_candidates as qualification
from scripts.check_execution_candidates import (Artifacts, TOLERANCE, classified, file_sha256,
                                                parent_memory_before_child, run)
from scripts.check_training_paths import compare
from tide import ModelConfig, TopologyConfig, load_checkpoint


def tiny_config():
    return ModelConfig(topology=TopologyConfig(n=2, layers=1), d=64, memory="K",
        kv_width=32, sharing="region_shared", adapter_rank=8, ffn_width=128,
        region_batch_size=4, checkpoint_reception=True, attention_backend="math")


def test_all_execution_paths_start_once_continue_three_updates_and_recover_in_new_process(tmp_path):
    result = run(tiny_config(), directory=tmp_path, candidates=("baseline", "weight_access", "packing", "combined"),
        device="cpu", batch=1, length=4, chunk=2, steps=3, artifact_budget_gib=1., restore_timeout=90.)
    assert result["state"] == "within_threshold" and result["qualified"]
    assert result["tolerance"] == {"atol": 1e-6, "rtol": 2e-3}
    assert result["anchor_optimizer_steps"] == [1]
    for case in result["candidates"]:
        assert case["config"]["region_batch_size"] == 4 and case["config"]["checkpoint_reception"]
        assert case["checks"]["fixed_start"]["comparison"]["exact"]
        assert case["checks"]["gradient_rng_start"]["comparison"]["exact"]
        for step, row in enumerate(case["updates"], 2):
            assert row["optimizer_update"]["checked"]
            assert set(row["optimizer_update"]["after"].values()) == {step}
            assert row["summary"]["dtype_elements"]["torch.int64"] > 0
        assert case["observations"]["vjp1"]["summary"]["tensor_count"] > case["observations"]["inference"]["summary"]["tensor_count"]
        restore = case["restore"]
        assert restore["new_process"] and restore["pid"] != restore["original_process_pid"]
        assert restore["config_exact"] and restore["source_exact"] and restore["saved_data_exact"]
        assert restore["checks"]["next_forward"]["comparison"]["exact"]
        assert restore["actual_update"] and set(restore["optimizer_update"]["after"].values()) == {5}
        memory = restore["parent_memory_before_child"]
        assert memory["before_cache_release"] == memory["after_cache_release"] == {}
        assert not memory["cache_release_requested"]
        assert json.loads((tmp_path/case["name"]/"parent-memory-before-child.json").read_text()) == memory
        model, state, slots = load_checkpoint(tmp_path/case["name"]/"checkpoint.pt")
        assert model.config.to_dict() == case["config"] and state.position == 16 and slots["state"]
        assert not (tmp_path/case["name"]/"restore-input.pt").exists()
    retained = [row for row in result["artifacts"] if row["retained"]]
    assert len(retained) == 5  # one common anchor and four recovery checkpoints
    for row in retained:
        assert file_sha256(tmp_path/row["path"]) == row["sha256"]
    assert all(not (tmp_path/row["path"]).exists() for row in result["artifacts"] if not row["retained"])
    assert json.loads((tmp_path/"result.json").read_text())["qualified"]


def test_discrete_clock_support_and_numeric_failures_keep_original_thresholds():
    reference = {"state": {"times": torch.tensor([2**53+3], dtype=torch.int64), "weight": torch.tensor([0.])},
                 "input_gradient": torch.tensor([1.]), "gradients": {"weight": torch.tensor([1.])}}
    actual = copy.deepcopy(reference)
    actual["state"]["times"] += 1
    changed = compare(actual, reference, TOLERANCE)
    assert classified([changed])["state"] == "metadata_failed"
    assert changed["first_differences"]["time_or_index"]["coordinates"][0]["actual"] == 2**53+4
    actual = copy.deepcopy(reference)
    actual["state"]["weight"] += 1e-8
    assert classified([compare(actual, reference, TOLERANCE)])["state"] == "support_failed"
    actual = copy.deepcopy(reference)
    actual["input_gradient"] += .01
    assert classified([compare(actual, reference, TOLERANCE)])["state"] == "finite_deviation"
    actual["gradients"]["weight"][:] = float("nan")
    assert classified([compare(actual, reference, TOLERANCE)])["state"] == "nonfinite"


def test_artifact_budget_precedes_write_and_retirement_requires_original_identity(tmp_path):
    store = Artifacts(tmp_path, .001)
    with pytest.raises(MemoryError, match="budget"):
        store.save(tmp_path/"large.pt", torch.zeros(100_000), "bounded test")
    assert not (tmp_path/"large.pt").exists()
    store = Artifacts(tmp_path, .01)
    row = store.save(tmp_path/"small.pt", torch.arange(4), "replay")
    assert row["sha256"] == file_sha256(tmp_path/"small.pt")
    with (tmp_path/"small.pt").open("ab") as stream:
        stream.write(b"different")
    with pytest.raises(RuntimeError, match="identified"):
        store.retire(tmp_path/"small.pt")


def test_large_training_and_shortened_qualification_are_refused_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="three updates"):
        run(tiny_config(), directory=tmp_path/"short", steps=2)
    large = ModelConfig(topology=TopologyConfig(kind="multiscale", profile="5-dd4", n=64, hubs=8),
                        d=768, memory="X", kv_width=128)
    with pytest.raises(ValueError, match="200M"):
        run(large, directory=tmp_path/"large")


def test_npu_cache_boundary_synchronizes_and_records_live_owners_without_moving_them(monkeypatch):
    events = []
    before = {"npu:0": dict(allocated_bytes=1234, reserved_bytes=8192)}
    after = {"npu:0": dict(allocated_bytes=1234, reserved_bytes=2048)}
    observations = iter((before, after))
    monkeypatch.setattr(qualification, "synchronize_all", lambda devices: events.append("sync"))
    def allocator(devices):
        events.append("allocator")
        return next(observations)
    monkeypatch.setattr(qualification, "allocator_stats", allocator)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(empty_cache=lambda: events.append("empty_cache")), raising=False)
    result = parent_memory_before_child(SimpleNamespace(type="npu"))
    assert events == ["sync", "allocator", "empty_cache", "sync", "allocator"]
    assert result["before_cache_release"] == before and result["after_cache_release"] == after
    assert result["cache_release_requested"]


def test_earlier_mode_failure_keeps_its_witness_without_duplicating_a_passing_vjp(tmp_path, monkeypatch):
    original = qualification.observe
    def altered_inference(*args, **kwargs):
        observed, state, work = original(*args, **kwargs)
        if kwargs.get("grad") is False:
            segment = observed["forward"]["segments"][0]
            segment["values"] = segment["values"] + .02
        return observed, state, work
    monkeypatch.setattr(qualification, "observe", altered_inference)
    # This regression concerns witness retention; the independent process and
    # actual checkpoint path are exercised by the complete recovery test above.
    monkeypatch.setattr(qualification, "check_restore", lambda *args, **kwargs:
                        dict(state="within_threshold", checks={}))
    result = run(tiny_config(), directory=tmp_path, candidates=("baseline",),
                 device="cpu", batch=1, length=4, chunk=2, artifact_budget_gib=1.)
    case = result["candidates"][0]
    assert result["state"] == "finite_deviation" and not result["qualified"]
    assert classified([case["checks"]["inference_vs_grad_forward"]])["state"] == "finite_deviation"
    assert classified([case["checks"]["same_owner_vjp_repeat"]])["state"] == "within_threshold"
    assert (tmp_path/"baseline/inference.pt").is_file()
    assert (tmp_path/"baseline/vjp1.pt").is_file()
    assert not (tmp_path/"baseline/vjp2.pt").exists()
