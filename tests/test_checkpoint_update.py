"""Metadata/resource guards and recovery wrapper only; no model execution."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.check_checkpoint_update as recovery
from tide import ModelConfig, TideModel, TopologyConfig


def limits(**overrides):
    result = dict(threads=4, max_parameters=200_000_000, artifact_budget_gib=8.,
                  restore_timeout_seconds=1800., chunk=32, seed=2718, device="cpu")
    return SimpleNamespace(**(result | overrides))


@pytest.mark.parametrize("override", [dict(artifact_budget_gib=10.1), dict(artifact_budget_gib=float("nan")),
    dict(restore_timeout_seconds=float("inf")), dict(max_parameters=200_000_001),
    dict(threads=33), dict(seed=-1), dict(device="cuda:0")])
def test_finite_resource_limits_precede_execution(override):
    with pytest.raises(ValueError):
        recovery.validate_limits(limits(**override))


def config():
    return ModelConfig(topology=TopologyConfig(n=2, layers=1), d=64, memory="K", kv_width=32)


def test_checkpoint_metadata_counts_on_meta_and_rejects_precision_or_size(tmp_path):
    cfg = config()
    with torch.device("meta"):
        model = TideModel(cfg)
    payload = dict(format="tide-instances-1", placement=None, config=cfg.to_dict(),
                   parameters=model.state_dict(), optimizer={"state": {0: {"step": torch.tensor(4.)}}})
    path = tmp_path/"metadata.pt"
    torch.save(payload, path)
    result = recovery.checkpoint_metadata(path, 200_000_000)
    assert result["parameters"] == sum(p.numel() for p in model.parameters())
    assert result["config"] == cfg.to_dict()
    with pytest.raises(ValueError, match="before real weight"):
        recovery.checkpoint_metadata(path, 1)
    payload["parameters"] = {k: v.double() if v.is_floating_point() else v for k, v in payload["parameters"].items()}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="FP32"):
        recovery.checkpoint_metadata(path, 200_000_000)


def test_retained_input_uses_last_saved_values_and_checks_production_identity(tmp_path):
    cfg, path = config(), tmp_path/"anchor.pt"
    original = torch.arange(128, dtype=torch.float32).reshape(1, 2, 64)
    torch.save(dict(config=cfg.to_dict(), source={"tide_content_sha256": "same"},
                    inputs=[torch.zeros_like(original), original]), path)
    value, metadata = recovery.retained_input(path, cfg.to_dict(), {"tide_content_sha256": "same"})
    assert torch.equal(value, original) and metadata["index"] == 1
    value.zero_()
    assert torch.equal(torch.load(path, mmap=True, weights_only=False)["inputs"][-1], original)
    with pytest.raises(ValueError, match="production package"):
        recovery.retained_input(path, cfg.to_dict(), {"tide_content_sha256": "changed"})
    with pytest.raises(ValueError, match="configuration"):
        recovery.retained_input(path, replace(cfg, ffn_packing="power2").to_dict(), {"tide_content_sha256": "same"})


def test_new_reference_explicitly_clears_gradients_and_sets_a_new_seed(monkeypatch):
    events = []
    optimizer = SimpleNamespace(zero_grad=lambda **kwargs: events.append(("clear", kwargs)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: events.append(("cpu_seed", seed)))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(manual_seed_all=lambda seed: events.append(("npu_seed", seed))), raising=False)
    recovery.initialize_reference_rng(optimizer, SimpleNamespace(type="npu"), 2718)
    assert events == [("clear", {"set_to_none": True}), ("cpu_seed", 2718), ("npu_seed", 2718)]


def test_worker_adds_its_own_allocator_boundary_without_changing_existing_checks(tmp_path, monkeypatch):
    bundle, output = tmp_path/"input.pt", tmp_path/"worker.json"
    torch.save(dict(options={"device": "cpu"}, scalar=torch.tensor(0.)), bundle)
    events = []
    monkeypatch.setattr(recovery, "configure_device", lambda device, threads: torch.device("cpu"))
    monkeypatch.setattr(recovery, "synchronize_all", lambda devices: events.append("sync"))
    def allocator(devices, reset=False):
        events.append("reset" if reset else "sample")
        return {}
    monkeypatch.setattr(recovery, "allocator_stats", allocator)
    expected = dict(state="within_threshold", pid=123, checks={"actual_update": {"comparison": {"exact": False}}})
    monkeypatch.setattr(recovery, "existing_restore_worker", lambda *args: dict(expected))
    result = recovery.restore_worker(bundle, output, device="cpu", threads=1)
    assert result["checks"] == expected["checks"]
    assert events == ["sync", "reset", "sample", "sync", "sample"]
    assert result["process_memory"]["before_checkpoint_load"] == {}
    assert json.loads(output.read_text())["checks"] == expected["checks"]


def allocator_row(allocated, reserved, peak_allocated, peak_reserved):
    return {"npu:0": {name: value*(1 << 30) for name, value in dict(
        allocated_bytes=allocated, reserved_bytes=reserved,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved).items()}}


def test_cpu_allocator_budget_is_an_explicit_noop():
    report = recovery.recovery_memory_budget(dict(before_cache_release={}, after_cache_release={}), {})
    assert report["within_budget"] and not report["violations"]
    assert report["conservative_overlap"] == {}


def test_parent_child_and_overlap_memory_failures_are_distinct():
    parent = dict(before_cache_release=allocator_row(2, 28, 24, 28),
                  after_cache_release=allocator_row(2, 3, 24, 28))
    child = allocator_row(1, 30, 25, 30)
    report = recovery.recovery_memory_budget(parent, child)
    assert report["within_budget"]
    assert report["conservative_overlap"]["npu:0"]["reserved"]["conservative_bound_bytes"] == 33*(1 << 30)
    parent["before_cache_release"]["npu:0"]["peak_allocated_bytes"] = 57*(1 << 30)
    refused = recovery.recovery_memory_budget(parent, child)
    assert refused["violations"][0]["scope"] == "parent_cumulative_peak"
    parent["before_cache_release"]["npu:0"]["peak_allocated_bytes"] = 24*(1 << 30)
    refused = recovery.recovery_memory_budget(parent, allocator_row(1, 58, 57, 58))
    assert any(row["scope"] == "child_cumulative_peak" for row in refused["violations"])
    # Both processes separately fit; their simultaneous reserved ownership does not.
    parent["after_cache_release"]["npu:0"]["reserved_bytes"] = 10*(1 << 30)
    refused = recovery.recovery_memory_budget(parent, allocator_row(1, 50, 40, 50))
    assert refused["parent"]["within_budget"] and refused["child"]["within_budget"]
    assert [row["scope"] for row in refused["violations"]] == ["conservative_parent_child_overlap"]
