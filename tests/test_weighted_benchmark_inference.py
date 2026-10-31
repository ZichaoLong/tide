"""Inference must verify independent complete windows and actual disk resume."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_benchmark_cases import Workload, fixture
from weighted_benchmark_precision import promote
from weighted_benchmark_runners import Eager
from weighted_benchmark_resume import check


def options(**overrides):
    values = dict(mode="inference", implementation="python", placement="native",
        schedule="prefill", dtype="float64", scoring_dtype="float64", profile=False,
        steps=0, packing="packed", workers=1, native_library=None, peer_devices=None)
    return SimpleNamespace(**(values | overrides))


def material(config, dtype):
    b, xs, span = fixture(config)
    b, xs = promote(b, xs, dtype)
    return b, xs, span


def test_inference_measurement_identity_requires_explicit_boolean_resume():
    from tidegraph.planning import MeasurementSpec
    record = MeasurementSpec(mode="inference", checkpoint_check=True).record("work", "execution")
    assert record["mode"] == "inference" and record["checkpoint_check"]
    with pytest.raises(ValueError, match="bool"):
        MeasurementSpec(mode="inference", checkpoint_check=1).record("work", "execution")


@pytest.mark.parametrize("storage_dtype", [torch.float16, torch.float64])
def test_inference_has_no_optimizer_master_or_parameter_gradients(tmp_path, monkeypatch, storage_dtype):
    from tidegraph import precision
    def forbidden(*args, **kwargs):
        raise AssertionError("inference constructed a training optimizer/master bank")
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    monkeypatch.setattr(precision, "FP32MasterOptimizer", forbidden)
    config = Workload("dag", "kv", width=4, samples=2, positions=2)
    args = options(dtype=str(storage_dtype).removeprefix("torch."),
                   scoring_dtype="float64" if storage_dtype == torch.float64 else "float32")
    create = lambda: material(config, storage_dtype)
    b, xs, span = create()
    initial = {name: value.detach().clone() for name, value in b.model.named_parameters()}
    candidate = Eager(b, xs, span, config, args, torch.device("cpu"))
    try:
        result, _ = candidate.step(0)
        assert result.trace and result.outputs
        assert candidate.optimizer is None and candidate.training_state() is None
        assert candidate.manifest()["master_dtype"] is None
        for name, value in b.model.named_parameters():
            assert value.grad is None and torch.equal(value, initial[name])
        resumed = check(candidate, create, config, args, torch.device("cpu"), tmp_path)
        assert resumed["exact"] and "inference" in resumed["scope"]
        saved = torch.load(tmp_path/"resume.pt", map_location="cpu", weights_only=True)
        assert saved["optimizer"] is None and saved["optimizer_layout"] is None
    finally:
        candidate.close()


def test_inference_resume_rejects_corrupted_pending_before_next_window(tmp_path, monkeypatch):
    import tidegraph.checkpoint as checkpoint
    config = Workload("pdg", "kv", width=4, samples=1, positions=1)
    create = lambda: material(config, torch.float64)
    b, xs, span = create()
    candidate = Eager(b, xs, span, config, options(), torch.device("cpu"))
    actual, _ = candidate.step(0)
    assert actual.continuation.pending
    load = checkpoint.load
    def damaged(*args, **kwargs):
        restored = load(*args, **kwargs)
        restored.pending[0].value.add_(1)
        return restored
    monkeypatch.setattr(checkpoint, "load", damaged)
    try:
        with pytest.raises(AssertionError):
            check(candidate, create, config, options(), torch.device("cpu"), tmp_path)
    finally:
        candidate.close()


def test_empty_inference_window_preserves_complete_cut_and_resume(tmp_path):
    from tidegraph.compare import equivalent
    config = Workload("dag", "kv", width=4, samples=2, positions=1)
    def create():
        b, _, span = material(config, torch.float64)
        return b, [], span
    b, xs, span = create()
    candidate = Eager(b, xs, span, config, options(), torch.device("cpu"))
    rb, rx, _ = create()
    reference = Eager(rb, rx, span, config, options(), torch.device("cpu"), reference=True)
    try:
        actual = candidate.step(0)
        equivalent(actual, reference.step(0), atol=0, rtol=0)
        assert actual[1] == 0 and not actual[0].trace and not actual[0].outputs
        assert actual[0].continuation.cut == span
        assert check(candidate, create, config, options(), torch.device("cpu"), tmp_path)["exact"]
    finally:
        candidate.close(); reference.close()


def test_staggered_inference_keeps_independent_records_and_exact_occurrence_resume(tmp_path):
    from collections import Counter
    from tidegraph.compare import equivalent
    from tidegraph.planning.recipes import prepare
    from tidegraph.planning.materialize import load, materialize
    plan, _ = prepare(dict(topology="fc", depth=2, region_width=2, width=4),
                      samples=3, positions=4, input_pattern="staggered-v1")
    path = tmp_path/"workload.json"
    path.write_text(json.dumps(plan))
    _, config = load(path)
    create = lambda: materialize(plan, max_bytes=10000000, dtype=torch.float64)
    b, xs, span = create()
    args = options(steps=1)
    candidate = Eager(b, xs, span, config, args, torch.device("cpu"))
    rb, rx, _ = create()
    reference = Eager(rb, rx, span, config, args, torch.device("cpu"), reference=True)
    try:
        for index in range(2):
            actual = candidate.step(index)
            expected = reference.step(index)
            equivalent(actual[0], expected[0], atol=1e-10, rtol=1e-8)
            torch.testing.assert_close(torch.tensor(actual[1], dtype=torch.float64),
                                       torch.tensor(expected[1], dtype=torch.float64), atol=1e-10, rtol=1e-8)
            counts = Counter(message.batch for message in actual[0].messages)
            assert len(set(counts.values())) > 1
        assert check(candidate, create, config, args, torch.device("cpu"), tmp_path)["exact"]
    finally:
        candidate.close(); reference.close()


def test_inference_entry_checks_all_windows_and_disk_resume(tmp_path, monkeypatch):
    import benchmark_weighted
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.chdir(ROOT)
    out = tmp_path/"inference"
    monkeypatch.setattr(sys, "argv", ["benchmark_weighted.py", "--device", "cpu",
        "--implementation", "python", "--mode", "inference", "--topology", "pdg",
        "--dtype", "float64", "--steps", "1", "--checkpoint-check", "--allow-dirty",
        "--output-dir", str(out)])
    benchmark_weighted.main()
    result = json.loads((out/"result.json").read_text())
    assert result["state"] == "passed" and result["post_measurement_complete_records"]
    assert result["exact_resume"]["exact"] and result["exact_resume"]["next_window"] == 2
    assert not result["autograd_enabled"] and not result["optimizer_constructed"]
    assert not result["parameter_values_checked"] and not result["optimizer_values_checked"]
    assert len(result["metrics"]) == 1 and result["metrics"][0]["pending"]
    assert any(row["phase"] == "exact_resume" for row in result["memory"]["phases"])


def test_standalone_inference_checks_disk_and_complete_forward(tmp_path):
    import subprocess
    binary = os.environ.get("TIDE_WEIGHTED_BENCHMARK")
    if not binary:
        pytest.skip("standalone qualification supplies TIDE_WEIGHTED_BENCHMARK")
    fixture_path = tmp_path/"fixture.json"
    made = subprocess.run([sys.executable, str(ROOT/"scripts/export_weighted_benchmark.py"),
        "--native-library", os.environ["TIDE_BUILD_DIR"], "--topology", "pdg",
        "--output", str(fixture_path)], capture_output=True, text=True, timeout=30)
    assert made.returncode == 0, made.stderr
    out = tmp_path/"run"
    measured = subprocess.run([binary, "--fixture", str(fixture_path), "--device", "cpu",
        "--dtype", "float64", "--mode", "inference", "--steps", "1", "--checkpoint-check", "true",
        "--output-dir", str(out)], capture_output=True, text=True, timeout=45)
    assert measured.returncode == 0, measured.stderr
    record = json.loads((out/"result.json").read_text())
    assert record["state"] == "passed" and record["post_measurement_complete_records"]
    assert record["exact_resume"]["exact"] and record["metrics"][0]["pending"]
    assert not record["autograd_enabled"] and not record["optimizer_constructed"]
