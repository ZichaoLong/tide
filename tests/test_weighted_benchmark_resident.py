"""Actual resident gradients/SGD slots and exact disk resume for the consumer."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from resident_test_target import target_device, owner_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_benchmark_cases import Workload, fixture
from weighted_benchmark_runners import Resident, Eager
from weighted_benchmark_resume import check
from tidegraph.compare import equivalent
from weighted_benchmark_precision import promote, tolerances


@pytest.mark.parametrize("precision,cards", [("float32", 1), ("float16", 1), ("float32", 2)])
def test_resident_actual_training_observation_and_resume(tmp_path, precision, cards):
    device = torch.device(target_device(cards=cards))
    args = SimpleNamespace(mode="training", implementation="resident", placement="native",
        schedule="prefill", dtype=precision, scoring_dtype="float32", profile=False, steps=0,
        peer_devices=list(owner_devices(device, cards)) if cards > 1 else None,
        native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"])
    config = Workload("hb", "kv", width=4, samples=1, positions=2)
    def create():
        b, xs, span = fixture(config)
        b, xs = promote(b, xs, getattr(torch, precision))
        return b, xs, span
    atol, rtol = tolerances(getattr(torch, precision))
    b, xs, span = create()
    candidate = Resident(b, xs, span, config, args, device)
    try:
        actual = candidate.step(0)
        rb, rx, _ = create()
        reference = Eager(rb, rx, span, config, args, torch.device("cpu"), reference=True)
        expected = reference.step(0)
        if precision == "float16":
            from tidegraph.qualification_weighted_precision import assert_records
            assert_records(actual[0], expected[0], torch.float16)
        else:
            equivalent(actual[0], expected[0], check_device=False, atol=atol, rtol=rtol)
        # The consumer exports loss as a Python float; apply its existing
        # FP32 tensor tolerance, not the record codec's exact scalar identity.
        torch.testing.assert_close(torch.tensor(actual[1]), torch.tensor(expected[1]),
                                   atol=atol, rtol=rtol)
        equivalent(candidate.parameters(), reference.parameters(), check_device=False, atol=atol, rtol=rtol)
        equivalent(candidate.training_state(), reference.training_state(), check_device=False, atol=atol, rtol=rtol)
        stats = candidate.manifest()["last_backward_statistics"]
        for name in ("adjoint_construct_us", "root_seed_us", "adjoint_run_us", "parameter_reduce_us"):
            assert stats[name] >= 0
        result = check(candidate, create, config, args, device, tmp_path)
        assert result["state"] == "passed" and result["exact"]
    finally:
        candidate.close()


@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_resident_inference_complete_reference_and_disk_resume(tmp_path, precision):
    device = torch.device(target_device())
    args = SimpleNamespace(mode="inference", implementation="resident", placement="native",
        schedule="prefill", dtype=precision, scoring_dtype="float32", profile=False, steps=0,
        peer_devices=None, native_library=os.environ["TIDE_BUILD_DIR"],
        resident_library=os.environ["TIDE_RESIDENT_LIBRARY"])
    config = Workload("hb", "kv", width=4, samples=1, positions=2)
    def create():
        b, xs, span = fixture(config)
        b, xs = promote(b, xs, getattr(torch, precision))
        return b, xs, span
    b, xs, span = create()
    candidate = Resident(b, xs, span, config, args, device)
    try:
        actual = candidate.step(0)
        rb, rx, _ = create()
        reference = Eager(rb, rx, span, config, args, torch.device("cpu"), reference=True)
        expected = reference.step(0)
        atol, rtol = tolerances(getattr(torch, precision))
        if precision == "float16":
            from tidegraph.qualification_weighted_precision import assert_records
            assert reference.storage_reference
            assert_records(actual[0], expected[0], torch.float16)
        else:
            equivalent(actual[0], expected[0], check_device=False, atol=atol, rtol=rtol)
        torch.testing.assert_close(torch.tensor(actual[1]), torch.tensor(expected[1]), atol=atol, rtol=rtol)
        assert reference.optimizer is None and candidate.training_state() is None
        assert candidate.manifest()["training"] is False
        result = check(candidate, create, config, args, device, tmp_path)
        assert result["exact"] and "inference" in result["scope"]
        saved = torch.load(tmp_path/"resume.pt", map_location="cpu", weights_only=True)
        assert saved["schema"] == "tide-continuation-v5" and saved["optimizer"] is None
    finally:
        candidate.close()
