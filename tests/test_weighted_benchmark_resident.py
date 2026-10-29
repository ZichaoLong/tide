"""Actual resident gradients/SGD slots and exact disk resume for the consumer."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from resident_test_target import target_device

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_benchmark_cases import Workload, fixture
from weighted_benchmark_runners import Resident, Eager
from weighted_benchmark_resume import check
from tidegraph.compare import equivalent


def test_resident_actual_training_observation_and_resume(tmp_path):
    device = torch.device(target_device())
    args = SimpleNamespace(mode="training", implementation="resident", placement="native",
        schedule="prefill", dtype="float32", profile=False, steps=0, peer_devices=None,
        native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"])
    config = Workload("hb", "kv", width=4, samples=1, positions=2)
    create = lambda: fixture(config)
    b, xs, span = create()
    candidate = Resident(b, xs, span, config, args, device)
    try:
        actual = candidate.step(0)
        rb, rx, _ = create()
        reference = Eager(rb, rx, span, config, args, torch.device("cpu"), reference=True)
        expected = reference.step(0)
        equivalent(actual[0], expected[0], check_device=False)
        # The consumer exports loss as a Python float; apply its existing
        # FP32 tensor tolerance, not the record codec's exact scalar identity.
        torch.testing.assert_close(torch.tensor(actual[1]), torch.tensor(expected[1]),
                                   atol=1e-6, rtol=1e-5)
        equivalent(candidate.parameters(), reference.parameters(), check_device=False)
        equivalent(candidate.training_state(), reference.training_state(), check_device=False)
        stats = candidate.manifest()["last_backward_statistics"]
        for name in ("adjoint_construct_us", "root_seed_us", "adjoint_run_us", "parameter_reduce_us"):
            assert stats[name] >= 0
        result = check(candidate, create, config, args, device, tmp_path)
        assert result["state"] == "passed" and result["exact"]
    finally:
        candidate.close()
