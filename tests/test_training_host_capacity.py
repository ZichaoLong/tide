"""Guard the actual training recovery footprint without allocating a 1B model."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from tidegraph.experiment.config import resolve
from tidegraph.planning import ExecutionSpec, capacity_budget
from tidegraph.planning.recipes import prepare


def test_measured_cuda_training_resume_peak_has_host_headroom():
    path = Path(__file__).resolve().parents[1]/"configs/capacity-training/cuda-scaled-loss.json"
    request = json.loads(path.read_text())["cases"][0]["config"]
    result = resolve(request)
    host = result["capacity_budget"]["host"]
    # 36baeb7/1a6b08f, result 01c4f8df..., original budget 35,261,194,368.
    # This is a measured lower bound, not an assertion that it is a universal
    # peak or that the independently observed FP32 numerical failure is fixed.
    observed_rss = 40_264_876_032
    assert host["estimated_tensor_reservation_bytes"] >= observed_rss+host["reserve_bytes"]
    request = deepcopy(request)
    request["resources"]["host_free_bytes"] = 36_000_000_000
    refused = resolve(request)["capacity_budget"]
    assert refused["host"]["reservation_exceeds_available"]
    assert not refused["automatic_admission"]
    assert refused["per_device"] == result["capacity_budget"]["per_device"]
    assert refused["disk"] == result["capacity_budget"]["disk"]


@pytest.mark.parametrize("payload_dtype", ["float16", "float32"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_snapshot_host_reservation_is_not_divided_by_device_count(payload_dtype, optimizer):
    plan, _ = prepare(dict(topology="fc", depth=3, region_width=2, width=8), samples=2, positions=2)
    hosts = []
    for count in (1, 2, 3):
        devices = tuple(f"npu:{i}" for i in range(count))
        spec = ExecutionSpec(client="python-native", device=devices[0], peer_devices=devices if count > 1 else (),
                             dtype=payload_dtype, placement="mixed-c")
        budget = capacity_budget(plan, spec, mode="training", optimizer=optimizer)
        assert budget["host"]["training_snapshot_extra_bytes"] > 0
        hosts.append(budget["host"]["training_snapshot_extra_bytes"])
    assert len(set(hosts)) == 1


def test_inference_standalone_and_no_checkpoint_do_not_acquire_python_recovery_copies():
    plan, _ = prepare(dict(topology="fc", depth=2, region_width=2, width=8))
    for client, mode, checkpoint in (("python-native", "inference", True),
                                     ("libtorch", "training", True),
                                     ("python-native", "training", False)):
        spec = ExecutionSpec(client=client, device="cuda:0", placement="mixed-c")
        budget = capacity_budget(plan, spec, mode=mode, checkpoint=checkpoint)
        assert budget["host"]["training_snapshot_extra_bytes"] == 0
