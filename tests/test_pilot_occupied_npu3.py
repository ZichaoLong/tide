import copy
import os
import threading
import time

import pytest

from pilotlab import occupied_npu3 as occupied
from pilotlab.cutover_review3 import PLANNED_CONTROLLER_CONTINUATION
from pilotlab.shared_npu3 import load_queue_helper


@pytest.fixture
def policy():
    return {"run_id": "r3-test", "candidate_ids": [2, 4, 5, 7, 8, 11, 12, 13, 14],
            "shared_ids": [12, 13], "resident": occupied.resident_identity(os.getpid()),
            "boot_id": occupied.boot_id(), "expires_unix": time.time() + 300,
            "ordinary_limits": {"max_npu_devices": 8, "target_npu_devices": 8}}


def test_queue_keeps_fifo_second_sample_locks_and_driver_identities(tmp_path, monkeypatch, policy):
    helper = load_queue_helper()
    samples = occupied.bind_queue_admission(helper, {"temporary_npu": policy})
    queue, locks = tmp_path / "queue", tmp_path / "locks"
    job = dict(project="test", priority=0, npu_count=1, candidate_ids=set(policy["candidate_ids"]))
    head, _ = helper.register_job(queue, job_id="first", **job)
    follower, _ = helper.register_job(queue, job_id="second", **job)
    pid = os.getpid()
    observations = iter([(pid,), (pid, os.getppid()), (pid,), (pid,)])

    def sample(mapping):
        return {12: helper.DeviceStatus(6, 0, 12, "OK", 0, 4000, 65536, next(observations)),
                15: helper.DeviceStatus(7, 1, 15, "OK", 0, 0, 65536, ())}

    monkeypatch.setattr(helper, "snapshot", sample)

    def schedule(job_id):
        samples.clear()
        return helper.attempt_schedule(queue_dir=queue, device_lock_dir=locks, job_id=job_id,
            mapping={}, candidate_ids=job["candidate_ids"], npu_count=1, max_hbm_mib=4096,
            max_aicore_percent=5, sample_seconds=0.001, stop_event=threading.Event())

    try:
        assert schedule("second").reason == "waiting_turn"
        assert schedule("first").reason == "device_changed_during_sampling"
        result = schedule("first")
        assert result.physical_ids == (12,)
        try:
            assert helper.try_lock(locks, 12) is None
            assert [s["processes"] for s in samples[12]] == [[policy["resident"]]] * 2
            assert helper.DeviceStatus(6, 0, 12, "OK", 0, 4000, 65536, (pid,)).process_ids == (pid,)
        finally:
            helper.release_locks(result.device_locks)
    finally:
        helper.release_locks([head, follower])


@pytest.mark.parametrize("change", ["start", "uid", "boot", "expiry", "memory", "compute", "health", "device"])
def test_occupied_card_requires_all_conditions(policy, change):
    helper = load_queue_helper()
    pid = os.getpid()
    if change == "start":
        policy["resident"]["start_ticks"] += 1
    if change == "uid":
        policy["resident"]["uid"] += 1
    if change == "boot":
        policy["boot_id"] = "00000000-0000-0000-0000-000000000000"
    if change == "expiry":
        policy["expires_unix"] = time.time() - 1
    occupied.bind_queue_admission(helper, {"temporary_npu": policy})
    card = helper.DeviceStatus(6, 0, 2 if change == "device" else 12, "Fault" if change == "health" else "OK",
                               6 if change == "compute" else 0, 4097 if change == "memory" else 4000, 65536, (pid,))
    assert not card.is_free(max_hbm_mib=4096, max_aicore_percent=5)


def test_fresh_worker_checks_resident_identity_and_idle_pressure(policy):
    manifest = {"temporary_npu": policy}
    device = {"physical_id": 13, "used_bytes": 4 * 2**30, "aicore_percent": 0,
              "external_occupants": [policy["resident"]]}
    assert occupied.worker_wait_reason(manifest, device, []) is None
    assert occupied.worker_wait_reason(manifest, {**device, "used_bytes": 5 * 2**30}, []) == "resident_pressure"
    for occupant in ([{**policy["resident"], "start_ticks": 1}], [policy["resident"], policy["resident"]]):
        assert occupied.worker_wait_reason(manifest, {**device, "external_occupants": occupant}, [{}]) == "resident_permission_changed"


def test_ninth_slot_is_bound_to_one_reviewed_run(policy):
    conf = {**policy["ordinary_limits"], "deadline_unix": policy["expires_unix"]}
    review = {"schema": PLANNED_CONTROLLER_CONTINUATION, "temporary_npu": policy}
    occupied.apply_reviewed_policy(conf, review, "r3-test")
    assert conf["max_npu_devices"] == conf["target_npu_devices"] == 9
    with pytest.raises(ValueError):
        occupied.apply_reviewed_policy(copy.deepcopy(conf), review, "r3-next")
    occupied.apply_reviewed_policy(conf, {}, "r3-next")
    assert "temporary_npu" not in conf
    assert conf["max_npu_devices"] == conf["target_npu_devices"] == 8
