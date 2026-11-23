"""Fake workers and fake NPU observations only; no real device/service starts."""
from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from pilotlab import shared_npu3 as shared


FAKE_WORKER = r'''
import json, os, signal, sys, time
from pathlib import Path
output, config = Path(sys.argv[1]), json.loads(sys.argv[2])
if config.get("orphan"):
    pid = os.fork()
    if pid == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(20)
        os._exit(0)
    (output / "orphan.pid").write_text(str(pid))
time.sleep(config.get("sleep", 0.1))
result = {"status": config.get("status", "completed"),
          "visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
          "pythonpath": os.environ.get("PYTHONPATH"),
          "hashseed": os.environ.get("PYTHONHASHSEED"),
          "affinity": sorted(os.sched_getaffinity(0))}
if "failure_class" in config:
    result["failure_class"] = config["failure_class"]
(output / "result.json").write_text(json.dumps(result))
print("fake worker done", flush=True)
raise SystemExit(config.get("code", 0))
'''


def manifest(tmp_path, *, backend="cpu", configs=None):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    configs = configs or [{}, {}]
    workers = []
    for index, config in enumerate(configs):
        output = tmp_path / f"worker-{index}"
        workers.append({"id": f"w{index}", "argv": [sys.executable, str(script), str(output), json.dumps(config)],
            "cwd": str(tmp_path), "output": str(output), "cpu_threads": 1,
            "reserved_rss_bytes": 16 * 2**20, "reserved_hbm_bytes": 128 * 2**20,
            "env": {"PYTHONHASHSEED": "17", "PYTHONPATH": str(tmp_path)}})
    return {"schema": shared.SCHEMA, "job_id": "shared-test", "backend": backend,
        "sharing_authorization": "round3-user-approved", "max_parallel": 8, "max_cpu_threads": 8,
        "max_rss_bytes": 2**30, "max_hbm_bytes": 2**30, "max_device_seconds": 30.,
        "deadline_unix": time.time() + 30, "poll_seconds": .02, "sharing_sample_seconds": .01,
        "resource_interval_seconds": .05, "admission_sample_seconds": .01,
        "stop_grace_seconds": .1, "kill_grace_seconds": 1., "workers": workers}


class FakeLease:
    """Real temporary flock, fake device metadata, no accelerator calls."""
    def __init__(self, request, output, stop):
        self.request, self.output, self.stop = request, output, stop
        self.physical_id = 7
        self.acquired_unix = self.acquired_monotonic = None
        self.released_unix = self.released_monotonic = None
        self.released = False
        self.runner = None
        self.handle = None
        self.samples = 0

    def acquire(self):
        self.handle = (self.output / "fake-device.lock").open("a+")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.acquired_unix, self.acquired_monotonic = time.time(), time.monotonic()

    def snapshot(self):
        self.samples += 1
        return {"physical_id": 7, "health": "OK", "used_bytes": 0, "total_bytes": 2**30, "process_ids": []}

    def pass_fds(self):
        return (self.handle.fileno(),)

    def release(self, status):
        if self.runner is not None:
            assert not self.runner.owned(), "lease must outlive every owned descendant"
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        self.released = True
        self.released_unix, self.released_monotonic = time.time(), time.monotonic()


def runner_with_fake_lease(request, output):
    runner = shared.GroupRunner(request, output, lease_factory=FakeLease)
    def factory(*args):
        value = FakeLease(*args)
        value.runner = runner
        return value
    runner.lease_factory = factory
    return runner


def test_cpu_group_preserves_independent_records_environment_and_affinity(tmp_path, monkeypatch):
    request = manifest(tmp_path, configs=[{"sleep": .2}, {"sleep": .2}])
    cpu = min(os.sched_getaffinity(0))
    for worker in request["workers"]:
        worker["cpu_affinity"] = [cpu]
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "inherited-not-overridden")
    monkeypatch.setattr(shared, "load_queue_helper", lambda: pytest.fail("CPU group must not load the NPU helper"))
    runner = shared.GroupRunner(request, tmp_path / "group")
    result = runner.run()
    assert result["status"] == "completed", result
    assert result["physical_device_seconds"] == 0
    assert result["peak_parallel_workers"] == 2
    for record in result["workers"].values():
        assert record["status"] == "completed" and record["return_code"] == 0
        assert record["pid"] > 0 and record["start_ticks"] > 0
        assert record["finished_unix"] >= record["started_unix"]
        payload = json.loads(Path(record["result_path"]).read_text())
        assert payload["visibility"] == "inherited-not-overridden"
        assert payload["hashseed"] == "17" and payload["pythonpath"] == str(tmp_path)
        assert payload["affinity"] == [cpu]
        assert Path(record["log"]).read_text().strip() == "fake worker done"


def test_group_card_seconds_are_charged_once_while_worker_seconds_overlap(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": .4}, {"sleep": .4}])
    runner = runner_with_fake_lease(request, tmp_path / "group")
    result = runner.run()
    assert result["status"] == "completed", result
    assert runner.lease.released
    # Admission and descendant cleanup also occupy the card. On a busy host
    # those intervals can exceed the workers' overlap, so compare the actual
    # lease duration instead of assuming worker-time > total occupied time.
    assert result["peak_parallel_workers"] == 2
    workers = list(result["workers"].values())
    assert max(row["started_unix"] for row in workers) < min(row["finished_unix"] for row in workers)
    assert result["worker_wall_seconds"] == pytest.approx(sum(row["wall_seconds"] for row in workers))
    assert result["physical_device_seconds"] == pytest.approx(
        result["allocation_released_unix"] - result["allocation_acquired_unix"], abs=.03)
    for record in result["workers"].values():
        assert "device_seconds" not in record
        assert json.loads(Path(record["result_path"]).read_text())["visibility"] == "7"


def test_allowed_capacity_failure_preserves_other_workers_and_failed_result(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[
        {"sleep": .03, "code": 1, "status": "failed", "failure_class": "capacity_error"}, {"sleep": .2}])
    request["workers"][0]["allowed_failure_classes"] = ["capacity_error"]
    runner = runner_with_fake_lease(request, tmp_path / "group")
    result = runner.run()
    assert result["status"] == "completed", result
    assert result["workers"]["w0"]["status"] == "allowed_failure"
    assert result["workers"]["w0"]["return_code"] == 1
    assert result["workers"]["w1"]["status"] == "completed"
    assert result["remaining_descendants"] == [] and runner.lease.released


def test_exit_zero_internal_failure_is_exposed_for_scheduler_scientific_validation(tmp_path):
    request = manifest(tmp_path, configs=[{"status": "failed", "code": 0}, {}])
    result = shared.GroupRunner(request, tmp_path / "group").run()
    assert result["status"] == "completed"
    assert result["workers"]["w0"]["result_status"] == "failed"
    assert result["workers"]["w0"]["return_code"] == 0


def test_unexpected_failure_cleans_daemonized_descendant_before_unlock(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[
        {"orphan": True, "sleep": .08, "code": 3, "status": "failed"}, {"sleep": 10}])
    runner = runner_with_fake_lease(request, tmp_path / "group")
    result = runner.run()
    assert result["status"] == "failed"
    assert runner.lease.released and not runner.owned()
    orphan = int((tmp_path / "worker-0/orphan.pid").read_text())
    assert shared.proc_identity(orphan) is None
    assert result["remaining_descendants"] == []
    assert all(record["status"] in shared.TERMINAL for record in result["workers"].values())


def test_visibility_override_and_cpu_affinity_escape_are_rejected_before_launch(tmp_path):
    request = manifest(tmp_path)
    request["workers"][0]["env"]["ASCEND_RT_VISIBLE_DEVICES"] = "4"
    with pytest.raises(ValueError, match="visibility"):
        shared.validate_manifest(request)
    request["workers"][0]["env"].pop("ASCEND_RT_VISIBLE_DEVICES")
    request["workers"][0]["cpu_affinity"] = [max(os.sched_getaffinity(0)) + 100]
    with pytest.raises(ValueError, match="affinity"):
        shared.validate_manifest(request)


@pytest.mark.parametrize("policy", ["stop", "continue_running"])
def test_stale_pid_identity_does_not_authorize_an_unknown_device_occupant(tmp_path, monkeypatch, policy):
    request = manifest(tmp_path, backend="npu")
    request["external_occupant_policy"] = policy
    runner = runner_with_fake_lease(request, tmp_path / "group")
    runner.identities["99999:10"] = {"pid": 99999, "start_ticks": 10}
    monkeypatch.setattr(shared, "process_table", lambda **kwargs: {
        99999: {"pid": 99999, "ppid": 1, "state": "S", "start_ticks": 11, "rss_bytes": 1}})
    owned = runner.owned()
    assert 99999 not in owned
    device = {"health": "OK", "used_bytes": 0, "total_bytes": 2**30, "process_ids": [99999]}
    if policy == "stop":
        with pytest.raises(shared.GroupBlocked, match="unknown_device_occupant"):
            shared.admission_reason(request, request["workers"][0], [], owned, device)
    else:
        assert shared.admission_reason(request, request["workers"][0], [], owned, device) is None
        runner.record_device_observation(owned, {**device, "observed_unix": time.time()})
        assert not runner.state["all_seen_processes"]
        assert runner.state["external_contention_observed"]


def test_admission_reserves_future_hbm_and_cpu_work_before_allocations_appear(tmp_path):
    request = manifest(tmp_path, backend="npu")
    active, next_worker = request["workers"]
    active["reserved_hbm_bytes"] = 700 * 2**20
    next_worker["reserved_hbm_bytes"] = 300 * 2**20
    device = {"health": "OK", "used_bytes": 10 * 2**20, "baseline_used_bytes": 100 * 2**20,
              "total_bytes": 2**30, "process_ids": []}
    assert shared.admission_reason(request, next_worker, [active], {}, device) == "hbm_reservations_full"
    request["max_cpu_threads"] = 1
    assert shared.admission_reason(request, next_worker, [active], {}, device) == "cpu_thread_reservations_full"
    with pytest.raises(shared.GroupBlocked, match="group_hbm_budget_exceeded"):
        shared.admission_reason(request, None, [], {}, dict(device, used_bytes=2**30 + 1))
    with pytest.raises(shared.GroupBlocked, match="group_rss_budget_exceeded"):
        shared.admission_reason(request, None, [], {}, None, owner_rss=2**30 + 1)


def test_deadline_prevents_launch_and_new_group_attempt_cannot_overwrite_old_one(tmp_path):
    request = manifest(tmp_path)
    request["deadline_unix"] = time.time() - 1
    result = shared.GroupRunner(request, tmp_path / "group").run()
    assert result["status"] == "failed" and "deadline" in result["reason"]
    assert all(record["status"] == "not_started" for record in result["workers"].values())
    with pytest.raises(FileExistsError, match="existing group attempt"):
        shared.GroupRunner(request, tmp_path / "group")


@pytest.mark.parametrize("policy", ["stop", "continue_running"])
def test_shared_adapter_reuses_real_queue_double_sample_and_lock_lifetime(tmp_path, monkeypatch, policy):
    helper = shared.load_queue_helper()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(helper, "run_npu_smi", lambda *args: "0 0 0 7 Ascend910B\n")
    samples = []
    def snapshot(mapping):
        samples.append(time.monotonic())
        return {7: helper.DeviceStatus(0, 0, 7, "OK", 0, 0, 1024, ())}
    monkeypatch.setattr(helper, "snapshot", snapshot)
    monkeypatch.setattr(shared, "load_queue_helper", lambda: helper)
    request = manifest(tmp_path, backend="npu")
    request["external_occupant_policy"] = policy
    output = tmp_path / "group"
    output.mkdir()
    import threading
    lease = shared.QueueLease(request, output, threading.Event())
    lease.acquire()
    assert len(samples) == 2
    assert samples[1] > samples[0]
    lock_path = tmp_path / "runtime/ascend-device-locks/physical-7.lock"
    with lock_path.open("a+") as competitor:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease.release("completed")
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competitor, fcntl.LOCK_UN)
    assert not (tmp_path / "runtime/ascend-job-queue/jobs/shared-test.json").exists()


@pytest.mark.parametrize("policy", ["stop", "continue_running"])
def test_group_device_seconds_budget_stops_workers_and_releases_lease(tmp_path, policy):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": 5}])
    request["external_occupant_policy"] = policy
    request["max_device_seconds"] = .08
    runner = runner_with_fake_lease(request, tmp_path / "group")
    result = runner.run()
    assert result["status"] == "failed" and "group_device_seconds_budget" in result["reason"]
    assert runner.lease.released and result["remaining_descendants"] == []
    assert result["physical_device_seconds"] >= .08


def test_new_unknown_occupant_rejects_next_worker_and_never_signals_stranger(tmp_path, monkeypatch):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": 5}, {}])
    runner = runner_with_fake_lease(request, tmp_path / "group")
    original_factory = runner.lease_factory
    unknown = 2**30
    def factory(*args):
        lease = original_factory(*args)
        normal_snapshot = lease.snapshot
        def snapshot():
            value = normal_snapshot()
            if runner.processes:
                value["process_ids"] = [unknown]
            return value
        lease.snapshot = snapshot
        return lease
    runner.lease_factory = factory
    signalled = []
    original_kill = os.kill
    def kill(pid, sig):
        signalled.append(pid)
        return original_kill(pid, sig)
    monkeypatch.setattr(shared.os, "kill", kill)
    result = runner.run()
    assert result["status"] == "failed" and "unknown_device_occupant" in result["reason"]
    assert result["workers"]["w1"]["status"] == "not_started"
    assert unknown not in signalled
    assert runner.lease.released
    assert result["resources"]["device"]["process_ids"] == [unknown]
    assert result["external_contention_observed"]


def test_cpu_thread_budget_serializes_the_finite_group(tmp_path):
    request = manifest(tmp_path, configs=[{"sleep": .1}, {"sleep": .1}])
    request["max_cpu_threads"] = 1
    result = shared.GroupRunner(request, tmp_path / "group").run()
    assert result["status"] == "completed"
    assert result["peak_parallel_workers"] == 1
    assert result["workers"]["w1"]["started_unix"] >= result["workers"]["w0"]["finished_unix"]


def test_changed_manifest_stops_before_launching_additional_workers(tmp_path, monkeypatch):
    request = manifest(tmp_path, configs=[{"sleep": 5}, {}])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(request))
    runner = shared.GroupRunner(request, tmp_path / "group", manifest_path=path)
    normal_start = runner.start_worker
    def start(worker):
        normal_start(worker)
        path.write_text(json.dumps({**request, "unreviewed_change": True}))
    monkeypatch.setattr(runner, "start_worker", start)
    result = runner.run()
    assert result["status"] == "failed" and "group_manifest_changed" in result["reason"]
    assert result["workers"]["w1"]["status"] == "not_started"
    assert result["remaining_descendants"] == []


def device_observer(runner, observe):
    original_factory = runner.lease_factory
    def factory(*args):
        lease = original_factory(*args)
        normal_snapshot = lease.snapshot
        def snapshot():
            value = normal_snapshot()
            observe(value, lease)
            return value
        lease.snapshot = snapshot
        return lease
    runner.lease_factory = factory


def test_external_arrival_permits_running_and_new_workers_and_never_signals_stranger(tmp_path, monkeypatch):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": .4}, {"sleep": .4}])
    request["external_occupant_policy"] = "continue_running"
    runner = runner_with_fake_lease(request, tmp_path / "group")
    unknown = 2**30
    def observe(value, lease):
        if runner.state["workers"]["w0"]["pid"]:
            value.update(process_ids=[unknown], used_bytes=128 * 2**20)
    device_observer(runner, observe)
    signalled = []
    original_kill = os.kill
    def kill(pid, sig):
        signalled.append(pid)
        return original_kill(pid, sig)
    monkeypatch.setattr(shared.os, "kill", kill)
    result = runner.run()
    assert result["status"] == "completed", result
    assert result["peak_parallel_workers"] == 2
    assert all(row["status"] == "completed" for row in result["workers"].values())
    assert unknown not in signalled
    assert all(row["pid"] != unknown for row in result["all_seen_processes"].values())
    assert result["external_contention_observed"] and result["peak_external_processes"] == 1
    events = [json.loads(line) for line in Path(result["device_observation_log"]).read_text().splitlines()]
    assert events[0]["external_occupants"] == []
    assert events[-1]["external_occupants"] == [
        {"pid": unknown, "start_ticks": None, "uid": None, "identity_status": "unavailable"}]
    # The unrelated process is still on the device when our group releases.
    assert result["resources"]["device"]["process_ids"] == [unknown]
    assert result["remaining_descendants"] == [] and runner.lease.released


@pytest.mark.parametrize("arrival_sample", [1, 2])
def test_external_hbm_growth_waits_for_headroom_without_killing_running_worker(tmp_path, arrival_sample):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": .15}, {}])
    request["external_occupant_policy"] = "continue_running"
    runner = runner_with_fake_lease(request, tmp_path / "group")
    idle_pressure_samples = []
    active_samples = []
    def observe(value, lease):
        first = runner.state["workers"]["w0"]
        if first["pid"]:
            active_samples.append(lease.samples)
            if len(active_samples) >= arrival_sample:
                value.update(process_ids=[2**30], used_bytes=2**30 + 1)
        if first["status"] == "completed":
            idle_pressure_samples.append(lease.samples)
            if len(idle_pressure_samples) > 2:
                value.update(process_ids=[], used_bytes=0)
    device_observer(runner, observe)
    result = runner.run()
    assert result["status"] == "completed", result
    assert len(idle_pressure_samples) > 2  # No own workers remained, yet admission waited.
    assert result["peak_hbm_bytes"] > request["max_hbm_bytes"]
    assert result["peak_parallel_workers"] == 1
    assert result["workers"]["w1"]["started_unix"] >= result["workers"]["w0"]["finished_unix"]
    assert "waiting_reason" not in result["workers"]["w1"]
    events = [json.loads(line) for line in Path(result["device_observation_log"]).read_text().splitlines()]
    assert any(event["hbm_above_admission_limit"] for event in events)
    assert not events[-1]["external_occupants"]
    assert runner.lease.released


def test_external_allocation_before_first_worker_does_not_poison_baseline(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[{}])
    request["external_occupant_policy"] = "continue_running"
    runner = runner_with_fake_lease(request, tmp_path / "group")
    def observe(value, lease):
        if lease.samples < 4:
            value.update(process_ids=[2**30], used_bytes=2**30 + 1)
    device_observer(runner, observe)
    result = runner.run()
    assert result["status"] == "completed", result
    assert runner.device_baseline_bytes == 0 and runner.lease.samples >= 5
    assert runner.lease.released


def test_coexistence_reserves_future_allocations_above_unattributed_card_usage(tmp_path):
    request = manifest(tmp_path, backend="npu")
    request["external_occupant_policy"] = "continue_running"
    active, next_worker = request["workers"]
    active["reserved_hbm_bytes"] = 400 * 2**20
    next_worker["reserved_hbm_bytes"] = 200 * 2**20
    device = {"health": "OK", "used_bytes": 750 * 2**20, "total_bytes": 2**30, "process_ids": [2**30]}
    assert shared.admission_reason(request, None, [active], {}, device) is None
    assert shared.admission_reason(request, next_worker, [active], {}, device) == "hbm_reservations_full"
    with pytest.raises(shared.GroupBlocked, match="group_rss_budget_exceeded"):
        shared.admission_reason(request, None, [], {}, device, owner_rss=2**30 + 1)


@pytest.mark.parametrize("failure_class", ["capacity_error", "numerical_error", "device_error"])
def test_coexistence_does_not_hide_real_worker_failure(tmp_path, failure_class):
    request = manifest(tmp_path, backend="npu", configs=[
        {"sleep": .2, "code": 1, "status": "failed", "failure_class": failure_class}, {"sleep": 5}])
    request["external_occupant_policy"] = "continue_running"
    runner = runner_with_fake_lease(request, tmp_path / "group")
    device_observer(runner, lambda value, lease: value.update(process_ids=[2**30]))
    result = runner.run()
    assert result["status"] == "failed" and "worker_process_failed:w0:1" in result["reason"]
    assert result["workers"]["w0"]["failure_class"] == failure_class
    assert result["remaining_descendants"] == [] and runner.lease.released


def test_coexistence_still_stops_for_actual_device_health_error(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[{"sleep": 5}])
    request["external_occupant_policy"] = "continue_running"
    runner = runner_with_fake_lease(request, tmp_path / "group")
    def observe(value, lease):
        value["process_ids"] = [2**30]
        if runner.processes:
            value["health"] = "ERROR"
    device_observer(runner, observe)
    result = runner.run()
    assert result["status"] == "failed" and "allocated_device_unhealthy" in result["reason"]
    assert result["resources"]["device"]["health"] == "ERROR"
    assert result["remaining_descendants"] == [] and runner.lease.released


def test_coexistence_waits_only_until_original_deadline(tmp_path):
    request = manifest(tmp_path, backend="npu", configs=[{}])
    request.update(external_occupant_policy="continue_running", deadline_unix=time.time() + .3)
    runner = runner_with_fake_lease(request, tmp_path / "group")
    device_observer(runner, lambda value, lease: value.update(process_ids=[2**30], used_bytes=2**30 + 1))
    result = runner.run()
    assert result["status"] == "failed" and "absolute_deadline" in result["reason"]
    assert result["workers"]["w0"]["status"] == "not_started"
    assert result["workers"]["w0"]["waiting_reason"] == "hbm_reservations_full"
    assert runner.lease.released


def test_external_presence_does_not_bypass_observation_cadence(tmp_path):
    request = manifest(tmp_path, backend="npu")
    request.update(external_occupant_policy="continue_running", resource_interval_seconds=60.)
    runner = runner_with_fake_lease(request, tmp_path / "group")
    device_observer(runner, lambda value, lease: value.update(process_ids=[2**30]))
    runner.lease = runner.lease_factory(request, runner.output, runner.stop)
    runner.lease.acquire()
    try:
        runner.resources(force=True)
        runner.resources()
        runner.resources()
        assert runner.lease.samples == 1
    finally:
        runner.lease.release("completed")


def test_unknown_external_policy_is_rejected(tmp_path):
    request = manifest(tmp_path, backend="npu")
    request["external_occupant_policy"] = "ignore_all_errors"
    with pytest.raises(ValueError, match="external_occupant_policy"):
        shared.validate_manifest(request)


@pytest.mark.parametrize("late_change,expected", [
    ({}, None), ({"used_bytes": 5 * 2**30}, "hbm_reservations_full"),
    ({"process_ids": [2**30]}, "exclusive_hbm_requires_empty_card"),
    ({"total_bytes": 48 * 2**30}, "exclusive_hbm_requires_64_gib"),
    ({"health": "Alarm"}, "allocated_device_unhealthy"),
])
def test_exclusive_whole_card_admission_on_second_sample(tmp_path, late_change, expected):
    request = manifest(tmp_path, backend="npu", configs=[{}])
    request.update(exclusive_hbm=True, max_hbm_bytes=52*2**30, max_parallel=1,
                   release_on_idle_hbm_wait=True, external_occupant_policy="continue_running")
    worker = request["workers"][0]
    worker["reserved_hbm_bytes"] = int(47.4189453125 * 2**30)
    card = {"health":"OK", "used_bytes":2883*2**20, "total_bytes":64*2**30, "process_ids":[]}
    assert shared.admission_reason({**request, "exclusive_hbm":False, "max_hbm_bytes":48*2**30},
                                   worker, [], {}, card) == "hbm_reservations_full"
    with pytest.raises(ValueError, match="non-appendable"):
        shared.validate_manifest({**request, "max_parallel":2})
    runner = runner_with_fake_lease(request, tmp_path/"group")
    def sample(value, lease):
        value.update(card)
        if lease.samples >= 3:
            value.update(late_change)
    device_observer(runner, sample)
    result = runner.run()
    actual = result["workers"]["w0"]
    assert runner.lease.released and result["remaining_descendants"] == []
    if expected == "allocated_device_unhealthy":
        assert result["status"] == "failed" and expected in result["reason"]
    elif expected:
        assert result["capacity_deferral"]["reason"] == expected
        assert actual["status"] == "not_started" and actual["pid"] is None
    else:
        assert actual["status"] == "completed" and result["peak_parallel_workers"] == 1
        assert [s["projected_bytes"] for s in actual["hbm_admission_samples"]] == [int(50.234375*2**30)] * 2
