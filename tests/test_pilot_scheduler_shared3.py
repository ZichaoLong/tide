from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from pilotlab.io import sha256
from pilotlab.scheduler3 import Blocked
from pilotlab import scheduler_shared3 as shared
from pilotlab.scheduler_shared3 import SharedSupervisor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def bare(tmp_path, **config):
    supervisor = SharedSupervisor.__new__(SharedSupervisor)
    supervisor.output = tmp_path
    supervisor.config = {"target_npu_devices": 7, "npu_workers_per_device": 4,
        "max_cpu_workers": 8, "max_cpu_threads": 320, "max_inflight_checkpoint_bytes": 24 * 2**30, **config}
    supervisor.state = {"tasks": {}, "shared_groups": {}, "all_seen_processes": {}}
    supervisor.processes, supervisor.handles = {}, {}
    supervisor.stop_reason = None
    supervisor._save = lambda: None
    return supervisor


def tasks(supervisor, count, *, device="npu:0", reservation=2**28, purpose=None):
    rows = []
    for index in range(count):
        row = {"id": f"task-{index}", "model_id": f"model-{index}", "device": device,
               "checkpoint_reservation_bytes": reservation, "purpose": purpose}
        supervisor.state["tasks"][row["id"]] = {"status": "pending", "device": device}
        rows.append(row)
    return rows


def capture_groups(supervisor):
    captured = []
    def capture(rows, backend):
        captured.append((rows, backend))
        for row in rows:
            supervisor.state["tasks"][row["id"]]["status"] = "running"
    supervisor.start_group = capture
    return captured


def test_many_small_tasks_fill_devices_with_shared_workers(tmp_path):
    s = bare(tmp_path)
    rows = tasks(s, 28)
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 28)
    assert [len(group) for group, _ in captured] == [4] * 7
    assert {backend for _, backend in captured} == {"npu"}


def test_few_ready_tasks_spread_before_sharing(tmp_path):
    s = bare(tmp_path)
    rows = tasks(s, 4)
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 4)
    assert [len(group) for group, _ in captured] == [1] * 4


def test_capacity_unknown_tasks_remain_exclusive(tmp_path):
    s = bare(tmp_path)
    rows = tasks(s, 28, purpose="capacity_calibration")
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 28)
    assert len(captured) == 28
    assert all(len(group) == 1 for group, _ in captured)


def test_hard_device_cap_includes_capacity_probes_and_queued_owners(tmp_path):
    s = bare(tmp_path, max_npu_devices=8, target_npu_devices=8)
    rows = tasks(s, 28, purpose="capacity_calibration")
    s.state["tasks"]["queued"] = {"status": "running", "device": "npu:0", "shared_group": "queued-owner"}
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 28)
    assert len(captured) == 7
    assert all(len(group) == 1 for group, _ in captured)


def test_cpu_pool_fills_160_threads_and_npu_hosts_keep_separate_headroom(tmp_path):
    s = bare(tmp_path, max_cpu_workers=80, max_cpu_task_threads=160, max_npu_devices=8,
             target_npu_devices=8, npu_workers_per_device=16, max_cpu_threads=320,
             max_inflight_checkpoint_bytes=100 * 2**30)
    cpu = tasks(s, 50, device="cpu")
    npu = []
    for index in range(50, 100):
        task = {"id": f"task-{index}", "model_id": f"model-{index}", "device": "npu:0"}
        s.state["tasks"][task["id"]] = {"status": "pending", "device": "npu:0"}
        npu.append(task)
    captured = capture_groups(s)
    s.start_ready_tasks(cpu+npu, 100)
    assert sum(len(group) for group, backend in captured if backend == "cpu") == 40
    assert sum(len(group) for group, backend in captured if backend == "npu") == 40
    assert sum(backend == "npu" for _, backend in captured) <= 8


def test_storage_reservation_limits_admission_without_shortening_tasks(tmp_path):
    s = bare(tmp_path, max_inflight_checkpoint_bytes=2 * 2**30)
    rows = tasks(s, 28, reservation=2**30)
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 28)
    assert sum(len(group) for group, _ in captured) == 2
    assert sum(r["status"] == "pending" for r in s.state["tasks"].values()) == 26


def test_cpu_pool_has_separate_concurrency_bound(tmp_path):
    s = bare(tmp_path, max_cpu_workers=3)
    rows = tasks(s, 8, device="cpu")
    s.state["tasks"][rows[0]["id"]]["status"] = "running"
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 8)
    assert sum(len(group) for group, _ in captured) == 2


def add_group(s, name, *, seconds=10., physical=2, release=None):
    path = s.output / name
    group = {"job_id": name, "manifest_sha256": "verified-manifest", "backend": "npu", "status": "running",
        "allocation_acquired_unix": 120.-seconds, "allocation_released_unix": release,
        "physical_device_seconds": seconds, "physical_id": physical,
        "workers": {f"w-{i}": {"status": "running"} for i in range(4)}, "all_seen_processes": {}}
    write(path / "group.json", group)
    s.state["shared_groups"][name] = {"path": str(path), "job_id": name,
        "manifest_sha256": "verified-manifest", "preparation": True, "backend": "npu", "started_unix": 90.}
    return group


def test_card_time_counts_each_lease_once_not_each_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.time, "time", lambda: 120.)
    s = bare(tmp_path)
    add_group(s, "g0", seconds=10, physical=2)
    add_group(s, "g1", seconds=20, physical=4)
    assert s.device_seconds() == 30
    assert s.preparation_device_seconds() == 30
    assert s.actual_npu_concurrency([]) == 2
    assert s.state["actual_npu_worker_concurrency"] == 8


def test_released_group_no_longer_counts_as_live_card(tmp_path):
    s = bare(tmp_path)
    add_group(s, "g0", seconds=10, release=120.)
    assert s.device_seconds() == 10
    assert s.actual_npu_concurrency([]) == 0


def test_changed_group_manifest_identity_blocks_without_interrupting_cleanup_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(shared.time, "time", lambda: 120.)
    s = bare(tmp_path)
    group = add_group(s, "g0")
    group["manifest_sha256"] = "changed"
    write(tmp_path / "g0/group.json", group)
    assert s.device_seconds() == 30
    assert s.stop_reason == "shared_group_observation_failed"
    assert "identity_mismatch" in s.state["shared_groups"]["g0"]["observation_error"]


@pytest.mark.parametrize("damage", [None, "result_hash", "task_hash", "no_release", "descendants"])
def test_worker_completion_needs_task_result_and_group_release(tmp_path, damage):
    s = bare(tmp_path)
    task = {"id": "t0", "model_id": "m0", "phase": "a1", "device": "npu:0",
            "target_tokens": 100, "source_identity": {"source": "frozen"}}
    directory = tmp_path / "tasks/t0/attempt-0"
    task_path = write(directory.parent / "task.json", task)
    result = {"status": "completed", "task_sha256": sha256(task_path),
              "source_identity": task["source_identity"], "training": {"tokens": 100}}
    if damage == "task_hash":
        result["task_sha256"] = "changed"
    result_path = write(directory / "result.json", result)
    group = add_group(s, "g0", release=120.)
    group.update(status="completed", queue_released_unix=120., remaining_descendants=[], workers={"t0": {
        "status": "completed", "return_code": 0, "result_sha256": sha256(result_path), "output": str(directory)}})
    write(tmp_path / "g0/queue.json", {"job_id": "g0", "status": "completed", "return_code": 0})
    if damage == "result_hash":
        group["workers"]["t0"]["result_sha256"] = "changed"
    elif damage == "no_release":
        group["allocation_released_unix"] = None
    elif damage == "descendants":
        group["remaining_descendants"] = [{"pid": 123}]
    write(tmp_path / "g0/group.json", group)
    record = {"status": "running", "shared_group": "g0", "task_path": str(task_path),
              "attempts": [{"path": str(directory)}]}
    assert s._poll(task, record)
    assert record["status"] == ("completed" if damage is None else "failed")


def finished_worker(tmp_path, *, failure_class=None):
    s = bare(tmp_path, max_workers=8)
    task = {"id": "t0", "model_id": "m0", "phase": "a1", "device": "npu:0",
            "target_tokens": 100, "source_identity": {"source": "frozen"}}
    directory = tmp_path / "tasks/t0/attempt-0"
    task_path = write(directory.parent / "task.json", task)
    checkpoint = directory / "checkpoint.pt"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"same-source checkpoint witness")
    result = {"status": "failed" if failure_class else "completed", "failure_class": failure_class,
              "task_sha256": sha256(task_path), "source_identity": task["source_identity"],
              "training": {"tokens": 100}, "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)}}
    result_path = write(directory / "result.json", result)
    group = add_group(s, "g0", release=120.)
    failed_group = failure_class == "infrastructure_transient"
    group.update(status="failed" if failed_group else "completed", queue_released_unix=120., remaining_descendants=[],
                 reason="worker_process_failed:t0:1" if failed_group else None,
                 workers={"t0": {"status": "failed" if failed_group else "allowed_failure" if failure_class else "completed",
                     "return_code": 1 if failure_class else 0, "result_sha256": sha256(result_path), "output": str(directory)}})
    write(tmp_path / "g0/group.json", group)
    write(tmp_path / "g0/queue.json", {"job_id": "g0", "status": group["status"], "return_code": int(failed_group)})
    s.state["shared_groups"]["g0"]["owner_return_code"] = int(failed_group)
    record = {"status": "running", "stage": "a1", "model_id": "m0", "device": "npu:0", "shared_group": "g0",
              "task_path": str(task_path), "attempts": [{"path": str(directory), "shared_group": "g0", "started_unix": 90.}]}
    s.state["tasks"]["t0"] = record
    s.state["peak_actual_npu_concurrency"] = 0
    s.resource_reason = lambda: None
    s.owned_members = lambda: {}
    return s, task, record, result, group


def test_completed_shared_restart_reuses_group_evidence_without_an_exclusive_queue_file(tmp_path):
    s, task, record, _, _ = finished_worker(tmp_path)
    assert s._poll(task, record) and record["status"] == "completed"
    assert not (Path(record["attempts"][0]["path"]) / "queue.json").exists()
    s.start_group = lambda *_: pytest.fail("a completed task must not relaunch")
    answer = s.run("a1", [task])
    assert answer[task["id"]]["status"] == "completed"


@pytest.mark.parametrize("queue_intact", [True, False])
def test_owner_exit_reopens_running_snapshot_before_settling_worker(tmp_path, queue_intact):
    supervisor, task, record, _, terminal = finished_worker(tmp_path)
    stale = deepcopy(terminal)
    stale.update(status="running", allocation_released_unix=None, queue_released_unix=None)
    stale["workers"][task["id"]].update(status="running", return_code=None, result_sha256=None)
    supervisor._group_observation_cache = {"g0": stale}
    if not queue_intact:
        (tmp_path / "g0/queue.json").unlink()

    assert supervisor._poll(task, record)
    assert record["attempts"][-1]["return_code"] == 0
    assert record["attempts"][-1]["release_unix"] == terminal["allocation_released_unix"]
    if queue_intact:
        assert record["status"] == "completed"
        supervisor.verify_completed(task, record)
    else:
        assert record["status"] == "failed" and record["reason"] == "shared_group_lease_release_missing"


@pytest.mark.parametrize("damage", ["owner_exit", "group_failed", "queue_failed", "queue_missing", "result_changed", "wrong_output"])
def test_completed_shared_restart_revalidates_owner_queue_and_result(tmp_path, damage):
    s, task, record, result, group = finished_worker(tmp_path)
    assert s._poll(task, record)
    if damage == "owner_exit":
        s.state["shared_groups"]["g0"]["owner_return_code"] = 1
    elif damage == "group_failed":
        group["status"] = "failed"
        write(tmp_path / "g0/group.json", group)
    elif damage == "queue_failed":
        write(tmp_path / "g0/queue.json", {"job_id": "g0", "status": "failed", "return_code": 1})
    elif damage == "queue_missing":
        (tmp_path / "g0/queue.json").unlink()
    elif damage == "result_changed":
        result["training"]["tokens"] = 99
        write(Path(record["result_path"]), result)
    else:
        group["workers"]["t0"]["output"] = str(tmp_path / "another-output")
        write(tmp_path / "g0/group.json", group)
    with pytest.raises(Blocked):
        s.verify_completed(task, record)


@pytest.mark.parametrize("damage", [None, "release_error", "descendants", "queue_corrupt"])
def test_only_clean_capacity_failures_receive_the_declared_exemption(tmp_path, damage):
    s, task, record, result, group = finished_worker(tmp_path, failure_class="capacity_error")
    task.update(phase="preflight", purpose="capacity_calibration")
    write(Path(record["task_path"]), task)
    result["task_sha256"] = sha256(record["task_path"])
    result_path = Path(record["attempts"][0]["path"]) / "result.json"
    write(result_path, result)
    group["workers"]["t0"]["result_sha256"] = sha256(result_path)
    if damage == "release_error":
        group["queue_release_error"] = "could not remove queue job"
    elif damage == "descendants":
        group["remaining_descendants"] = [{"pid": 999}]
    elif damage == "queue_corrupt":
        (tmp_path / "g0/queue.json").write_text("{broken")
    write(tmp_path / "g0/group.json", group)
    assert s._poll(task, record)
    if damage is None:
        assert not record.get("observation_error")
        assert s.run("a1", [task], allowed_failures=("capacity_error",))["t0"]["failure_class"] == "capacity_error"
    else:
        assert record["observation_error"]
        with pytest.raises(Blocked):
            s.run("a1", [task], allowed_failures=("capacity_error",))


@pytest.mark.parametrize("damage", [None, "checkpoint", "wrong_task", "queue_error", "second_attempt"])
def test_shared_transient_retry_is_once_same_identity_and_only_after_verified_release(tmp_path, damage):
    s, task, record, result, group = finished_worker(tmp_path, failure_class="infrastructure_transient")
    task_hash = sha256(record["task_path"])
    if damage == "checkpoint":
        Path(result["checkpoint"]["path"]).write_bytes(b"corrupted")
    elif damage == "wrong_task":
        result["task_sha256"] = "f"*64
        result_path = Path(record["attempts"][0]["path"]) / "result.json"
        write(result_path, result)
        group["workers"]["t0"]["result_sha256"] = sha256(result_path)
    elif damage == "queue_error":
        group["queue_release_error"] = "release unverified"
    elif damage == "second_attempt":
        record["attempts"].append(deepcopy(record["attempts"][0]))
    write(tmp_path / "g0/group.json", group)
    assert s._poll(task, record)
    assert sha256(record["task_path"]) == task_hash
    if damage is None:
        assert record["status"] == "pending"
        assert record["retry_checkpoint"] == result["checkpoint"]["path"]
        assert record["retry_checkpoint_sha256"] == result["checkpoint"]["sha256"]
    else:
        assert record["status"] == "failed" and "retry_checkpoint" not in record


@pytest.mark.parametrize("damage", ["missing", "corrupt", "identity", "nonobject"])
def test_uncached_bad_group_and_queue_charge_once_from_launch_until_verified_cleanup(tmp_path, monkeypatch, damage):
    s = bare(tmp_path)
    group = add_group(s, "g0")
    path = tmp_path / "g0/group.json"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("{")
    elif damage == "identity":
        group["job_id"] = "another-job"
        write(path, group)
    else:
        path.write_text("[]")
    (tmp_path / "g0/queue.json").write_text("{broken")
    s.state["tasks"]["ignored-worker-seconds"] = {"shared_group": "g0", "device_seconds": 999.}
    monkeypatch.setattr(shared.time, "time", lambda: 120.)
    assert s.device_seconds() == 30
    assert s.preparation_device_seconds() == 30
    s.state["shared_groups"]["g0"]["cleanup_verified_unix"] = 125.
    monkeypatch.setattr(shared.time, "time", lambda: 1000.)
    assert s.device_seconds() == 35
    assert s.state["shared_groups"]["g0"]["accounting_estimated"] is True


def test_group_accounting_never_decreases_after_a_stale_observation(tmp_path, monkeypatch):
    s = bare(tmp_path)
    group = add_group(s, "g0", seconds=10., release=120.)
    assert s.device_seconds() == 10
    group["physical_device_seconds"] = 5.
    write(tmp_path / "g0/group.json", group)
    assert s.device_seconds() == 10


def test_cpu_pool_precedes_a_full_npu_backlog(tmp_path):
    s = bare(tmp_path, max_cpu_workers=2)
    rows = tasks(s, 20)
    for index in range(4):
        task = {"id": f"cpu-{index}", "device": "cpu", "cpu_threads": 4}
        s.state["tasks"][task["id"]] = {"status": "pending", "device": "cpu"}
        rows.append(task)
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 6)
    assert [backend for _, backend in captured[:2]] == ["cpu", "cpu"]
    assert sum(len(group) for group, backend in captured if backend == "npu") == 4


@pytest.mark.parametrize("resource,limit,expected", [("max_rss_bytes", 48*2**30, 3), ("max_cpu_threads", 12, 3)])
def test_unknown_capacity_tasks_respect_total_rss_and_thread_reservations(tmp_path, resource, limit, expected):
    s = bare(tmp_path, **{resource: limit})
    rows = tasks(s, 20, purpose="capacity_calibration")
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 20)
    assert sum(len(group) for group, _ in captured) == expected


def test_existing_queued_reservations_reduce_admission_and_sixteen_way_cap_is_preserved(tmp_path):
    s = bare(tmp_path, target_npu_devices=1, npu_workers_per_device=16)
    rows = tasks(s, 16)
    for row in rows:
        row["reserved_hbm_bytes"] = 2**30
    captured = capture_groups(s)
    s.start_ready_tasks(rows, 16)
    assert [len(group) for group, _ in captured] == [16]
    second = bare(tmp_path / "other", max_rss_bytes=32*2**30, max_cpu_threads=8)
    pending = tasks(second, 8)
    second.state["tasks"]["already-queued"] = {"status": "running", "device": "npu:0",
        "reserved_rss_bytes": 16*2**30, "cpu_threads": 4, "checkpoint_reservation_bytes": 2**28}
    captured = capture_groups(second)
    second.start_ready_tasks(pending, 8)
    assert sum(len(group) for group, _ in captured) == 1


def launch_fixture(tmp_path, *, count=2):
    budget = write(tmp_path / "budget.json", {"max_device_seconds": 1000., "device_seconds": 0., "segments": {}})
    s = bare(tmp_path, python=sys.executable, source=str(tmp_path), foundation=str(tmp_path), data=str(tmp_path),
             budget_ledger=str(budget), run_id="test-sharing", deadline_unix=shared.time.time()+600)
    rows = tasks(s, count, device="cpu")
    for row in rows:
        row.update(phase="a1", target_tokens=100, source_identity={"source": "frozen"}, cpu_threads=1,
                   evaluation_budget_key="model-group", eval_tokens=1, evaluation_token_limit=1000)
        s.state["tasks"][row["id"]].update(attempts=[], stage="a1", model_id=row["model_id"])
    return s, rows


def test_group_reservation_is_atomic_when_a_later_member_exceeds_budget(tmp_path):
    s, rows = launch_fixture(tmp_path)
    for row in rows:
        row["evaluation_token_limit"] = 34
    original = deepcopy(s.state)
    with pytest.raises(Blocked, match="evaluation_token_limit"):
        s.start_group(rows, "cpu")
    assert s.state == original
    assert not (tmp_path / "tasks").exists()


def test_failed_popen_refunds_unexecuted_reservations_and_leaves_no_running_attempt(tmp_path, monkeypatch):
    s, rows = launch_fixture(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected spawn failure")
    monkeypatch.setattr(shared.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failure"):
        s.start_group(rows, "cpu")
    assert not s.processes and not s.handles
    for record in s.state["tasks"].values():
        assert record["status"] == "failed" and record["attempts"][-1]["status"] == "not_started"
        assert not record.get("evaluation_reserved_tokens")
    assert s.state["shared_groups"]["group-000000"]["launch_aborted_before_spawn"] is True


@pytest.mark.parametrize("policy", [None, "continue_running"])
def test_external_policy_is_frozen_into_each_group_manifest(tmp_path, monkeypatch, policy):
    s, rows = launch_fixture(tmp_path)
    if policy is not None:
        s.config["external_occupant_policy"] = policy
    for row in rows:
        row["device"] = "npu:0"
    def fail(*args, **kwargs):
        raise OSError("test stops before any real launch")
    monkeypatch.setattr(shared.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="before any real launch"):
        s.start_group(rows, "npu")
    value = json.loads((tmp_path / "groups/group-000000/manifest.json").read_text())
    assert value["external_occupant_policy"] == (policy or "stop")


def test_process_and_every_member_are_registered_before_a_failed_identity_read(tmp_path, monkeypatch):
    s, rows = launch_fixture(tmp_path)
    class Process:
        pid = 99999999
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode
    process = Process()
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(shared, "proc_identity", lambda *a, **kw: None)
    with pytest.raises(Blocked, match="before_identity_capture"):
        s.start_group(rows, "cpu")
    assert s.processes["group-000000"] is process
    assert all(record["status"] == "running" and record["provisional_pid"] == process.pid
               for record in s.state["tasks"].values())
    s.owned_members = lambda: {}
    monkeypatch.setattr(os, "killpg", lambda *a: setattr(process, "returncode", -15))
    s.stop_all()
    assert not s.processes and not s.handles
    assert all(record["status"] == "failed" for record in s.state["tasks"].values())


def test_broken_group_and_queue_cannot_interrupt_cleanup_or_erase_card_time(tmp_path, monkeypatch):
    s, _, record, _, _ = finished_worker(tmp_path)
    (tmp_path / "g0/group.json").write_text("{broken")
    (tmp_path / "g0/queue.json").write_text("{broken")
    class Process:
        pid = 99999999
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            return self.returncode
    process = Process()
    s.processes["g0"] = process
    s._save = lambda: s.state.update(device_seconds=s.device_seconds())
    monkeypatch.setattr(shared.time, "time", lambda: 120.)
    monkeypatch.setattr(os, "killpg", lambda *a: setattr(process, "returncode", -15))
    s.stop_all()
    assert process.returncode == -15 and record["status"] == "failed"
    assert not s.processes and s.state["remaining_descendants"] == []
    assert s.state["device_seconds"] == 30
