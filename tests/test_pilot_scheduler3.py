"""CPU-only fault witnesses for round-three supervision and resource accounting.

No worker, queue helper, device, or training is started. Temporary artifacts and
fake process identities exercise the same supervisor paths used after restart.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from pilotlab import scheduler3 as s


TARGET = 12 * 1024**2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def task(**changes):
    value = {"id": "formal-one", "model_id": "model-one", "device": "cpu",
             "target_tokens": TARGET,
             "source_identity": {"source": "frozen-source", "foundation": "frozen-foundation",
                                 "data": "frozen-data"}}
    value.update(changes)
    return value


def result(request, **changes):
    value = {"status": "completed", "training": {"tokens": request["target_tokens"]},
             "source_identity": deepcopy(request.get("source_identity"))}
    value.update(changes)
    return value


@pytest.fixture
def supervisor_factory(tmp_path, monkeypatch):
    # The real controller enables subreaping; unit tests must not change the
    # process's ownership policy or inspect quota / launch a queue request.
    monkeypatch.setattr(s, "child_subreaper", lambda *args: False)
    supervisors = []

    def make(*, config=None, output=None):
        if config is None:
            ledger = tmp_path / "budget.json"
            if not ledger.exists():
                write_json(ledger, {"deadline_unix": time.time() + 3600,
                                    "max_device_seconds": 7200 * 3600,
                                    "device_seconds": 0, "segments": {}})
            source = tmp_path / "source"
            source.mkdir(exist_ok=True)
            config = {"budget_ledger": str(ledger), "deadline_unix": time.time() + 3600,
                      "python": sys.executable, "source": str(source),
                      "foundation": str(tmp_path / "foundation"), "data": str(tmp_path / "data"),
                      "run_id": "unit-r3", "max_workers": 1, "poll_seconds": 0,
                      "storage_reserve_bytes": 100, "max_rss_bytes": 1000}
        supervisor = s.Supervisor(config, output or tmp_path / "round")
        supervisors.append(supervisor)
        return supervisor

    yield make
    for supervisor in supervisors:
        if not supervisor.lock.closed:
            supervisor.close()


@pytest.mark.parametrize("tokens", [0, TARGET - 1, TARGET + 1, None])
def test_exit_zero_does_not_complete_wrong_token_exposure(tokens):
    request = task()
    status, reason = s.evaluate_outcome(request, result(request, training={"tokens": tokens}), 0)
    assert status == "failed"
    assert reason == "declared_token_target_not_reached"


@pytest.mark.parametrize("worker_result,code", [(None, 0), ({}, 0), ([], 0),
                                                 ({"status": "failed", "error": "bad gradient"}, 0)])
def test_missing_or_failed_worker_result_is_not_outer_success(worker_result, code):
    assert s.evaluate_outcome(task(), worker_result, code)[0] == "failed"


def test_completed_result_still_requires_successful_process_and_queue():
    request = task()
    answer = result(request)
    assert s.evaluate_outcome(request, answer, 0) == ("completed", None)
    assert s.evaluate_outcome(request, answer, 1)[0] == "failed"
    for queue in ({}, {"status": "failed", "return_code": 0},
                  {"status": "completed", "return_code": 1}):
        assert s.evaluate_outcome(request, answer, 0, queue)[0] == "failed"


@pytest.mark.parametrize("missing_both", [False, True])
def test_source_identity_must_be_present_and_match(missing_both):
    request = task()
    answer = result(request)
    if missing_both:
        request.pop("source_identity")
        answer.pop("source_identity")
    else:
        answer["source_identity"] = {"source": "different"}
    assert s.evaluate_outcome(request, answer, 0)[0] == "failed"


@pytest.mark.parametrize("status,expected", [("stopped_budget", "stopped_budget"),
                                              ("incomplete", "stopped_budget"),
                                              ("interrupted", "interrupted")])
def test_explicit_budget_or_signal_status_survives_nonzero_worker_exit(status, expected):
    request = task()
    answer = result(request, status=status, stop_reason="global_deadline")
    assert s.evaluate_outcome(request, answer, 3)[0] == expected


def test_only_predeclared_optimizer_failure_is_recipe_rejection():
    request = task()
    answer = result(request, status="recipe_rejected", failure_class="declared_optimizer_instability")
    assert s.evaluate_outcome(request, answer, 0)[0] == "recipe_rejected"
    answer["failure_class"] = "unknown_numerical_failure"
    assert s.evaluate_outcome(request, answer, 0)[0] == "failed"


def test_recipe_rejection_with_wrong_source_identity_blocks_instead_of_continuing():
    request = task()
    answer = result(request, status="recipe_rejected", failure_class="declared_optimizer_instability",
                    source_identity={"source": "wrong-tree"})
    assert s.evaluate_outcome(request, answer, 0)[0] == "failed"


def test_two_controllers_cannot_own_the_same_round(supervisor_factory):
    first = supervisor_factory()
    with pytest.raises(BlockingIOError):
        supervisor_factory(config=first.config, output=first.output)
    first.close()
    recovered = supervisor_factory(config=first.config, output=first.output)
    assert recovered.state["controller_identity"]["pid"] == os.getpid()


def saved_record(supervisor, request, *, status="completed", is_npu=False):
    directory = supervisor.output / "tasks" / request["id"] / "attempt-0"
    directory.mkdir(parents=True)
    task_path = write_json(directory.parent / "task.json", request)
    result_path = write_json(directory / "result.json", result(request, task_sha256=s.sha256(task_path)))
    record = {"status": status, "stage": "A1", "model_id": request["model_id"],
              "task_path": str(task_path), "result_path": str(result_path),
              "result_sha256": s.sha256(result_path), "device_seconds": 0,
              "process": None,
              "attempts": [{"path": str(directory), "status": status, "process": None,
                            "started_unix": time.time() - 2, "device_seconds": 0,
                            "return_code": 0 if status == "completed" else None}]}
    if is_npu:
        write_json(directory / "queue.json", {"status": "completed", "return_code": 0})
    supervisor.state["tasks"][request["id"]] = record
    supervisor._save()
    return record, directory


def forbid_launch(*args, **kwargs):
    pytest.fail("recovery must not launch an already completed or existing task")


@pytest.mark.parametrize("dynamic", [False, True])
def test_finished_stage_waits_for_resource_owners_while_still_supervising(supervisor_factory, monkeypatch, dynamic):
    owner = supervisor_factory()
    calls = []
    drains = iter((False, False, True))
    monkeypatch.setattr(owner, "supervision_tick", lambda: calls.append("observe"))
    monkeypatch.setattr(owner, "resource_reason", lambda: calls.append("budget"))
    monkeypatch.setattr(owner, "stage_drained", lambda: next(drains))
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    monkeypatch.setattr(owner, "start_ready_tasks", lambda *args: None)
    answer = owner.run_dynamic("a", lambda _: {"requests": [], "done": True}) if dynamic else owner.run("a", [])
    assert answer == {}
    assert calls == ["observe", "budget"] * 3


def test_owner_failure_after_last_task_prevents_stage_completion(supervisor_factory, monkeypatch):
    owner = supervisor_factory()
    ticks = []
    def observe():
        ticks.append(1)
        if len(ticks) == 2:
            raise s.Blocked("dynamic_owner_late_failure")
    monkeypatch.setattr(owner, "supervision_tick", observe)
    monkeypatch.setattr(owner, "resource_reason", lambda: None)
    monkeypatch.setattr(owner, "stage_drained", lambda: False)
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    with pytest.raises(s.Blocked, match="dynamic_owner_late_failure"):
        owner.run_dynamic("a", lambda _: {"requests": [], "done": True})
    assert owner.state["stage_b_gate"] == "closed"
    assert owner.state["status"] == "a_blocked_needs_repair"


def test_dynamic_dependencies_advance_before_unrelated_slow_task(supervisor_factory, monkeypatch):
    owner = supervisor_factory()
    owner.config["max_workers"] = 2
    monkeypatch.setattr(owner, "resource_reason", lambda: None)
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    started, polls = [], {}
    fast = task(id="fast", phase="a1")
    slow = task(id="slow", phase="a1")
    child = task(id="fast-a2", phase="a2")
    def start(request, record):
        started.append((request["id"], owner.state["tasks"].get("slow", {}).get("status")))
        record.update(status="running", task_path=str(owner.output/"tasks"/request["id"]/"task.json"),
            attempts=[{"path": str(owner.output/"tasks"/request["id"]/"attempt-0"), "return_code": 0}])
    def poll(request, record):
        polls[request["id"]] = polls.get(request["id"], 0)+1
        if request["id"] == "slow" and polls[request["id"]] < 4:
            return False
        path = write_json(Path(record["attempts"][0]["path"])/"result.json",
                          result(request, task_sha256=s.sha256(record["task_path"])))
        record.update(status="completed", result_path=str(path), result_sha256=s.sha256(path))
        return True
    monkeypatch.setattr(owner, "_start", start)
    monkeypatch.setattr(owner, "_poll", poll)
    def producer(verified):
        requests = [{"stage": "a1", "tasks": [fast, slow]}]
        if "fast" in verified:
            requests.append({"stage": "a2", "tasks": [child]})
        return {"requests": requests, "done": len(verified) == 3}
    completed = owner.run_dynamic("stage-a", producer)
    assert len(completed) == 3
    assert ("fast-a2", "running") in started
    assert owner.state["tasks"]["fast-a2"]["stage"] == "a2"


def test_dynamic_imported_failure_closes_before_launching_other_work(supervisor_factory, monkeypatch):
    owner = supervisor_factory()
    failed = task(id="failed")
    record, _ = saved_record(owner, failed, status="failed")
    record["stage"] = "a1"
    monkeypatch.setattr(owner, "_start", forbid_launch)
    monkeypatch.setattr(owner, "resource_reason", lambda: None)
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    with pytest.raises(s.Blocked):
        owner.run_dynamic("stage-a", lambda verified: {"requests": [
            {"stage": "a1", "tasks": [failed, task(id="unrelated")]}], "done": False})
    assert owner.state["stage_b_gate"] == "closed"


def test_dynamic_scientific_recipe_rejection_is_returned_without_busy_loop(supervisor_factory, monkeypatch):
    owner = supervisor_factory()
    request = task(id="rejected")
    record, directory = saved_record(owner, request)
    record.update(status="recipe_rejected", stage="a1")
    write_json(directory/"result.json", result(request, status="recipe_rejected",
        failure_class="declared_optimizer_instability", task_sha256=s.sha256(record["task_path"])))
    record["result_sha256"] = s.sha256(directory/"result.json")
    monkeypatch.setattr(owner, "resource_reason", lambda: None)
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    monkeypatch.setattr(owner, "_start", forbid_launch)
    value = owner.run_dynamic("stage-a", lambda verified: {"requests": [
        {"stage": "a1", "tasks": [request]}], "done": "rejected" in verified})
    assert value["rejected"]["status"] == "recipe_rejected"


def test_imported_recipe_rejection_does_not_need_or_launch_a_new_attempt(supervisor_factory, monkeypatch):
    from pilotlab import resume3
    owner = supervisor_factory()
    request = task(id="rejected")
    record, directory = saved_record(owner, request)
    answer = result(request, status="recipe_rejected", failure_class="declared_optimizer_instability",
                    task_sha256=s.sha256(record["task_path"]))
    write_json(directory/"result.json", answer)
    record.update(status="recipe_rejected", stage="a1", attempts=[], imported_from="bound-old-state",
                  result_sha256=s.sha256(directory/"result.json"))
    reference = {"path": record["result_path"], "sha256": record["result_sha256"]}
    monkeypatch.setattr(resume3, "imported_entry", lambda *args: (
        {"disposition": "recipe_rejected", "result": reference, "worker_return_code": 1}, request, answer))
    monkeypatch.setattr(owner, "resource_reason", lambda: None)
    monkeypatch.setattr(owner, "owned_members", lambda: {})
    monkeypatch.setattr(owner, "_start", forbid_launch)
    value = owner.run_dynamic("stage-a", lambda verified: {"requests": [
        {"stage": "a1", "tasks": [request]}], "done": "rejected" in verified})
    assert value["rejected"] == answer
    assert record["attempts"] == []


def test_restart_reuses_valid_completed_result_without_launch(supervisor_factory, monkeypatch):
    first = supervisor_factory()
    request = task()
    saved_record(first, request)
    first.close()
    recovered = supervisor_factory(config=first.config, output=first.output)
    monkeypatch.setattr(recovered, "resource_reason", lambda: None)
    monkeypatch.setattr(recovered, "_start", forbid_launch)
    assert recovered.run("A1", [request])[request["id"]]["status"] == "completed"


@pytest.mark.parametrize("damage", ["missing", "bad_json", "bad_hash", "wrong_tokens", "wrong_identity"])
def test_restart_does_not_trust_a_stale_completed_label(supervisor_factory, monkeypatch, damage):
    first = supervisor_factory()
    request = task()
    record, directory = saved_record(first, request)
    path = directory / "result.json"
    if damage == "missing":
        path.unlink()
    elif damage == "bad_json":
        path.write_text('{"status":')
    elif damage == "bad_hash":
        write_json(path, {**result(request), "extra": "changed after completion"})
    else:
        answer = result(request)
        if damage == "wrong_tokens":
            answer["training"]["tokens"] -= 1
        else:
            answer["source_identity"] = {"source": "wrong-frozen-tree"}
        write_json(path, answer)
        # Even a digest-consistent stale terminal record must satisfy today's
        # same immutable request, not only agree with its own stored hash.
        record["result_sha256"] = s.sha256(path)
        first._save()
    first.close()
    recovered = supervisor_factory(config=first.config, output=first.output)
    monkeypatch.setattr(recovered, "resource_reason", lambda: None)
    monkeypatch.setattr(recovered, "_start", forbid_launch)
    with pytest.raises((s.Blocked, ValueError, FileNotFoundError, RuntimeError)):
        recovered.run("A1", [request])


def test_restart_adopts_existing_finished_worker_without_duplicate_launch(supervisor_factory, monkeypatch):
    first = supervisor_factory()
    request = task()
    saved_record(first, request, status="running")
    first.close()
    recovered = supervisor_factory(config=first.config, output=first.output)
    monkeypatch.setattr(recovered, "resource_reason", lambda: None)
    monkeypatch.setattr(recovered, "_start", forbid_launch)
    assert recovered.run("A1", [request])[request["id"]]["status"] == "completed"
    assert len(recovered.state["tasks"][request["id"]]["attempts"]) == 1


def queue_event(status, when):
    return json.dumps({"elapsed_seconds": 0, "event": "ascend_queue", "status": status,
                       "timestamp": datetime.fromtimestamp(when, timezone.utc).isoformat()}, sort_keys=True)


def test_queue_shorter_than_poll_interval_is_fully_charged(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("worker output\n" + queue_event("running", 100) + "\n"
                   + '{"event": "ascend_queue", broken\n'
                   + queue_event("completed", 100.75) + "\n")
    assert s.queue_interval(log, now=1000) == (100, 100.75, .75)
    log.write_text(queue_event("running", 100) + "\n")
    assert s.queue_interval(log, now=102) == (100, None, 2)


@pytest.mark.parametrize("damage", ["missing", "truncated", "bad_timestamp", "backwards", "missing_release"])
def test_completed_queue_cannot_hide_missing_or_invalid_accounting(supervisor_factory, monkeypatch, damage):
    supervisor = supervisor_factory()
    request = task(device="npu:0")
    record, directory = saved_record(supervisor, request, status="running", is_npu=True)
    if damage == "truncated":
        (directory / "stdout.log").write_text('{"event": "ascend_queue", broken\n')
    elif damage == "bad_timestamp":
        (directory / "stdout.log").write_text(json.dumps(
            {"event": "ascend_queue", "status": "running", "timestamp": "not-a-time"}) + "\n")
    elif damage == "backwards":
        (directory / "stdout.log").write_text(queue_event("running", 100) + "\n"
                                               + queue_event("completed", 99) + "\n")
    elif damage == "missing_release":
        (directory / "stdout.log").write_text(queue_event("running", time.time() - 1) + "\n")
    # A parser may raise, but cleanup must then preserve a conservative debit;
    # restarting a repaired segment must not make unknown device cost free.
    try:
        supervisor._poll(request, record)
    except (ValueError, s.Blocked):
        supervisor.stop_all()
    assert record["status"] != "completed"
    assert record["device_seconds"] > 0
    assert s.read(supervisor.config["budget_ledger"])["device_seconds"] > 0


def test_completed_short_queue_cost_reaches_global_ledger(supervisor_factory):
    supervisor = supervisor_factory()
    request = task(device="npu:0")
    record, directory = saved_record(supervisor, request, status="running", is_npu=True)
    (directory / "stdout.log").write_text(queue_event("running", 100) + "\n"
                                           + queue_event("completed", 100.75) + "\n")
    assert supervisor._poll(request, record)
    assert record["status"] == "completed"
    assert record["device_seconds"] == .75
    assert s.read(supervisor.config["budget_ledger"])["device_seconds"] == .75


@dataclass
class Quota:
    used_bytes: int
    soft_bytes: int
    hard_bytes: int


@pytest.mark.parametrize("boundary,reason", [
    ("deadline", "global_deadline"),
    ("global_hours", "global_device_hour_limit"),
    ("preparation_hours", "preparation_device_hour_limit"),
    ("hard_quota", "storage_reserve_exhausted"),
    ("above_soft_quota", None),
    ("filesystem", "storage_reserve_exhausted"),
    ("rss", "process_tree_rss_limit"),
])
def test_resource_boundaries_stop_before_admission(supervisor_factory, monkeypatch, boundary, reason):
    supervisor = supervisor_factory()
    ledger = s.read(supervisor.config["budget_ledger"])
    if boundary == "deadline":
        ledger["deadline_unix"] = time.time() - 1
    elif boundary == "global_hours":
        ledger["device_seconds"] = ledger["max_device_seconds"]
    elif boundary == "preparation_hours":
        supervisor.config["preparation"] = True
        supervisor.state["device_seconds"] = 48 * 3600
    write_json(supervisor.config["budget_ledger"], ledger)
    used = {"hard_quota": 1901, "above_soft_quota": 1200}.get(boundary, 0)
    quota = Quota(used, 1000, 2000)
    monkeypatch.setattr(s, "current_quota", lambda: quota)
    monkeypatch.setattr(s.shutil, "disk_usage", lambda path: SimpleNamespace(free=99 if boundary == "filesystem" else 10000))
    own = {"pid": os.getpid(), "start_ticks": 1, "ppid": 0, "pgrp": os.getpid(),
           "state": "S", "rss_bytes": 1001 if boundary == "rss" else 100}
    monkeypatch.setattr(s, "process_table", lambda **kwargs: {os.getpid(): own})
    monkeypatch.setattr(s, "family_members", lambda seeds, table: set(table))
    assert supervisor.resource_reason() == reason
    if boundary == "above_soft_quota":
        assert supervisor.resource_cache["available_storage_bytes"] == 800


@pytest.mark.parametrize("used,hard,free,reason", [
    (1999, 2000, 100, None), (2000, 2000, 100, "storage_exhausted"),
    (2001, 2000, 100, "storage_exhausted"), (1500, 2000, 0, "storage_exhausted"),
    (3000, 0, 1, None), (3000, 0, 0, "storage_exhausted"),
])
def test_default_storage_admission_uses_physical_headroom(supervisor_factory, monkeypatch, used, hard, free, reason):
    supervisor = supervisor_factory()
    del supervisor.config["storage_reserve_bytes"]
    monkeypatch.setattr(s, "current_quota", lambda: Quota(used, 1000, hard))
    monkeypatch.setattr(s.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    own = {"pid": os.getpid(), "ppid": 0, "start_ticks": 1, "rss_bytes": 100}
    monkeypatch.setattr(s, "process_table", lambda **kwargs: {os.getpid(): own})
    assert supervisor.resource_reason() == reason


def test_global_ledger_adds_distinct_segments_without_resetting_budget(supervisor_factory, tmp_path):
    first = supervisor_factory(output=tmp_path / "segment-one")
    second = supervisor_factory(config=first.config, output=tmp_path / "segment-two")
    deadline = s.read(first.config["budget_ledger"])["deadline_unix"]
    first.state["tasks"]["one"] = {"status": "completed", "device_seconds": 10}
    first._save()
    second.state["tasks"]["two"] = {"status": "completed", "device_seconds": 20}
    second._save()
    ledger = s.read(first.config["budget_ledger"])
    assert ledger["device_seconds"] == 30
    assert ledger["deadline_unix"] == deadline


def test_latest_polled_device_cost_is_enforced_before_next_dispatch(supervisor_factory):
    supervisor = supervisor_factory()
    ledger = s.read(supervisor.config["budget_ledger"])
    supervisor.state["tasks"]["active"] = {
        "status": "running", "device_seconds": ledger["max_device_seconds"]}
    # _poll updates live tasks before the next _save; admitting another task
    # must not depend on whether the just-observed cost is on disk yet.
    supervisor.last_resources = time.time()
    supervisor.resource_cache = {"available_storage_bytes": 10000, "rss_bytes": 1}
    assert supervisor.resource_reason() == "global_device_hour_limit"


def test_preparation_device_budget_is_cumulative_across_segments(supervisor_factory, tmp_path):
    first = supervisor_factory(output=tmp_path / "prep-one")
    first.config["preparation"] = True
    second = supervisor_factory(config=first.config, output=tmp_path / "prep-two")
    for supervisor, key in ((first, "one"), (second, "two")):
        supervisor.state["tasks"][key] = {"status": "completed", "device_seconds": 24 * 3600}
        supervisor._save()
    second.last_resources = time.time()
    second.resource_cache = {"available_storage_bytes": 10000, "rss_bytes": 1}
    assert second.resource_reason() == "preparation_device_hour_limit"


def test_unavailable_resource_observation_blocks_and_closes_gate(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()

    def unavailable():
        raise RuntimeError("quota unavailable")

    monkeypatch.setattr(s, "current_quota", unavailable)
    monkeypatch.setattr(supervisor, "_start", forbid_launch)
    with pytest.raises(RuntimeError, match="quota unavailable"):
        supervisor.run("A1", [task()])
    assert supervisor.state["stage_b_gate"] == "closed"
    assert supervisor.state["status"] != "completed"


def test_process_table_observation_failure_still_persists_closed_gate(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    monkeypatch.setattr(supervisor, "_start", forbid_launch)
    monkeypatch.setattr(s, "current_quota", lambda: Quota(0, 10000, 20000))

    def unreadable(**kwargs):
        raise RuntimeError("unreadable owned process table")

    monkeypatch.setattr(s, "process_table", unreadable)
    with pytest.raises(RuntimeError, match="unreadable owned process table"):
        supervisor.run("A1", [task()])
    state = s.read(supervisor.state_path)
    assert state.get("stage_b_gate") == "closed"
    assert state["status"] != "running"


@pytest.mark.parametrize("artifact", ["result.json", "heartbeat.json", "queue.json"])
def test_bad_artifact_cannot_interrupt_exception_cleanup(supervisor_factory, monkeypatch, artifact):
    supervisor = supervisor_factory()
    request = task(device="npu:0" if artifact == "queue.json" else "cpu")
    record, directory = saved_record(supervisor, request, status="running", is_npu=artifact == "queue.json")
    (directory / artifact).write_text('{"partial":')
    supervisor.stop_reason = "injected_controller_exception"
    supervisor.stop_all()
    assert supervisor.state["stage_b_gate"] == "closed"
    assert supervisor.state["status"] != "completed"
    assert record["status"] != "completed"
    assert s.read(supervisor.state_path)["stage_b_gate"] == "closed"
    if request["device"] == "cpu":
        assert record["device_seconds"] == 0
        assert s.read(supervisor.config["budget_ledger"])["device_seconds"] == 0


class FakeProcess:
    pid = 987654321

    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = None
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def test_process_identity_failure_after_spawn_still_reaps_and_closes(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    monkeypatch.setattr(supervisor, "resource_reason", lambda: None)
    processes = []

    def popen(*args, **kwargs):
        process = FakeProcess(kwargs["stdout"])
        processes.append(process)
        return process

    original_identity = s.proc_identity

    def identity(pid, **kwargs):
        if pid == FakeProcess.pid:
            raise RuntimeError("injected process identity read failure")
        return original_identity(pid, **kwargs)

    monkeypatch.setattr(s.subprocess, "Popen", popen)
    monkeypatch.setattr(s, "proc_identity", identity)
    monkeypatch.setattr(s.os, "killpg", lambda pgid, sig: setattr(processes[0], "returncode", -sig))
    with pytest.raises(RuntimeError, match="identity read failure"):
        supervisor.run("A1", [task()])
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert processes[0].waited
    assert processes[0].stdout.closed
    assert supervisor.state["stage_b_gate"] == "closed"


def test_cleanup_finds_recorded_orphan_after_task_root_has_exited(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    request = task()
    record, _ = saved_record(supervisor, request)
    orphan = {"pid": 987654322, "start_ticks": 123, "ppid": os.getpid(),
              "pgrp": 987654322, "state": "S", "rss_bytes": 10}
    supervisor.state["all_seen_processes"][f'{orphan["pid"]}:{orphan["start_ticks"]}'] = orphan
    alive = {orphan["pid"]}

    def identity(pid, **kwargs):
        return orphan.copy() if pid in alive else None

    def kill(pid_or_group, sig):
        alive.discard(pid_or_group)

    monkeypatch.setattr(s, "proc_identity", identity)
    monkeypatch.setattr(s, "process_table", lambda **kwargs: {p: orphan.copy() for p in alive})
    monkeypatch.setattr(s.os, "killpg", kill)
    monkeypatch.setattr(s.os, "kill", kill)
    supervisor.stop_all()
    assert not alive, "a dead task root does not establish descendant termination"
    assert supervisor.state["stage_b_gate"] == "closed"


def test_reused_pid_and_process_group_are_not_owned_descendants(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    original = {"pid": 987654322, "start_ticks": 123, "ppid": os.getpid(),
                "pgrp": 987654322, "state": "S", "rss_bytes": 10}
    reused = {**original, "start_ticks": 999, "ppid": 1}
    supervisor.state["tasks"]["old"] = {"status": "running", "process": original}
    supervisor.state["all_seen_processes"]["987654322:123"] = original
    monkeypatch.setattr(s, "process_table", lambda **kwargs: {reused["pid"]: reused})
    assert supervisor.owned_members() == {}, "same pgrp number is not verified process ownership"


@pytest.mark.parametrize("damage", [None, "bad_hash", "wrong_source", "second_failure"])
def test_infrastructure_retry_is_same_identity_verified_and_at_most_once(supervisor_factory, damage):
    supervisor = supervisor_factory()
    request = task()
    record, directory = saved_record(supervisor, request, status="running")
    checkpoint = directory / "full-checkpoint.pt"
    checkpoint.write_bytes(b"complete test checkpoint")
    answer = result(request, status="failed", failure_class="infrastructure_transient",
                    task_sha256=s.sha256(record["task_path"]),
                    checkpoint={"path": str(checkpoint), "sha256": s.sha256(checkpoint)})
    if damage == "bad_hash":
        checkpoint.write_bytes(b"corrupted checkpoint")
    elif damage == "wrong_source":
        answer["source_identity"] = {"source": "wrong-tree"}
    elif damage == "second_failure":
        record["attempts"].append(deepcopy(record["attempts"][0]))
    write_json(directory / "result.json", answer)
    supervisor._poll(request, record)
    assert record["status"] == ("pending" if damage is None else "failed")
    if damage is None:
        assert record["retry_checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("damage", [None, "wrong_class", "changed_result", "wrong_task_hash"])
def test_only_bound_capacity_failure_can_be_tolerated(supervisor_factory, monkeypatch, damage):
    supervisor = supervisor_factory()
    request = task(phase="preflight", purpose="capacity_calibration")
    record, directory = saved_record(supervisor, request, status="failed")
    record["stage"] = "capacity-large"
    answer = result(request, status="failed", failure_class="capacity_error",
                    task_sha256=s.sha256(record["task_path"]))
    if damage == "wrong_class":
        answer["failure_class"] = "unknown_numerical_failure"
    if damage == "wrong_task_hash":
        answer["task_sha256"] = "0" * 64
    write_json(directory / "result.json", answer)
    record["result_sha256"] = s.sha256(directory / "result.json")
    if damage == "changed_result":
        write_json(directory / "result.json", {**answer, "changed_after_exit": True})
    supervisor._save()
    monkeypatch.setattr(supervisor, "resource_reason", lambda: None)
    monkeypatch.setattr(supervisor, "_start", forbid_launch)
    if damage is None:
        answers = supervisor.run("capacity-large", [request], allowed_failures={"capacity_error"})
        assert answers[request["id"]]["status"] == "failed"
        assert record["status"] == "failed"
    else:
        with pytest.raises(s.Blocked):
            supervisor.run("capacity-large", [request], allowed_failures={"capacity_error"})


def test_capacity_exception_is_never_available_to_learning_tasks(supervisor_factory):
    supervisor = supervisor_factory()
    with pytest.raises(ValueError, match="capacity"):
        supervisor.run("a1", [task(phase="a1", purpose="capacity_calibration")], allowed_failures={"capacity_error"})
    assert not supervisor.state["tasks"]


@pytest.mark.parametrize("start,target,eval_tokens,confirmation,expected", [
    (0, 2**18, 16384, False, 32800),
    (2**18, 2**20, 16384, False, 32800),
    (2**20, 2 * 2**20, 16384, False, 32800),
    (0, 12 * 2**20, 16384, False, 213024),
    (0, 12 * 2**20, 8192, False, 106528),
    (12 * 2**20, 12 * 2**20, 256, True, 131616),
])
def test_evaluation_reservation_counts_real_segment_boundaries(start, target, eval_tokens, confirmation, expected):
    request = task(target_tokens=target, evaluation_start_tokens=start, eval_tokens=eval_tokens,
                   eval_interval_tokens=2**20, allow_confirmation=confirmation, confirmation_tokens=131072)
    assert s.evaluation_bound(request) == expected


def fake_spawn(monkeypatch):
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        process = FakeProcess(kwargs["stdout"])
        process.returncode = 0
        return process

    monkeypatch.setattr(s.subprocess, "Popen", popen)
    monkeypatch.setattr(s, "proc_identity", lambda pid, **kwargs: {
        "pid": pid, "start_ticks": 123, "pgrp": pid, "ppid": os.getpid(), "state": "S", "rss_bytes": 1})
    return calls


def pending_record(supervisor, request):
    record = {"status": "pending", "stage": request.get("phase", "a1"), "attempts": [],
              "device": request["device"], "device_seconds": 0, "model_id": request["model_id"]}
    supervisor.state["tasks"][request["id"]] = record
    return record


@pytest.mark.parametrize("external", [False, True])
def test_evaluation_budget_is_reserved_before_spawn_and_charges_retry(supervisor_factory, monkeypatch, external):
    supervisor = supervisor_factory()
    calls = fake_spawn(monkeypatch)
    request = task(phase="a1", target_tokens=2**18, eval_tokens=16384,
                   evaluation_budget_key="one-model", evaluation_token_limit=65600)
    if external:
        supervisor.config["prior_evaluation_reservations"] = {"one-model": 32800}
    else:
        supervisor.state["tasks"]["previous"] = {"status": "completed", "evaluation_budget_key": "one-model",
                                                 "evaluation_reserved_tokens": 32800}
    record = pending_record(supervisor, request)
    supervisor._start(request, record)
    assert len(calls) == 1
    assert record["evaluation_reserved_tokens"] == 32800
    with pytest.raises(s.Blocked, match="evaluation_token_limit"):
        supervisor._start(request, record)
    assert len(calls) == 1 and len(record["attempts"]) == 1


def test_preparation_claim_includes_other_segments_before_spawn(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    calls = fake_spawn(monkeypatch)
    ledger = s.read(supervisor.config["budget_ledger"])
    ledger["preparation_token_reservations"] = {"another-segment": {"reserved_tokens": 64 * 2**20 - 16}}
    write_json(supervisor.config["budget_ledger"], ledger)
    request = task(phase="preflight", budget_tokens=32)
    record = pending_record(supervisor, request)
    with pytest.raises(s.Blocked, match="preparation_token_limit"):
        supervisor._start(request, record)
    assert calls == [] and record["attempts"] == []


def test_long_queue_ids_stay_distinct_and_within_helper_limit(supervisor_factory, monkeypatch):
    supervisor = supervisor_factory()
    calls = fake_spawn(monkeypatch)
    supervisor.config["run_id"] = "run" * 30
    for suffix in ("a", "b"):
        request = task(id="model" * 35 + suffix, device="npu:0")
        supervisor._start(request, pending_record(supervisor, request))
    job_ids = [command[command.index("--job-id") + 1] for command in calls]
    assert all(len(job_id) <= 128 for job_id in job_ids)
    assert len(set(job_ids)) == 2


@pytest.fixture(scope="module")
def generated_round3_subjects():
    # Exercise the production catalog: hand-written list-only fixtures missed
    # the multiscale tuples that blocked the first real preflight submission.
    from pilotlab.catalog3 import generate_subjects
    return generate_subjects()


def contains_tuple(value):
    if isinstance(value, tuple):
        return True
    if isinstance(value, dict):
        return any(contains_tuple(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_tuple(item) for item in value)
    return False


def test_all_216_generated_tasks_freeze_identically_before_and_after_json_load(generated_round3_subjects, tmp_path):
    subjects = generated_round3_subjects
    assert len(subjects) == 216
    assert any(row["group"].startswith("multiscale") and contains_tuple(row["model"]) for row in subjects)
    for row in subjects:
        request = task(id=row["id"] + "-preflight", model_id=row["id"], model=row["model"],
                       phase="preflight", target_tokens=1024)
        original = deepcopy(request)
        path = tmp_path / request["id"] / "task.json"
        assert s.freeze_json(path, request) == path
        before = path.read_bytes(), path.stat().st_mtime_ns
        assert s.freeze_json(path, request) == path
        assert s.freeze_json(path, s.read(path)) == path
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        assert request == original, "freezing must not normalize the caller's in-memory model in place"


def test_json_normalization_still_rejects_real_generated_model_change(generated_round3_subjects, tmp_path):
    row = next(row for row in generated_round3_subjects if row["group"].startswith("multiscale"))
    request = task(model_id=row["id"], model=row["model"], phase="preflight")
    path = s.freeze_json(tmp_path / "task.json", request)
    before = path.read_bytes(), path.stat().st_mtime_ns
    changed = s.read(path)
    changed["model"]["config"]["d"] *= 2
    with pytest.raises(ValueError, match="immutable request differs"):
        s.freeze_json(path, changed)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
