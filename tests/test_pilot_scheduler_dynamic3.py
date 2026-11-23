"""Bounded filesystem/process simulations; never allocate or probe a device."""
from copy import deepcopy
from pathlib import Path

import pytest

from pilotlab import scheduler_dynamic3 as dynamic
from pilotlab import scheduler_shared3 as shared
from pilotlab.io import sha256
from pilotlab.scheduler3 import Blocked, read
from pilotlab.shared_npu3 import read_admission_chain, worker_spec_digest
from test_pilot_scheduler_shared3 import launch_fixture, write


class Process:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        assert self.returncode is not None
        return self.returncode


@pytest.fixture
def owner(tmp_path, monkeypatch, request):
    supervisor, rows = launch_fixture(tmp_path, count=1)
    supervisor.config.update(dynamic_shared_admission=True, max_npu_devices=8, max_workers=80,
        npu_workers_per_device=16, max_cpu_workers=40, max_cpu_task_threads=160,
        max_rss_bytes=1024*2**30, dynamic_owner_max_total_workers=getattr(request, "param", 256),
        dynamic_owner_max_commands=getattr(request, "param", 256)+1)
    supervisor.state["controller_identity"] = {"pid": 98765432, "start_ticks": 10}
    supervisor.state["peak_actual_npu_concurrency"] = 0
    supervisor.resource_reason = lambda: None
    supervisor.owned_members = lambda: {}
    processes = []

    def spawn(*args, **kwargs):
        process = Process(90000000+len(processes))
        processes.append(process)
        return process

    monkeypatch.setattr(shared.subprocess, "Popen", spawn)
    monkeypatch.setattr(shared, "proc_identity", lambda pid, **kwargs: {"pid": pid, "start_ticks": 11, "pgrp": pid})
    rows[0].update(device="npu:0", cpu_threads=4, batch_size=2)
    supervisor.state["tasks"][rows[0]["id"]]["device"] = "npu:0"
    key = supervisor.start_group(rows, "npu")
    observe(supervisor, key)
    return supervisor, rows[0], key


def observe(supervisor, key, *, target=1):
    entry = supervisor.state["shared_groups"][key]
    manifest = read(entry["manifest_path"])
    digest = entry["manifest_sha256"]
    group = {"job_id": entry["job_id"], "manifest_sha256": digest, "backend": "npu", "status": "running",
        "allocation_acquired_unix": 100., "allocation_released_unix": None, "physical_device_seconds": 5.,
        "physical_id": int(key.split("-")[-1]), "all_seen_processes": {},
        "policy": {"admission_limit": target, "unmeasured_exploration_used": True},
        "admission": {"next_sequence": 0, "last_sha256": digest, "closed": False,
            "commands": [], "accepted_task_ids": [worker["id"] for worker in manifest["workers"]]},
        "workers": {worker["id"]: {"status": "running", "output": worker["output"],
            "worker_spec_sha256": worker_spec_digest(worker), "admission_sequence": None,
            "admission_command_sha256": digest} for worker in manifest["workers"]}}
    write(Path(entry["path"])/"group.json", group)
    return group


def add_task(supervisor, name, **fields):
    task = {"id": name, "model_id": "model-"+name, "phase": "a1", "device": "npu:0", "target_tokens": 100,
        "source_identity": {"source": "frozen"}, "cpu_threads": 4, "checkpoint_reservation_bytes": 16*2**20,
        "reserved_rss_bytes": 2**30, "reserved_hbm_bytes": 2**30, **fields}
    supervisor.state["tasks"][name] = {"status": "pending", "stage": task["phase"], "model_id": task["model_id"],
                                      "device": task["device"], "attempts": []}
    return task


def complete(supervisor, task, key, *, status="completed", code=0):
    entry = supervisor.state["shared_groups"][key]
    group = read(Path(entry["path"])/"group.json")
    record = supervisor.state["tasks"][task["id"]]
    result = {"status": status, "source_identity": task["source_identity"],
        "task_sha256": sha256(record["task_path"]), "training": {"tokens": task["target_tokens"]}}
    if status == "recipe_rejected":
        result["failure_class"] = "declared_optimizer_instability"
    path = Path(record["attempts"][-1]["path"])/"result.json"
    write(path, result)
    group["workers"][task["id"]].update(status="allowed_failure" if code else "completed", return_code=code,
        result_sha256=sha256(path), cleanup_verified=True, descendant_cleanup_verified=True,
        cleanup_verified_unix=120., remaining_descendants=[], finished_unix=119.,
        process={"pid": 99000000, "start_ticks": 12, "pgrp": 99000000})
    write(Path(entry["path"])/"group.json", group)
    supervisor._group_observation_cache = {}
    return group


def acknowledge(supervisor, key):
    entry = supervisor.state["shared_groups"][key]
    group = read(Path(entry["path"])/"group.json")
    chain = read_admission_chain(read(entry["manifest_path"]), entry["manifest_path"])
    group["admission"].update(next_sequence=chain["next_sequence"], last_sha256=chain["last_sha256"],
        closed=chain["closed"], commands=[{k: v for k, v in row.items() if k != "command"} for row in chain["commands"]],
        accepted_task_ids=[worker["id"] for worker in chain["published_workers"]])
    for command in chain["commands"]:
        for worker in command["command"].get("workers", []):
            group["workers"].setdefault(worker["id"], {"status": "pending", "output": worker["output"],
                "worker_spec_sha256": worker_spec_digest(worker), "admission_sequence": command["sequence"],
                "admission_command_sha256": command["sha256"]})
    write(Path(entry["path"])/"group.json", group)
    supervisor._group_observation_cache = {}
    return group


def release(supervisor, key):
    entry = supervisor.state["shared_groups"][key]
    group = acknowledge(supervisor, key)
    group.update(status="completed", allocation_released_unix=130., queue_released_unix=130.,
                 remaining_descendants=[], physical_device_seconds=30.)
    write(Path(entry["path"])/"group.json", group)
    write(Path(entry["path"])/"queue.json", {"job_id": entry["job_id"], "status": "completed", "return_code": 0})
    supervisor.processes[key].returncode = 0
    return group


def test_individual_completion_releases_task_slot_while_owner_and_sibling_remain(owner):
    supervisor, task, key = owner
    sibling = add_task(supervisor, "sibling")
    supervisor.append_group(key, [sibling])
    acknowledge(supervisor, key)
    complete(supervisor, task, key)
    assert supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    assert supervisor.state["tasks"][task["id"]]["status"] == "completed"
    supervisor.verify_completed(task, supervisor.state["tasks"][task["id"]])
    assert supervisor.owner_running(key)
    assert supervisor.state["tasks"]["sibling"]["status"] == "running"
    assert supervisor.actual_npu_concurrency([]) == 1


@pytest.mark.parametrize("damage", ["cleanup", "descendants", "result", "worker_spec", "task"])
def test_early_completion_requires_full_worker_binding_and_cleanup(owner, damage):
    supervisor, task, key = owner
    group = complete(supervisor, task, key)
    worker = group["workers"][task["id"]]
    if damage == "cleanup":
        worker["cleanup_verified"] = False
    elif damage == "descendants":
        worker["remaining_descendants"] = [{"pid": 123}]
    elif damage == "result":
        worker["result_sha256"] = "changed"
    elif damage == "worker_spec":
        worker["worker_spec_sha256"] = "changed"
    else:
        path = Path(supervisor.state["tasks"][task["id"]]["task_path"])
        path.chmod(0o600)
        path.write_text("{}")
    write(Path(supervisor.state["shared_groups"][key]["path"])/"group.json", group)
    with pytest.raises((Blocked, ValueError)):
        supervisor._poll(task, supervisor.state["tasks"][task["id"]])


@pytest.mark.parametrize("code", [0, 1])
def test_declared_recipe_rejection_is_returned_without_hiding_later_owner_failure(owner, code):
    supervisor, task, key = owner
    group = complete(supervisor, task, key, status="recipe_rejected", code=code)
    assert supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    assert supervisor.state["tasks"][task["id"]]["status"] == "recipe_rejected"
    group.update(status="failed", reason="later sibling failed")
    write(Path(supervisor.state["shared_groups"][key]["path"])/"group.json", group)
    with pytest.raises(Blocked, match="dynamic_owner_failed"):
        supervisor.supervision_tick()


def test_completed_result_does_not_mask_a_later_group_failure_in_run_dynamic(owner):
    supervisor, task, key = owner
    complete(supervisor, task, key)
    stopped = []
    supervisor.stop_all = lambda: stopped.append(True)
    offered = False

    def producer(results):
        nonlocal offered
        if not offered:
            offered = True
            return {"requests": [{"stage": "a1", "tasks": [task]}], "done": False}
        if results:
            entry = supervisor.state["shared_groups"][key]
            group = read(Path(entry["path"])/"group.json")
            group.update(status="failed", reason="later owner cleanup failure")
            write(Path(entry["path"])/"group.json", group)
        return {"requests": [], "done": bool(results)}

    with pytest.raises(Blocked, match="dynamic_owner_failed"):
        supervisor.run_dynamic("fixture", producer)
    assert stopped == [True]


def test_stage_drain_closes_owner_once_then_requires_full_lease_release(owner):
    supervisor, task, key = owner
    complete(supervisor, task, key)
    supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    assert supervisor.stage_drained() is False
    assert supervisor.stage_drained() is False
    entry = supervisor.state["shared_groups"][key]
    assert len(entry["admission_commands"]) == 1
    assert entry["admission_commands"][0]["kind"] == "close"
    release(supervisor, key)
    assert supervisor.stage_drained() is True
    assert not supervisor.processes and not supervisor.handles
    assert supervisor.device_seconds() == 30.
    # Final boundary checks reopen terminal evidence despite the hot cache.
    (Path(entry["path"])/"queue.json").unlink()
    with pytest.raises(Blocked, match="lease_release_missing"):
        supervisor.stage_drained()


@pytest.mark.parametrize("entry_point", ["supervision_tick", "poll_worker"])
def test_dynamic_owner_finishes_between_observation_and_exit_check(owner, monkeypatch, entry_point):
    supervisor, task, key = owner
    record = supervisor.state["tasks"][task["id"]]
    def exits(group_id):
        assert group_id == key
        if key in supervisor.processes and supervisor.processes[key].returncode is None:
            complete(supervisor, task, key)
            release(supervisor, key)
        return False
    monkeypatch.setattr(supervisor, "owner_running", exits)

    if entry_point == "supervision_tick":
        supervisor.supervision_tick()
        assert supervisor.state["shared_groups"][key]["release_verified"]
    assert supervisor._poll(task, record)
    assert record["status"] == "completed"
    supervisor.verify_completed(task, record)


@pytest.mark.parametrize("completed", [True, False])
def test_global_stop_discards_live_cache_and_reads_individual_terminal_receipts(owner, completed):
    supervisor, task, key = owner
    entry = supervisor.state["shared_groups"][key]
    stale = read(Path(entry["path"])/"group.json")
    group = complete(supervisor, task, key,
        status="completed" if completed else "budget_exhausted", code=0 if completed else 2)
    group.update(status="failed", reason="signal_15", allocation_released_unix=130.,
        queue_released_unix=130., remaining_descendants=[], physical_device_seconds=30.)
    write(Path(entry["path"])/"group.json", group)
    write(Path(entry["path"])/"queue.json", {"job_id":entry["job_id"],"status":"failed","return_code":1})
    supervisor.processes[key].returncode = 1
    supervisor._group_observation_cache = {key:stale}
    supervisor.stop_reason = "selected_recipe_confirmation_failed"
    supervisor.stop_all()
    record = supervisor.state["tasks"][task["id"]]
    assert record["status"] == ("completed" if completed else "stopped_budget")
    assert record["device_seconds"] == record["attempts"][-1]["device_seconds"] == 0
    assert record["result_sha256"] == group["workers"][task["id"]]["result_sha256"]
    assert supervisor.state["stage_b_gate"] == "closed"
    assert supervisor.state["blocking_reason"] == "selected_recipe_confirmation_failed"
    assert supervisor.state["actual_npu_concurrency"] == 0
    assert not supervisor.processes


def test_append_reserves_and_records_before_publish_then_replays_same_bytes_once(owner, monkeypatch):
    supervisor, _, key = owner
    task = add_task(supervisor, "append", evaluation_budget_key="new-model", eval_tokens=4, evaluation_token_limit=1000)
    snapshots = []
    publish = supervisor.publish_admissions
    monkeypatch.setattr(supervisor, "publish_admissions", lambda group_id: snapshots.append(deepcopy(supervisor.state)))
    supervisor.append_group(key, [task])
    entry = supervisor.state["shared_groups"][key]
    command = entry["admission_commands"][0]
    assert not Path(command["inbox_path"]).exists()
    assert snapshots[0]["tasks"][task["id"]]["status"] == "running"
    assert snapshots[0]["tasks"][task["id"]]["evaluation_reserved_tokens"] > 0
    attempts = deepcopy(supervisor.state["tasks"][task["id"]]["attempts"])
    cost = supervisor.state["tasks"][task["id"]]["evaluation_reserved_tokens"]
    publish(key)
    publish(key)
    assert sha256(command["inbox_path"]) == command["sha256"]
    assert supervisor.state["tasks"][task["id"]]["attempts"] == attempts
    assert supervisor.state["tasks"][task["id"]]["evaluation_reserved_tokens"] == cost
    assert len(list(Path(command["inbox_path"]).parent.glob("*.json"))) == 1
    # A valid published, not-yet-acknowledged tail is not a missing worker.
    supervisor.group_observation(key)
    acknowledge(supervisor, key)
    assert task["id"] in supervisor.group_observation(key)["workers"]


@pytest.mark.parametrize("point", ["outbox", "state", "inbox", "after_inbox"])
def test_append_failure_distinguishes_proved_nonpublication_from_possible_execution(owner, monkeypatch, point):
    supervisor, _, key = owner
    task = add_task(supervisor, "append", evaluation_budget_key="imported-model", eval_tokens=4, evaluation_token_limit=1000)
    record = supervisor.state["tasks"][task["id"]]
    record.update(evaluation_reserved_tokens=73, evaluation_budget_key="imported-model")
    frozen = dynamic.freeze_json
    once = [False]

    def fail_freeze(path, value):
        target = Path(path)
        if not once[0] and ((point == "outbox" and target.parent.name == "outbox")
                or (point in {"inbox", "after_inbox"} and target.parent.name == "inbox")):
            once[0] = True
            if point == "after_inbox":
                frozen(path, value)
            raise OSError("injected publication interruption")
        return frozen(path, value)

    def fail_state():
        if point == "state" and not once[0]:
            once[0] = True
            raise OSError("injected publication interruption")

    monkeypatch.setattr(dynamic, "freeze_json", fail_freeze)
    monkeypatch.setattr(supervisor, "_save", fail_state)
    with pytest.raises(OSError, match="publication interruption"):
        supervisor.append_group(key, [task])
    attempt = record["attempts"][-1]
    if point == "after_inbox":
        assert record["status"] == "running" and not attempt.get("admission_not_published")
        assert record["evaluation_reserved_tokens"] > 73
        assert Path(attempt["admission_inbox_path"]).is_file()
    else:
        assert record["status"] == "failed" and attempt["status"] == "not_started"
        assert attempt["admission_not_published"] is True
        assert record["evaluation_reserved_tokens"] == 73
        assert not Path(attempt["admission_inbox_path"]).exists()
    with pytest.raises(Blocked, match="previously_aborted"):
        supervisor.publish_admissions(key)


def test_idle_owners_count_against_hard_cap_and_probe_drains_only_one(owner):
    supervisor, task, first = owner
    complete(supervisor, task, first)
    supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    for index in range(1, 8):
        row = add_task(supervisor, f"existing-{index}")
        key = supervisor.start_group([row], "npu")
        observe(supervisor, key)
    probe = add_task(supervisor, "probe", phase="preflight", purpose="capacity_calibration")
    for _ in range(3):
        supervisor.start_ready_tasks([probe], 80)
    assert len(supervisor.state["shared_groups"]) == 8
    closing = [entry for entry in supervisor.state["shared_groups"].values() if entry.get("close_requested")]
    assert len(closing) == 1 and closing[0]["close_requested"] == "capacity_probe_lane"
    assert supervisor.state["tasks"][probe["id"]]["status"] == "pending"
    release(supervisor, first)
    supervisor.supervision_tick()
    supervisor.start_ready_tasks([probe], 80)
    record = supervisor.state["tasks"][probe["id"]]
    assert record["status"] == "running"
    entry = supervisor.state["shared_groups"][record["shared_group"]]
    assert not entry["dynamic_admission"] and entry["task_ids"] == [probe["id"]]


def test_mixed_probe_backlog_does_not_reduce_training_owner_target(owner):
    supervisor, _, key = owner
    observe(supervisor, key, target=4)
    rows = [add_task(supervisor, "probe", phase="preflight", purpose="capacity_calibration")]
    rows += [add_task(supervisor, f"training-{index}") for index in range(3)]
    supervisor.start_ready_tasks(rows, 80)
    assert len(supervisor.state["shared_groups"][key]["task_ids"]) == 4
    assert all(supervisor.state["tasks"][row["id"]]["shared_group"] == key for row in rows[1:])
    assert supervisor.state["tasks"][rows[0]["id"]]["shared_group"] != key


def test_single_extra_pending_worker_enables_only_the_declared_first_exploration(owner):
    supervisor, _, key = owner
    group = observe(supervisor, key, target=1)
    group["policy"]["unmeasured_exploration_used"] = False
    write(Path(supervisor.state["shared_groups"][key]["path"])/"group.json", group)
    supervisor.config["max_npu_devices"] = 1
    rows = [add_task(supervisor, f"waiting-{index}") for index in range(3)]
    supervisor.start_ready_tasks(rows, 80)
    assert len(supervisor.state["shared_groups"][key]["task_ids"]) == 2
    assert sum(record["status"] == "pending" for record in supervisor.state["tasks"].values()) == 2


@pytest.mark.parametrize("owner", [1], indirect=True)
def test_finite_owner_limit_drains_then_admits_remaining_task_to_new_owner(owner):
    supervisor, task, key = owner
    entry = supervisor.state["shared_groups"][key]
    # The frozen limit applies independently of the larger global task set.
    assert read(entry["manifest_path"])["dynamic_admission"]["max_total_workers"] == 1
    supervisor.config["max_npu_devices"] = 1
    next_task = add_task(supervisor, "next-owner")
    supervisor.start_ready_tasks([next_task], 80)
    assert entry["close_requested"] == "finite_owner_admission_limit"
    assert supervisor.state["tasks"][next_task["id"]]["status"] == "pending"
    complete(supervisor, task, key)
    supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    release(supervisor, key)
    supervisor.supervision_tick()
    supervisor.start_ready_tasks([next_task], 80)
    assert supervisor.state["tasks"][next_task["id"]]["status"] == "running"
    assert supervisor.state["tasks"][next_task["id"]]["shared_group"] != key


def test_cpu_reservations_include_the_extra_waiter_and_stop_at_160_threads(owner):
    supervisor, _, key = owner
    rows = [add_task(supervisor, f"cpu-{index}", device="cpu") for index in range(45)]
    supervisor.start_ready_tasks(rows, 80)
    running_cpu = [row for row in supervisor.state["tasks"].values() if row.get("device") == "cpu" and row["status"] == "running"]
    assert len(running_cpu) == 40 and sum(row["cpu_threads"] for row in running_cpu) == 160
    assert supervisor.owner_running(key)


def test_dynamic_capacity_admission_is_rejected_before_budget_or_publication(owner):
    supervisor, _, key = owner
    task = add_task(supervisor, "bad-probe", phase="preflight", purpose="capacity_calibration")
    before = deepcopy(supervisor.state)
    with pytest.raises(Blocked, match="not_exclusive"):
        supervisor.append_group(key, [task])
    assert supervisor.state == before


def test_exited_worker_waits_for_descendant_cleanup_without_blocking_healthy_owner(owner):
    supervisor, task, key = owner
    entry = supervisor.state["shared_groups"][key]
    group = read(Path(entry["path"])/"group.json")
    group["workers"][task["id"]].update(status="exited_waiting_cleanup", return_code=0,
                                        remaining_descendants=[{"pid": 123}])
    write(Path(entry["path"])/"group.json", group)
    supervisor.supervision_tick()
    assert supervisor._poll(task, supervisor.state["tasks"][task["id"]]) is False
    assert supervisor.state["tasks"][task["id"]]["status"] == "running"
    complete(supervisor, task, key)
    assert supervisor._poll(task, supervisor.state["tasks"][task["id"]]) is True


@pytest.mark.parametrize("field,ceiling", [("max_inflight_checkpoint_bytes", 2**28+2*16*2**20),
                                         ("max_cpu_threads", 12), ("max_rss_bytes", 6*2**30)])
def test_append_and_waiting_tasks_share_global_reservation_limits(owner, field, ceiling):
    supervisor, _, key = owner
    supervisor.config.update(max_npu_devices=1, **{field: ceiling})
    observe(supervisor, key, target=16)
    tasks = [add_task(supervisor, f"extra-{index}") for index in range(10)]
    supervisor.start_ready_tasks(tasks, 80)
    assert len(supervisor.state["shared_groups"][key]["task_ids"]) == 3
    assert all(task["target_tokens"] == 100 for task in tasks)


def test_owner_target_never_admits_more_than_sixteen_workers(owner):
    supervisor, _, key = owner
    supervisor.config["max_npu_devices"] = 1
    observe(supervisor, key, target=16)
    tasks = [add_task(supervisor, f"extra-{index}") for index in range(20)]
    supervisor.start_ready_tasks(tasks, 80)
    assert len(supervisor.state["shared_groups"][key]["task_ids"]) == 16
    assert len(supervisor.state["shared_groups"]) == 1


def test_resource_based_pool_fills_eight_cards_and_the_cpu_pool(owner):
    supervisor, _, first = owner
    supervisor.config.update(shared_execution=True, max_workers=None, max_cpu_threads=None)
    owner_ids = [first]
    for index in range(1, 8):
        task = add_task(supervisor, f"seed-card-{index}")
        owner_ids.append(supervisor.start_group([task], "npu"))
    for key in owner_ids:
        observe(supervisor, key, target=16)

    cpu = [add_task(supervisor, f"cpu-{n}", device="cpu") for n in range(42)]
    npu = [add_task(supervisor, f"npu-{n}") for n in range(130)]
    limit = dynamic.worker_limit(supervisor.config)
    supervisor.start_ready_tasks(cpu + npu, limit - len(owner_ids))

    records = supervisor.state["tasks"].values()
    active_cpu = [row for row in records if row["status"] == "running" and row["device"] == "cpu"]
    assert len(active_cpu) == 40 and sum(row["cpu_threads"] for row in active_cpu) == 160
    for key in owner_ids:
        assert len(supervisor.state["shared_groups"][key]["task_ids"]) == 16
    assert sum(row["status"] == "running" for row in records) == limit == 168
    assert sum(row["status"] == "pending" for row in records) == 12


def test_terminal_owner_cache_is_finite_and_stage_boundary_does_not_use_it(owner, monkeypatch):
    supervisor, task, key = owner
    complete(supervisor, task, key)
    supervisor._poll(task, supervisor.state["tasks"][task["id"]])
    supervisor.stage_drained()
    release(supervisor, key)
    assert supervisor.stage_drained()
    entry = supervisor.state["shared_groups"][key]
    path = Path(entry["path"])/"group.json"
    path.write_text("{corrupt")
    supervisor._group_observation_cache = {}
    assert supervisor.group_observation(key)["status"] == "completed"
    supervisor.config["terminal_group_recheck_seconds"] = 0
    supervisor._group_observation_cache = {}
    with pytest.raises(ValueError):
        supervisor.group_observation(key)


def test_large_task_exclusive_release_and_retry_preserves_reservation(owner):
    supervisor, _, normal = owner
    supervisor.config.update(exclusive_max_hbm_bytes=52*2**30)
    large = add_task(supervisor, "large", reserved_hbm_bytes=int(47.4189453125*2**30),
                     evaluation_budget_key="large", eval_tokens=4, evaluation_token_limit=1000)
    impossible = add_task(supervisor, "too-large", reserved_hbm_bytes=53*2**30)
    supervisor.start_ready_tasks([impossible, large], 2)
    record = supervisor.state["tasks"]["large"]
    key = record["shared_group"]
    entry = supervisor.state["shared_groups"][key]
    manifest = read(entry["manifest_path"])
    assert manifest["exclusive_hbm"] and manifest["max_hbm_bytes"] == 52*2**30
    assert manifest["max_parallel"] == 1 and "dynamic_admission" not in manifest
    assert read(supervisor.state["shared_groups"][normal]["manifest_path"])["max_hbm_bytes"] == 48*2**30
    assert supervisor.state["tasks"]["too-large"]["reason"] == "reservation_exceeds_exclusive_hbm_limit"
    reserved = record["evaluation_reserved_tokens"]
    group = observe(supervisor, key)
    worker = group["workers"]["large"]
    worker.update(status="not_started", reason="hbm_admission_deferred")
    group.update(status="completed", allocation_released_unix=130., queue_released_unix=130.,
                 remaining_descendants=[], capacity_deferral={"task_id":"large", "reason":"hbm_reservations_full"})
    write(Path(entry["path"])/"group.json", group)
    write(Path(entry["path"])/"queue.json", {"job_id":entry["job_id"], "status":"completed", "return_code":0})
    supervisor.processes[key].returncode = 0
    supervisor._group_observation_cache = {}
    assert supervisor._poll(large, record) and record["status"] == "pending" and entry["release_verified"]
    other = add_task(supervisor, "other")
    supervisor.start_ready_tasks([large, other], 1)
    assert supervisor.state["tasks"]["other"]["status"] == "running" and record["status"] == "pending"
    record["hbm_retry_after_unix"] = 0
    supervisor.start_ready_tasks([large], 1)
    assert record["shared_group"] != key and record["evaluation_reserved_tokens"] == reserved


def test_shared_owner_drains_unstarted_admissions_before_hbm_requeue(owner):
    supervisor, task, key = owner
    supervisor.config["exclusive_max_hbm_bytes"] = 52*2**30
    initial = add_task(supervisor, "initial")
    key = supervisor.start_group([initial], "npu")
    group = observe(supervisor, key)
    sibling = add_task(supervisor, "queued")
    supervisor.append_group(key, [sibling])
    entry = supervisor.state["shared_groups"][key]
    group["capacity_deferral"] = {"task_id": "initial", "reason": "hbm_reservations_full"}
    write(Path(entry["path"])/"group.json", group)
    supervisor.supervision_tick()
    assert entry["close_requested"] == "hbm_admission_deferred"
    group = release(supervisor, key)
    for worker in group["workers"].values():
        worker.update(status="not_started", reason="hbm_admission_deferred")
    write(Path(entry["path"])/"group.json", group)
    supervisor.supervision_tick()
    for row in (initial, sibling):
        record = supervisor.state["tasks"][row["id"]]
        assert supervisor._poll(row, record) and record["status"] == "pending"
    assert entry["release_verified"] and supervisor.state["tasks"][task["id"]]["status"] == "running"


def test_checkpoint_write_budget_uses_free_space_above_reserve(owner):
    supervisor, _, _ = owner
    supervisor.config.update(use_live_checkpoint_headroom=True, storage_reserve_bytes=24*2**30,
                             max_inflight_checkpoint_bytes=24*2**30)
    for free, expected in ((22, 0), (33, 9), (60, 24)):
        supervisor.resource_cache = {"available_storage_bytes":free*2**30}
        assert supervisor.available_checkpoint_writes() == expected*2**30
