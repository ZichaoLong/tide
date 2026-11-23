"""Read-only terminal observations must not overclaim release or integrity."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pilotlab import audit3
from pilotlab.io import sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path):
    output = tmp_path / "round"
    output.mkdir()
    source = tmp_path / "frozen-source.py"
    source.write_text("# frozen training source\n")
    original = tmp_path / "original-data"
    original.mkdir()
    (original / "train.bin").write_bytes(b"training data")
    write_json(original / "manifest.json", {"source": "fixed"})
    data = tmp_path / "data"
    data.mkdir()
    (data / "selection.idx").write_bytes(b"indices")
    manifest = {"schema_version": "tide-pilot-data-v3", "source_directory": str(original),
        "source_manifest_sha256": sha256(original / "manifest.json"),
        "source_files": {"train.bin": {"sha256": sha256(original / "train.bin"), "size_bytes": 13}},
        "files": {"selection.idx": {"sha256": sha256(data / "selection.idx"), "size_bytes": 7}}}
    write_json(data / "manifest.json", manifest)
    identity = {"source_files": {str(source): sha256(source)}, "data_manifest_sha256": sha256(data / "manifest.json")}
    prepared = {"schema": "pilot-r3-prepared-inputs-v1", "source_identity": identity, "output": str(output), "data": str(data)}
    prepared_path = write_json(tmp_path / "prepared.json", prepared)
    process = {"pid": 12345, "start_ticks": 555, "state": "S", "ppid": 1}
    state = {"schema": "pilot-r3-scheduler-v1", "status": "completed", "tasks": {},
             "controller_identity": process, "all_seen_processes": {"12345:555": process}}
    write_json(output / "scheduler.json", state)
    proc_root, cgroup_root, runtime = (tmp_path / value for value in ("proc", "cgroup", "runtime"))
    for path in (proc_root, cgroup_root, runtime):
        path.mkdir()
    (cgroup_root / "cgroup.procs").touch()
    (proc_root / "locks").write_text("")
    service = {"status": "observed", "unit": "test.service", "properties": {
        "ActiveState": "inactive", "MainPID": "0", "ControlGroup": "/background.slice/test.service"}}
    return prepared_path, prepared, state, service, proc_root, cgroup_root, runtime


def add_task(prepared, state, *, status="completed", phase="a1", failed_capture=False):
    output = Path(prepared["output"])
    task = {"id": "task-one", "phase": phase, "device": "npu:0", "source_identity": prepared["source_identity"]}
    directory = output / "tasks/task-one/attempt-0"
    directory.mkdir(parents=True)
    task_path = write_json(directory.parent / "task.json", task)
    checkpoint = directory / "checkpoints/step-000001.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"complete checkpoint")
    cp = {"path": str(checkpoint), "sha256": sha256(checkpoint), "size_bytes": checkpoint.stat().st_size}
    write_json(directory / "checkpoint.json", {"schema": "pilot-r3-checkpoint-generations-v1", "current": cp, "previous": None})
    result = {"status": status, "source_identity": prepared["source_identity"], "task_sha256": sha256(task_path),
              "checkpoint": cp}
    if failed_capture:
        failure = directory / "failure.pt"
        failure.write_bytes(b"decisive failure payload")
        result["failure_capture"] = {"path": str(failure), "sha256": sha256(failure)}
    result_path = write_json(directory / "result.json", result)
    write_json(directory / "queue.json", {"status": "completed" if status == "completed" else "failed",
                                          "job_id": "r3-job-one", "physical_ids": [3], "return_code": 0})
    process = {"pid": 23456, "start_ticks": 666, "state": "S", "ppid": 12345}
    attempt = {"path": str(directory), "status": status, "process": process,
               "command": ["python", "run_on_free_npu.py", "--job-id", "r3-job-one", "--", "worker"]}
    state["tasks"][task["id"]] = {"status": status, "task_path": str(task_path), "result_path": str(result_path),
        "result_sha256": sha256(result_path), "process": process, "attempts": [attempt]}
    write_json(output / "scheduler.json", state)
    return task, result, checkpoint, directory


def run_fixture(monkeypatch, values):
    prepared_path, prepared, state, service, proc_root, cgroup_root, runtime = values
    monkeypatch.setattr(audit3, "systemd_observation", lambda unit: copy.deepcopy(service))
    return audit3.audit(prepared_path, "test.service", proc_root=proc_root, cgroup_root=cgroup_root, runtime=runtime)


def test_terminal_release_and_hashes_are_observed_without_mutating_inputs(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    add_task(values[1], values[2])
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = run_fixture(monkeypatch, values)
    after = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert report["status"] == "passed", report["findings"]
    assert report["lifecycle"]["status"] == "terminal"
    assert report["lifecycle"]["queue"]["released"] is True
    assert report["lifecycle"]["cgroup"]["status"] == "absent"
    assert before == after


def test_live_controller_with_final_report_is_not_a_released_service(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    service, proc_root, cgroup_root = values[3:6]
    service["properties"].update(ActiveState="active", MainPID="12345")
    group = cgroup_root / "background.slice/test.service"
    group.mkdir(parents=True)
    (group / "cgroup.procs").write_text("12345\n")
    (group / "cgroup.events").write_text("populated 1\n")
    fields = ["S", "1", "1", *(["0"] * 16), "555", "0", "0"]
    (proc_root / "12345").mkdir()
    (proc_root / "12345/stat").write_text("12345 (controller) " + " ".join(fields))
    report = run_fixture(monkeypatch, values)
    assert report["status"] == report["lifecycle"]["status"] == "running"
    assert report["lifecycle"]["processes"]["released"] is False
    assert report["lifecycle"]["cgroup"]["released"] is False


@pytest.mark.parametrize("mounts", [("systemd",), ("unified", "systemd")])
def test_named_hierarchy_reports_service_and_descendant_members(tmp_path, mounts):
    group = "background.slice/test.service"
    service = {"properties": {"MainPID": "12345", "ControlGroup": "/" + group}}
    for mount in mounts:
        path = tmp_path / mount / group
        path.mkdir(parents=True)
        (tmp_path / mount / "cgroup.procs").touch()
        (path / "cgroup.procs").touch()
    active = tmp_path / "systemd" / group
    (active / "cgroup.procs").write_text("12345\n")
    (active / "workers").mkdir()
    (active / "workers/cgroup.procs").write_text("23456\n")
    observed = audit3.cgroup_observation(service, root=tmp_path)
    assert observed["members"] == [12345, 23456]
    assert observed["status"] == "populated" and observed["released"] is False


def test_missing_hierarchy_or_live_main_pid_cannot_certify_release(tmp_path):
    service = {"properties": {"MainPID": "0", "ControlGroup": "/background.slice/test.service"}}
    missing = audit3.cgroup_observation(service, root=tmp_path)
    assert missing["status"] == "unknown" and missing["released"] is None
    (tmp_path / "cgroup.procs").touch()
    service["properties"]["MainPID"] = "12345"
    changing = audit3.cgroup_observation(service, root=tmp_path)
    assert changing["status"] == "changing" and changing["released"] is None


def test_reused_pid_is_not_the_recorded_descendant():
    state = {"controller_identity": {"pid": 5, "start_ticks": 10}, "tasks": {}}
    current = {5: {"pid": 5, "start_ticks": 20, "ppid": 1, "state": "S"}}
    result = audit3.process_observation(state, {"properties": {"MainPID": "0"}}, {"members": []}, current, [])
    assert result["released"] is True and result["live_processes"] == []
    assert result["reused_pids"] == [{"pid": 5, "recorded_start_ticks": 10, "current_start_ticks": 20}]


def test_missing_and_changed_data_are_distinct_findings(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    data = Path(values[1]["data"])
    (data / "selection.idx").unlink()
    (tmp_path / "original-data/train.bin").write_bytes(b"changed data!")
    report = run_fixture(monkeypatch, values)
    assert report["lifecycle"]["status"] == "terminal"
    assert report["status"] == "incomplete"
    assert report["finding_counts"]["missing"] == 1
    assert report["finding_counts"]["mismatch"] == 1


def retirement(output, task, checkpoint, reference):
    item = {"path": str(checkpoint), "sha256": reference["sha256"], "bytes": reference["size_bytes"],
            "task_id": task["id"], "reason": "successful A candidate retired after recorded decision"}
    write_json(output / "checkpoint-retirement.json", {"schema": "pilot-r3-retirement-v1", "files": [item]})
    checkpoint.unlink()


def test_only_successful_new_round_checkpoints_can_use_retirement(tmp_path):
    values = fixture(tmp_path)
    prepared, state = values[1:3]
    task, result, checkpoint, directory = add_task(prepared, state)
    retirement(Path(prepared["output"]), task, checkpoint, result["checkpoint"])
    report = audit3.artifact_observations(state, prepared, Path(prepared["output"]))
    assert [value["status"] for value in report["files"] if value["kind"] == "checkpoint"] == ["retired"]
    state["tasks"][task["id"]]["attempts"][0]["status"] = "failed"
    report = audit3.artifact_observations(state, prepared, Path(prepared["output"]))
    assert [value["status"] for value in report["files"] if value["kind"] == "checkpoint"] == ["invalid"]


def test_formal_final_cannot_be_retired_and_failure_payload_must_remain(tmp_path):
    values = fixture(tmp_path)
    prepared, state = values[1:3]
    task, result, checkpoint, directory = add_task(prepared, state, phase="b", failed_capture=True)
    retirement(Path(prepared["output"]), task, checkpoint, result["checkpoint"])
    (directory / "failure.pt").unlink()
    report = audit3.artifact_observations(state, prepared, Path(prepared["output"]))
    assert [value["status"] for value in report["files"] if value["kind"] == "checkpoint"] == ["invalid"]
    assert [value["status"] for value in report["files"] if value["kind"] == "failure_capture"] == ["missing"]


def test_terminal_queue_event_does_not_prove_lock_release(tmp_path):
    values = fixture(tmp_path)
    prepared, state, runtime = values[1], values[2], values[6]
    add_task(prepared, state)
    lock = runtime / "ascend-job-queue/jobs/r3-job-one.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("pid=23456\n")
    stat = lock.stat()
    import os
    locks = [{"pid": 23456, "device": (os.major(stat.st_dev), os.minor(stat.st_dev)), "inode": stat.st_ino}]
    result = audit3.queue_observation(state, {"released": True, "live_processes": []}, runtime=runtime, locks=locks, lock_error=None)
    assert result["status"] == "active" and result["released"] is False
    assert lock.read_text() == "pid=23456\n"


def test_aborted_wrapper_without_registration_has_no_observed_allocation(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    _, _, _, directory = add_task(values[1], values[2], status="failed")
    (directory / "queue.json").unlink()
    report = run_fixture(monkeypatch, values)
    queue = report["lifecycle"]["queue"]
    assert queue["released"] is True
    assert queue["attempts"][0]["status"] == "not_allocated_observed"
    assert report["status"] == "incomplete"
    assert any(value.get("scope") == "queue" and value["status"] == "missing" for value in report["findings"])


def test_unavailable_observations_are_unknown_not_released(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    values[3].clear()
    values[3].update(status="unknown", error="user manager unavailable")
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "incomplete" and report["lifecycle"]["status"] == "unknown"
    assert report["lifecycle"]["cgroup"]["released"] is None
    assert report["finding_counts"]["unknown"] >= 2


def test_malformed_records_are_reported_instead_of_aborting_audit(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    state = values[2]
    state["tasks"] = {"broken": {"attempts": [None]}}
    write_json(Path(values[1]["output"]) / "scheduler.json", state)
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "incomplete"
    assert report["finding_counts"]["invalid"] >= 1


def test_launch_draft_binds_prepared_and_launcher_bytes(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    launcher = tmp_path / "launch.sh"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    write_json(tmp_path / "launch-draft.json", {"prepared_sha256": sha256(values[0]),
        "launcher": str(launcher), "launcher_sha256": sha256(launcher)})
    values[1]["unreviewed_change"] = True
    write_json(values[0], values[1])
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "incomplete"
    assert report["launch_binding"][0]["status"] == "mismatch"


def shared_fixture(tmp_path, *, backend="npu"):
    values = fixture(tmp_path)
    prepared, state = values[1:3]
    task, result, checkpoint, attempt_dir = add_task(prepared, state)
    task["device"] = "npu:0" if backend == "npu" else "cpu"
    record = state["tasks"][task["id"]]
    write_json(Path(record["task_path"]), task)
    result["task_sha256"] = sha256(record["task_path"])
    write_json(Path(record["result_path"]), result)
    record["result_sha256"] = sha256(record["result_path"])
    (attempt_dir / "queue.json").unlink()
    output = Path(prepared["output"])
    directory = output / "groups/group-a"
    command = ["python", "--task", record["task_path"], "--output", str(attempt_dir), "--device", task["device"]]
    owner = {"pid": 34567, "start_ticks": 777, "state": "S", "ppid": 12345}
    worker_process = record["process"]
    record.update(shared_group="group-a", device_seconds=0, process=owner)
    record["attempts"][0].update(shared_group="group-a", group_path=str(directory), command=command,
                                 device_seconds=0, process=owner, worker_process=worker_process)
    manifest = {"schema": "pilot-r3-shared-group-v1", "job_id": "group-job", "backend": backend,
                "workers": [{"id": task["id"], "output": str(attempt_dir), "argv": command}]}
    path = write_json(directory / "manifest.json", manifest)
    seconds = 10 if backend == "npu" else 0
    group = {"schema": "pilot-r3-shared-group-state-v1", "job_id": "group-job", "backend": backend,
        "manifest_sha256": sha256(path), "status": "completed", "owner": owner,
        "owner_cgroup": "0::/background.slice/test.service", "all_seen_processes": {"23456:666": worker_process},
        "allocation_acquired_unix": 100 if backend == "npu" else None,
        "allocation_released_unix": 110 if backend == "npu" else None, "physical_device_seconds": seconds,
        "physical_id": 3 if backend == "npu" else None, "remaining_descendants": [],
        "workers": {task["id"]: {"status": "completed", "return_code": 0, "pid": 23456, "start_ticks": 666,
            "process": worker_process, "output": str(attempt_dir), "started_unix": 101, "finished_unix": 109,
            "result_path": record["result_path"], "result_sha256": record["result_sha256"]}}}
    write_json(directory / "group.json", group)
    if backend == "npu":
        write_json(directory / "queue.json", {"status": "completed", "job_id": "group-job", "return_code": 0,
                                              "physical_ids": [3]})
    state["shared_groups"] = {"group-a": {"path": str(directory), "manifest_path": str(path), "job_id": "group-job",
        "manifest_sha256": sha256(path), "backend": backend, "task_ids": [task["id"]], "device_seconds": seconds,
        "preparation": True, "process": owner}}
    budget = tmp_path / "budget.json"
    write_json(budget, {"device_seconds": seconds + 7, "segments": {
        str(output): {"device_seconds": seconds, "preparation_device_seconds": seconds},
        "another-segment": {"device_seconds": 7, "preparation_device_seconds": 0}}})
    state.update(config={"budget_ledger": str(budget)}, device_seconds=seconds)
    write_json(output / "scheduler.json", state)
    return values, group, directory


@pytest.mark.parametrize("backend", ["cpu", "npu"])
def test_shared_terminal_audit_verifies_one_group_charge_and_no_worker_queue_file(tmp_path, monkeypatch, backend):
    values, group, directory = shared_fixture(tmp_path, backend=backend)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "passed", report["findings"]
    assert report["physical_accounting"]["status"] == "passed"
    assert report["physical_accounting"]["physical_device_seconds"] == (10 if backend == "npu" else 0)
    assert report["lifecycle"]["processes"]["recorded_identity_count"] == 3
    assert len(report["lifecycle"]["queue"]["attempts"]) == (1 if backend == "npu" else 0)
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("debit", [9., 10., 15.])
def test_supplementary_accounting_retains_overcharges_but_never_accepts_undercharges(tmp_path, debit):
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    output = Path(prepared["output"])
    state["shared_groups"]["group-a"]["device_seconds"] = state["device_seconds"] = debit
    ledger_path = Path(state["config"]["budget_ledger"])
    ledger = json.loads(ledger_path.read_text())
    ledger["device_seconds"] = debit + 7
    ledger["segments"][str(output)].update(device_seconds=debit, preparation_device_seconds=debit)
    write_json(ledger_path, ledger)
    shared = {"groups": {"group-a": {"group": group, "owner_alive": False}}, "scheduler_live": False}
    strict = audit3.shared_accounting_observation(state, prepared, output, shared)
    review = audit3.shared_accounting_observation(state, prepared, output, shared, allow_conservative_debits=True)
    assert strict["status"] == ("passed" if debit == 10 else "incomplete")
    assert review["status"] == ("passed" if debit >= 10 else "incomplete")
    assert review["conservative_excess_seconds"] == max(0., debit-10)
    assert json.loads(ledger_path.read_text()) == ledger


@pytest.mark.parametrize("damage", [None,"actually_counted","unmarked","live","source_changed"])
def test_stopped_worker_estimate_requires_proof_of_ledger_exclusion(tmp_path, damage):
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    record=state["tasks"]["task-one"]
    state["status"]="a_blocked_needs_repair"
    record.update(status="failed", reason="cleanup_evidence_error:stale cache", device_seconds=12.)
    record["attempts"][0].update(device_seconds=12., accounting_estimated=True,
        accounting_reason="unreadable_current_attempt_accounting")
    cgroup={"path":"/background.slice/test.service","released":True}
    table={}
    if damage=="actually_counted":state["device_seconds"]+=12
    elif damage=="unmarked":record["attempts"][0].pop("accounting_estimated")
    elif damage=="live":table[state["controller_identity"]["pid"]]=state["controller_identity"]
    elif damage=="source_changed":
        path=directory/"manifest.json"; v=json.loads(path.read_text());v["unexpected"]=True;write_json(path,v)
    strict=audit3.shared_group_observations(state,Path(prepared["output"]),cgroup,table)
    assert any(v["status"]=="mismatch" for v in strict["checks"])
    reviewed=audit3.shared_group_observations(state,Path(prepared["output"]),cgroup,table,
        allow_inert_worker_estimates=True)
    assert any(v["status"]=="mismatch" for v in reviewed["checks"]) == (damage is not None)
    assert record["device_seconds"]==12.  # Evidence and charges are never rewritten.


@pytest.mark.parametrize("damage", [None, "command_chain", "worker_cleanup", "close_ack"])
def test_dynamic_terminal_audit_requires_append_bindings_cleanup_and_close(tmp_path, monkeypatch, damage):
    from pilotlab import shared_npu3
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    appended = manifest.pop("workers")
    manifest.update(workers=[], dynamic_admission={"inbox_dir": str(directory/"inbox")})
    write_json(path, manifest)
    group["manifest_sha256"] = state["shared_groups"]["group-a"]["manifest_sha256"] = sha256(path)
    group["workers"]["task-one"].update(descendant_cleanup_verified=damage != "worker_cleanup",
        remaining_descendants=[], worker_spec_sha256="spec")
    def chain(*args):
        if damage == "command_chain":
            raise ValueError("admission command digest mismatch")
        return {"published_workers": appended, "acknowledged_workers": appended,
                "commands": [], "acknowledged_closed": damage != "close_ack"}
    monkeypatch.setattr(shared_npu3, "read_admission_chain", chain, raising=False)
    monkeypatch.setattr(shared_npu3, "worker_spec_digest", lambda _: "spec", raising=False)
    write_json(directory/"group.json", group)
    write_json(Path(prepared["output"])/"scheduler.json", state)
    report = run_fixture(monkeypatch, values)
    assert report["status"] == ("passed" if damage is None else "incomplete"), report["findings"]


@pytest.mark.parametrize("damage", ["manifest", "segment_charge", "worker_double_charge", "owner_cgroup", "worker_checkpoint_escape"])
def test_shared_audit_rejects_bad_bindings_accounting_and_checkpoint_ownership(tmp_path, monkeypatch, damage):
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    record = state["tasks"]["task-one"]
    if damage == "manifest":
        path = directory / "manifest.json"
        value = json.loads(path.read_text())
        value["changed"] = True
        write_json(path, value)
    elif damage == "segment_charge":
        path = Path(state["config"]["budget_ledger"])
        ledger = json.loads(path.read_text())
        ledger["segments"][prepared["output"]]["device_seconds"] = 20
        ledger["device_seconds"] = 27
        write_json(path, ledger)
    elif damage == "worker_double_charge":
        record["device_seconds"] = 10
        write_json(Path(prepared["output"]) / "scheduler.json", state)
    elif damage == "owner_cgroup":
        group["owner_cgroup"] = "0::/unrelated.service"
        write_json(directory / "group.json", group)
    else:
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"wrong checkpoint owner")
        result = json.loads(Path(record["result_path"]).read_text())
        result["checkpoint"] = {"path": str(outside), "sha256": sha256(outside), "size_bytes": outside.stat().st_size}
        write_json(Path(record["result_path"]), result)
        record["result_sha256"] = sha256(record["result_path"])
        group["workers"]["task-one"]["result_sha256"] = record["result_sha256"]
        write_json(directory / "group.json", group)
        write_json(Path(prepared["output"]) / "scheduler.json", state)
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "incomplete"
    assert any(value["status"] in {"mismatch", "invalid"} for value in report["findings"])


def test_group_only_recorded_descendant_prevents_false_terminal_release(tmp_path, monkeypatch):
    values, group, directory = shared_fixture(tmp_path)
    proc_root = values[4]
    child = proc_root / "23456"
    child.mkdir()
    fields = ["S", "1", "1", *(["0"] * 16), "666", "0", "0"]
    (child / "stat").write_text("23456 (escaped-worker) " + " ".join(fields))
    report = run_fixture(monkeypatch, values)
    assert report["lifecycle"]["status"] == "running"
    assert report["lifecycle"]["processes"]["live_processes"][0]["pid"] == 23456
    assert report["lifecycle"]["queue"]["released"] is not True


def test_terminal_cpu_group_requires_terminal_group_evidence(tmp_path, monkeypatch):
    values, group, directory = shared_fixture(tmp_path, backend="cpu")
    group["status"] = "running"
    write_json(directory / "group.json", group)
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "incomplete"
    assert any(row.get("reason") == "shared_group_not_terminal" for row in report["findings"])


@pytest.mark.parametrize("worker_status", ["pending", "starting", "running", "exited_waiting_cleanup"])
def test_live_group_result_publication_is_pending_and_does_not_hide_other_damage(tmp_path, monkeypatch, worker_status):
    values, group, directory = shared_fixture(tmp_path, backend="cpu")
    prepared, state, service, proc_root, cgroup_root = values[1:6]
    state["status"] = "running"
    record = state["tasks"]["task-one"]
    record["status"] = record["attempts"][0]["status"] = "running"
    if worker_status == "pending":
        record["attempts"][0]["status"] = "admitting"
    group["status"] = "running"
    worker = group["workers"]["task-one"]
    worker.update(status=worker_status, return_code=None, finished_unix=None)
    worker.pop("result_sha256")
    write_json(directory / "group.json", group)
    write_json(Path(prepared["output"]) / "scheduler.json", state)
    service["properties"].update(ActiveState="active", MainPID="12345")
    active_group = cgroup_root / "background.slice/test.service"
    active_group.mkdir(parents=True)
    (active_group / "cgroup.procs").write_text("12345\n34567\n")
    for pid, start in ((12345, 555), (34567, 777)):
        folder = proc_root / str(pid)
        folder.mkdir()
        fields = ["S", "1", "1", *(["0"] * 16), str(start), "0", "0"]
        (folder / "stat").write_text(f"{pid} (active owner) " + " ".join(fields))
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "running"
    result = next(row for row in report["artifacts"]["files"] if row["kind"] == "result")
    assert result["status"] == "pending"
    assert result["reason"] == "shared_worker_result_publication_pending"
    assert not report["findings"], report["findings"]
    observed = json.loads(Path(record["result_path"]).read_text())
    observed["task_sha256"] = "wrong"
    write_json(Path(record["result_path"]), observed)
    report = run_fixture(monkeypatch, values)
    result = next(row for row in report["artifacts"]["files"] if row["kind"] == "result")
    assert result["status"] == "mismatch"


def test_live_group_budget_refresh_is_pending_but_terminal_debit_must_match(tmp_path):
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    output = Path(prepared["output"])
    state["status"] = "running"
    state["shared_groups"]["group-a"]["status"] = "running"
    group.update(status="running", physical_device_seconds=11, allocation_released_unix=None)
    write_json(directory / "group.json", group)
    ledger_path = Path(state["config"]["budget_ledger"])
    ledger = json.loads(ledger_path.read_text())
    ledger["segments"][prepared["output"]].update(device_seconds=12, preparation_device_seconds=12)
    ledger["device_seconds"] = 19
    write_json(ledger_path, ledger)
    table = {identity["pid"]: identity for identity in (state["controller_identity"], group["owner"])}
    shared = audit3.shared_group_observations(state, output, {"path": "/background.slice/test.service"}, table)
    accounting = audit3.shared_accounting_observation(state, prepared, output, shared)
    assert accounting["status"] == "pending", accounting
    assert accounting["physical_device_seconds"] == 11
    assert accounting["scheduler_device_seconds"] == 10
    assert all(row["status"] in {"passed", "pending"} for row in accounting["checks"])
    state["status"] = state["shared_groups"]["group-a"]["status"] = "completed"
    group.update(status="completed", allocation_released_unix=111)
    write_json(directory / "group.json", group)
    shared = audit3.shared_group_observations(state, output, {"path": "/background.slice/test.service"}, {})
    accounting = audit3.shared_accounting_observation(state, prepared, output, shared)
    assert accounting["status"] == "incomplete"
    assert any(row["status"] == "mismatch" and row["reason"] == "scheduler group debit:group-a" for row in accounting["checks"])


def test_shared_not_started_worker_without_result_is_distinct_from_missing_evidence(tmp_path, monkeypatch):
    values, group, directory = shared_fixture(tmp_path)
    prepared, state = values[1:3]
    record = state["tasks"]["task-one"]
    attempt = Path(record["attempts"][0]["path"])
    for path in [attempt / "result.json", attempt / "checkpoint.json", *list((attempt / "checkpoints").glob("*.pt"))]:
        path.unlink()
    record["status"] = record["attempts"][0]["status"] = "interrupted"
    group.update(status="failed", allocation_acquired_unix=None, allocation_released_unix=None,
                 physical_device_seconds=0, physical_id=None)
    group["workers"]["task-one"] = {"status": "not_started", "pid": None, "started_unix": None,
                                        "output": str(attempt), "return_code": None}
    write_json(directory / "group.json", group)
    write_json(directory / "queue.json", {"status": "failed", "job_id": "group-job", "physical_ids": []})
    state.update(status="a_blocked_needs_repair", device_seconds=0)
    state["shared_groups"]["group-a"]["device_seconds"] = 0
    write_json(Path(prepared["output"]) / "scheduler.json", state)
    write_json(Path(state["config"]["budget_ledger"]), {"device_seconds": 0, "segments": {
        prepared["output"]: {"device_seconds": 0, "preparation_device_seconds": 0}}})
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "passed", report["findings"]
    assert any(row["kind"] == "result" and row["status"] == "not_started" for row in report["artifacts"]["files"])


def test_external_selected_evidence_keeps_its_own_identity(tmp_path, monkeypatch):
    values = fixture(tmp_path)
    old = tmp_path / "explicit-prior-selection"
    task = {"model_id": "old-selected", "source_identity": {"source": "old-frozen"}}
    task_path = write_json(old / "task.json", task)
    result_path = write_json(old / "result.json", {"source_identity": task["source_identity"], "task_sha256": sha256(task_path)})
    write_json(Path(values[1]["output"]) / "selected.json", {"old-selected": {
        "confirmation": [{"task_path": str(task_path), "result_path": str(result_path)}]}})
    report = run_fixture(monkeypatch, values)
    assert report["status"] == "passed", report["findings"]
    assert report["selected_references"]["external_reference_count"] == 1
