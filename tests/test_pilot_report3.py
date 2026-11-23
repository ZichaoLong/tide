"""Read-only report witnesses for exposure, real concurrency and fail-closed views."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from pilotlab import inspect3, report3


IDENTITY = {"source": "frozen-source", "data": "frozen-data"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n")
    return str(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_relay_report_reuses_verified_bytes_only_within_one_snapshot(tmp_path):
    root = tmp_path/"new-round"
    original = write(tmp_path/"released-round/review.json", {"status":"passed"})
    ref = {"path":original, "sha256":sha(original)}
    snapshot = report3.Snapshot(root)
    assert snapshot.read_import_reference(ref) == {"status":"passed"}
    with pytest.raises(ValueError, match="reference_changed"):
        snapshot.read_import_reference(dict(ref, sha256="wrong"))
    write(original, {"status":"changed"})
    assert snapshot.read_import_reference(ref) == {"status":"passed"}
    with pytest.raises(ValueError, match="reference_changed"):
        report3.Snapshot(root).read_import_reference(ref)


@pytest.fixture
def round_state(tmp_path):
    root = tmp_path/"round"
    root.mkdir()
    ledger = write(tmp_path/"budget.json", {"device_seconds": 7, "max_device_seconds": 7200*3600,
                                           "deadline_unix": 1000, "started_unix": 50})
    state = {"schema": "pilot-r3-scheduler-v1", "config": {"budget_ledger": ledger},
             "status": "running", "stage": "a3", "started_unix": 90, "updated_unix": 120,
             "tasks": {}, "peak_actual_npu_concurrency": 0, "device_seconds": 7,
             "all_seen_processes": {}}
    write(root/"scheduler.json", state)
    return root, state


def queue_line(status, stamp):
    return json.dumps({"event": "ascend_queue", "status": status,
                       "timestamp": datetime.fromtimestamp(stamp, timezone.utc).isoformat()}, sort_keys=True)+"\n"


def add_task(root, state, task_id, phase, target, segments, *, model_id="tide-one", seed=11,
             family="tide", queue_times=None, status="completed", health=True):
    task = {"id": task_id, "model_id": model_id, "phase": phase, "target_tokens": target,
            "model": {"family": family}, "seed": seed, "source_identity": IDENTITY,
            "recipe_id": "r0", "init_recipe": "default", "batch_size": 4, "lr": .001,
            "sequence_length": 128, "cpu_threads": 6, "device": "npu:0" if queue_times else "cpu"}
    if phase.startswith("b"):
        kind = {"b": "formal", "b_repeat": "repeat", "b_confirmation": "confirmation"}[phase]
        task["gate_file"] = str(root/("stage-b-gate.json" if kind == "formal" else f"stage-b-{kind}-gate.json"))
    task_path = write(root/"tasks"/task_id/"task.json", task)
    attempts = []
    results = []
    for i, (start, end) in enumerate(segments):
        final = i == len(segments)-1
        worker_status = status if final else "failed"
        directory = root/"tasks"/task_id/f"attempt-{i}"
        result = {"schema": "pilot-r3-worker-result-v1", "id": task_id, "model_id": model_id,
            "phase": phase, "status": worker_status, "source_identity": IDENTITY, "task_sha256": sha(task_path),
            "training": {"tokens": end, "from_tokens": start, "segment_tokens": end-start,
                         "seconds": 1., "tokens_per_second": end-start},
            "health": {"eligible": health, "blocking_reasons": [] if health else ["numerical_failure"], "sample_count": 2},
            "recovery": {"checkpoint_load": {"status": "exact", "scope": "fresh host owners"},
                         "numeric_trajectory": {"status": "not_tested", "atol": 1e-5, "rtol": 1e-4}},
            "validation_initial": {"status": "completed", "nll": 5.},
            "validation_final": {"status": "completed", "nll": 4.},
            "initialization": {"resolved_recipe": "upstream-default-lm-boundary-v1"}}
        if start:
            result["resume"] = {"from_tokens": start, "checkpoint_load": {"status": "exact"},
                                "source_task_id": "previous", "state_reused": True}
        if phase == "b_confirmation":
            result["confirmation_final"] = {"status": "completed", "nll": 4.2}
        result_path = write(directory/"result.json", result)
        attempts.append({"path": str(directory), "status": worker_status,
                         "return_code": 0 if worker_status == "completed" else 1,
                         "started_unix": 99, "device_seconds": 0})
        if queue_times:
            begin, finish = queue_times[i]
            (directory/"stdout.log").write_text(queue_line("running", begin)+queue_line("completed", finish))
            write(directory/"queue.json", {"status": "completed", "return_code": 0})
        results.append((result_path, result))
    record = {"model_id": model_id, "stage": phase, "status": status, "task_path": task_path,
              "attempts": attempts, "result_path": results[-1][0], "result_sha256": sha(results[-1][0]),
              "device_seconds": 0, "device": task["device"]}
    state["tasks"][task_id] = record
    write(root/"scheduler.json", state)
    return task, record, results


def gate(root, kind, task_ids):
    filename = "stage-b-gate.json" if kind == "formal" else f"stage-b-{kind}-gate.json"
    manifest = root/"model-manifest.json"
    if not manifest.exists():
        write(manifest, {"source_identity": IDENTITY, "subjects": [{"id": "tide-one", "group": "fc"}]})
    audit = root/"stage-a-report.json"
    write(audit, {"status": "passed", "source_identity": IDENTITY, "blocking_reasons": [],
                  "eligible_subject_count": 216})
    write(root/filename, {"status": "passed", "gate_kind": kind, "source_identity": IDENTITY,
        "manifest_path": str(manifest), "manifest_sha256": sha(manifest),
        "report_path": str(audit), "report_sha256": sha(audit), "allowed_task_ids": task_ids,
        "task_sha256": {tid: sha(root/"tasks"/tid/"task.json") for tid in task_ids}})


def test_actual_intervals_catch_short_jobs_and_resume_does_not_duplicate_exposure(round_state):
    root, state = round_state
    add_task(root, state, "one", "a1", 10, [(0, 10)], queue_times=[(100, 100.75)])
    add_task(root, state, "two", "a2", 25, [(10, 18), (15, 25)], queue_times=[(100.3, 101.05), (101.2, 101.8)])
    _, record, _ = add_task(root, state, "three", "a3", 40, [(25, 40)], queue_times=[(101.8, 102.6)])
    write(root/"selected.json", {"tide-one": {"recipe_id": "r0", "init_recipe": "default",
          "confirmation": [{"task_path": record["task_path"], "result_path": record["result_path"]}]}})
    report = report3.build_report(root, now=130)
    assert report["exposure"]["unique_loss_tokens_by_category"]["tuning"] == 40
    assert report["exposure"]["physically_computed_loss_tokens_by_category"]["tuning"] == 43
    assert report["concurrency"]["peak"] == 2
    assert report["concurrency"]["scheduler_polled_peak"] == 0
    assert report["concurrency"]["device_seconds"] == pytest.approx(2.9)
    assert report["state_loading"]["actual_training_continuations"] == 2
    assert report["trajectory_reproduction"]["status_counts"] == {"not_tested": 3}
    assert report["selected_recipes"][0]["batch_size"] == 4
    assert report["selected_recipes"][0]["cpu_threads"] == 6
    assert report["selected_recipes"][0]["lr"] == .001
    assert not report["observation_errors"]


def test_formal_counts_distinct_models_and_confirmation_adds_zero_learning(round_state):
    root, state = round_state
    tokens = report3.FORMAL_TOKENS
    add_task(root, state, "formal", "b", tokens, [(0, tokens)], seed=41)
    add_task(root, state, "repeat", "b_repeat", tokens, [(0, tokens)], seed=42)
    add_task(root, state, "confirmed", "b_confirmation", tokens, [(tokens, tokens)], seed=41)
    gate(root, "formal", ["formal"])
    gate(root, "repeat", ["repeat"])
    gate(root, "confirmation", ["confirmed"])
    report = report3.build_report(root, now=130)
    assert report["exposure"]["unique_loss_tokens_by_category"]["formal"] == tokens*2
    assert report["exposure"]["completed_subject_models"] == 1
    assert report["exposure"]["completed_healthy_subject_models"] == 1
    assert report["state_loading"]["actual_training_continuations"] == 0
    assert report["learning"]["confirmation_evaluations"][0]["confirmation_nll"] == 4.2
    assert all(report["gates"][kind]["binding_verified"] for kind in ("formal", "repeat", "confirmation"))


def test_known_trajectory_failure_is_not_relabelled_as_state_loading_failure(round_state):
    root, state = round_state
    _, record, results = add_task(root, state, "one", "a1", 10, [(0, 10)])
    path, result = results[-1]
    result["recovery"]["numeric_trajectory"] = {"status": "outside_tolerance", "atol": 1e-5, "rtol": 1e-4, "max_error": .002}
    write(path, result)
    record["result_sha256"] = sha(path)
    write(root/"scheduler.json", state)
    report = report3.build_report(root)
    assert report["execution"]["verified_completed_tasks"] == 1
    assert report["state_loading"]["status_counts"] == {"exact": 1}
    assert report["trajectory_reproduction"]["status_counts"] == {"outside_tolerance": 1}
    assert report["tasks"][0]["numeric_trajectory"]["atol"] == 1e-5


@pytest.mark.parametrize("damage", ["result_hash", "source", "tokens", "missing_result", "missing_gate", "changed_audit"])
def test_bad_evidence_never_establishes_campaign_completion(round_state, damage):
    root, state = round_state
    tokens = report3.FORMAL_TOKENS
    _, record, results = add_task(root, state, "formal", "b", tokens, [(0, tokens)])
    gate(root, "formal", ["formal"])
    path, result = results[-1]
    if damage == "result_hash":
        result["extra"] = "changed"
        write(path, result)
    elif damage == "source":
        result["source_identity"] = {"wrong": True}
        write(path, result)
        record["result_sha256"] = sha(path)
    elif damage == "tokens":
        result["training"]["tokens"] -= 1
        result["training"]["segment_tokens"] -= 1
        write(path, result)
        record["result_sha256"] = sha(path)
    elif damage == "missing_result":
        Path(path).unlink()
    elif damage == "missing_gate":
        (root/"stage-b-gate.json").unlink()
    else:
        write(root/"stage-a-report.json", {"status": "blocked"})
    state["status"] = "completed"
    write(root/"scheduler.json", state)
    report = report3.build_report(root)
    assert report["status"] != "completed"
    assert report["exposure"]["completed_subject_models"] == 0
    assert report["execution"]["verified_completed_tasks"] == 0


def test_recorded_block_stays_blocked_and_health_remains_separate(round_state):
    root, state = round_state
    add_task(root, state, "failed", "a1", 10, [(0, 10)], health=False, status="failed")
    state.update(status="a_blocked_needs_repair", stage_b_gate="closed", blocking_reason="known_nonfinite")
    write(root/"scheduler.json", state)
    report = report3.build_report(root)
    assert report["status"] == "a_blocked_needs_repair"
    assert report["blocking_reason"] == "known_nonfinite"
    assert report["numerical_health"]["eligibility_counts"] == {"not_qualified": 1}
    assert report["exposure"]["unique_loss_tokens_by_category"]["tuning"] == 10
    assert report["execution"]["verified_completed_tasks"] == 0


def test_missing_release_does_not_invent_device_time_or_concurrency(round_state):
    root, state = round_state
    _, record, _ = add_task(root, state, "one", "a1", 10, [(0, 10)], queue_times=[(100, 101)])
    directory = Path(record["attempts"][0]["path"])
    (directory/"stdout.log").write_text(queue_line("running", 100))
    report = report3.build_report(root, now=300)
    assert not report["concurrency"]["accounting_complete"]
    assert report["concurrency"]["device_seconds"] == 0
    assert report["concurrency"]["ledger_device_seconds"] == 7
    assert report["performance"]["aggregate_npu_loss_tokens_per_allocation_window_second"] is None


def test_only_verified_live_allocation_is_extended_to_observation_time(round_state, monkeypatch):
    root, state = round_state
    _, record, _ = add_task(root, state, "one", "a1", 10, [(0, 3)], queue_times=[(100, 101)], status="running")
    directory = Path(record["attempts"][0]["path"])
    (directory/"stdout.log").write_text(queue_line("running", 100))
    write(directory/"queue.json", {"status": "running"})
    monkeypatch.setattr(report3, "process_observation", lambda identity: {"alive": True})
    report = report3.build_report(root, now=105)
    assert report["concurrency"]["verified_live_open_allocations"] == 1
    assert report["concurrency"]["device_seconds"] == 5
    assert report["concurrency"]["peak"] == 1


def test_report_and_view_are_readonly_and_explicit_output_cannot_overwrite(round_state, capsys):
    root, state = round_state
    add_task(root, state, "one", "a1", 10, [(0, 10)])
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert report3.main(["--round", str(root)]) == 0
    json.loads(capsys.readouterr().out)
    assert inspect3.main(["--round", str(root)]) == 0
    assert "Read-only" in capsys.readouterr().out
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert before == after
    exported = root.parent/"explicit-observation.json"
    assert report3.main(["--round", str(root), "--output", str(exported)]) == 0
    assert exported.is_file()
    with pytest.raises(FileExistsError):
        report3.main(["--round", str(root), "--output", str(root/"scheduler.json")])
    assert before[root/"scheduler.json"] == ((root/"scheduler.json").read_bytes(), (root/"scheduler.json").stat().st_mtime_ns)


def test_view_imports_do_not_import_torch_workers_or_scheduler():
    code = ("import sys; from pilotlab import report3, inspect3; "
            "assert 'torch' not in sys.modules; assert 'pilotlab.worker3' not in sys.modules; "
            "assert 'pilotlab.scheduler3' not in sys.modules; assert 'pilotlab.round3' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_scheduler_is_unavailable_without_creating_round(tmp_path):
    path = tmp_path/"absent"
    report = report3.build_report(path)
    assert report["status"] == "unavailable"
    assert report["observation_errors"]
    assert not path.exists()


def add_shared_group(root, state, group_id, backend, workers, *, allocation=(100, 110), physical_id=2):
    directory = root / "groups" / group_id
    declared, observed = [], {}
    for index, (task_id, tokens, begin, finish) in enumerate(workers):
        task, record, results = add_task(root, state, task_id, "a1", tokens, [(0, tokens)], model_id=task_id)
        task["device"] = "npu:0" if backend == "npu" else "cpu"
        write(record["task_path"], task)
        result_path, result = results[0]
        result.update(task_sha256=sha(record["task_path"]), started_unix=begin, finished_unix=finish)
        write(result_path, result)
        record.update(result_sha256=sha(result_path), shared_group=group_id, device_seconds=0)
        attempt = record["attempts"][0]
        command = ["python", "--task", record["task_path"], "--output", attempt["path"], "--device", task["device"]]
        attempt.update(shared_group=group_id, group_path=str(directory), command=command)
        declared.append({"id": task_id, "output": attempt["path"], "argv": command})
        observed[task_id] = {"status": "completed", "return_code": 0, "output": attempt["path"],
            "result_path": result_path, "result_sha256": sha(result_path), "started_unix": begin, "finished_unix": finish,
            "pid": 900000 + index, "start_ticks": 17 + index}
        # These are worker-only logs: there must be no queue event requirement.
        (Path(attempt["path"]) / "stdout.log").write_text("ordinary worker log\n")
    manifest = {"schema": "pilot-r3-shared-group-v1", "job_id": group_id, "backend": backend, "workers": declared}
    manifest_path = write(directory / "manifest.json", manifest)
    group = {"schema": "pilot-r3-shared-group-state-v1", "job_id": group_id, "backend": backend,
        "manifest_sha256": sha(manifest_path), "status": "completed", "workers": observed,
        "physical_id": physical_id if backend == "npu" else None,
        "allocation_acquired_unix": allocation[0] if backend == "npu" else None,
        "allocation_released_unix": allocation[1] if backend == "npu" else None,
        "physical_device_seconds": allocation[1]-allocation[0] if backend == "npu" else 0,
        "remaining_descendants": []}
    write(directory / "group.json", group)
    state.setdefault("shared_groups", {})[group_id] = {"path": str(directory), "job_id": group_id,
        "manifest_path": manifest_path, "manifest_sha256": sha(manifest_path), "backend": backend,
        "task_ids": list(observed), "device_seconds": group["physical_device_seconds"]}
    state["device_seconds"] = sum(value["device_seconds"] for value in state["shared_groups"].values())
    write(root / "scheduler.json", state)
    return group


def test_shared_cards_and_cpu_workers_have_separate_concurrency_and_throughput(round_state):
    root, state = round_state
    add_shared_group(root, state, "npu-group", "npu", [("n1", 10, 101, 109), ("n2", 20, 102, 108)])
    add_shared_group(root, state, "cpu-group", "cpu", [("c1", 30, 104, 112), ("c2", 40, 105, 111)])
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report = report3.build_report(root, now=120)
    assert not report["observation_errors"], report["observation_errors"]
    assert report["execution"]["verified_completed_tasks"] == 4
    assert report["concurrency"]["peak"] == 1
    assert report["concurrency"]["device_seconds"] == 10
    assert report["concurrency"]["logical_workers"]["peak"] == 4
    assert report["concurrency"]["npu_workers"]["peak"] == 2
    assert report["concurrency"]["cpu_workers"]["peak"] == 2
    assert report["performance"]["cpu_loss_tokens"] == 70
    assert report["performance"]["cpu_training_loss_tokens_per_second"] == 35
    assert report["performance"]["aggregate_cpu_loss_tokens_per_worker_window_second"] == 8.75
    assert report["performance"]["aggregate_npu_loss_tokens_per_allocation_window_second"] == 3
    assert "CPU_worker_peak=2" in inspect3.render(report)
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_report_surfaces_external_contention_without_claiming_exclusive_timing(round_state):
    root, state = round_state
    group = add_shared_group(root, state, "g", "npu", [("n1", 10, 101, 109)])
    group.update(external_contention_observed=True, peak_external_processes=2,
                 device_observation_log=str(root / "groups/g/device-observations.jsonl"))
    write(root / "groups/g/group.json", group)
    report = report3.build_report(root, now=120)
    assert report["performance"]["external_contention_observed"]
    assert "do not establish exclusivity" in report["performance"]["contention_scope"]
    observed = report["resources"]["shared_groups"][0]
    assert observed["external_contention_observed"] and observed["peak_external_processes"] == 2
    assert observed["device_observation_log"] == group["device_observation_log"]


def test_audited_import_reuses_result_without_inventing_new_device_time(round_state):
    import shutil
    root, state = round_state
    task, record, results = add_task(root, state, "old-task", "a1", 100, [(0, 100)])
    old = root.parent / "old-tasks"
    shutil.copytree(root / "tasks", old)
    old_task = old / "old-task/task.json"
    old_result = old / "old-task/attempt-0/result.json"
    # Retain the byte-identical local request, and only reference old results.
    record["prior_attempts"] = [{**record["attempts"][0], "path": str(old_result.parent)}]
    record.update(attempts=[], result_path=str(old_result), result_sha256=sha(old_result), imported_completion=True)
    def reference(path):
        return {"path": str(path), "sha256": sha(path)}
    review_path = write(root/"reconciliation.json", {"status": "supplemental_review_completed"})
    imported = {"schema": "pilot-r3-resume-import-v1", "reconciliation": reference(Path(review_path)),
        "tasks": {"old-task": {"task": reference(old_task), "result": reference(old_result),
            "attempt_path": str(old_result.parent), "disposition": "completed", "worker_return_code": 0}}}
    import_path = write(root/"import.json", imported)
    state.setdefault("config", {})["resume_import"] = reference(Path(import_path))
    write(root/"scheduler.json", state)
    report = report3.build_report(root, now=120)
    assert not report["observation_errors"], report["observation_errors"]
    assert report["execution"]["verified_completed_tasks"] == 1
    assert report["exposure"]["unique_loss_tokens_by_category"]["tuning"] == 100
    assert report["concurrency"]["device_seconds"] == 0
    assert report["performance"]["aggregate_npu_loss_tokens_per_allocation_window_second"] is None


@pytest.mark.parametrize("cleanup_verified", [True, False])
def test_dynamic_worker_completion_is_separate_from_still_live_card_owner(round_state, monkeypatch, cleanup_verified):
    from pilotlab import shared_npu3
    root, state = round_state
    group = add_shared_group(root, state, "g", "npu", [("n1", 10, 101, 109)])
    path = root / "groups/g/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["dynamic_admission"] = {"inbox_dir": str(path.parent/"inbox")}
    write(path, manifest)
    state["shared_groups"]["g"]["manifest_sha256"] = sha(path)
    group.update(manifest_sha256=sha(path), status="running", allocation_released_unix=None,
                 remaining_descendants=[{"pid": 123}], owner={"pid": 123, "start_ticks": 1})
    worker = group["workers"]["n1"]
    worker.update(process={"pid": 456, "start_ticks": 1}, descendant_cleanup_verified=cleanup_verified,
                  remaining_descendants=[], worker_spec_sha256="worker-spec")
    # Chain integrity itself is exercised by runner tests. This witness tests
    # report semantics when a verified worker finishes inside a live lease.
    monkeypatch.setattr(shared_npu3, "read_admission_chain", lambda *args: {
        "published_workers": manifest["workers"], "acknowledged_workers": manifest["workers"],
        "commands": [], "closed": False, "next_sequence": 0}, raising=False)
    monkeypatch.setattr(shared_npu3, "worker_spec_digest", lambda _: "worker-spec", raising=False)
    monkeypatch.setattr(report3, "process_observation", lambda identity: {"alive": bool(identity and identity.get("pid") == 123)})
    write(path.parent/"group.json", group)
    write(root/"scheduler.json", state)
    report = report3.build_report(root, now=120)
    assert report["execution"]["verified_completed_tasks"] == int(cleanup_verified)
    attempt = report["tasks"][0]["attempts"][0]
    assert not attempt["group_completion_verified"]
    assert attempt["worker_completion_verified"] is cleanup_verified
    assert report["concurrency"]["verified_live_open_allocations"] == 1


@pytest.mark.parametrize("damage", ["manifest", "worker_result", "no_release"])
def test_shared_binding_or_release_damage_never_verifies_worker_completion(round_state, damage):
    root, state = round_state
    group = add_shared_group(root, state, "g", "npu", [("n1", 10, 101, 109)])
    if damage == "manifest":
        path = root / "groups/g/manifest.json"
        value = json.loads(path.read_text())
        value["changed"] = True
        write(path, value)
    elif damage == "worker_result":
        group["workers"]["n1"]["result_sha256"] = "wrong"
        write(root / "groups/g/group.json", group)
    else:
        group["allocation_released_unix"] = None
        write(root / "groups/g/group.json", group)
    report = report3.build_report(root, now=500)
    assert report["execution"]["verified_completed_tasks"] == 0
    assert report["observation_errors"]
    if damage == "no_release":
        assert report["concurrency"]["device_seconds"] == 0


def test_same_physical_card_is_not_counted_twice_for_overlapping_bad_leases(round_state):
    root, state = round_state
    add_shared_group(root, state, "g1", "npu", [("n1", 10, 101, 109)], allocation=(100, 110))
    add_shared_group(root, state, "g2", "npu", [("n2", 10, 106, 114)], allocation=(105, 115))
    report = report3.build_report(root, now=120)
    assert report["concurrency"]["peak"] == 1 and report["concurrency"]["device_seconds"] == 15
    assert not report["concurrency"]["accounting_complete"]
    assert any("overlapping" in item["reason"] for item in report["observation_errors"])


def shared_admission_log(root, group_id, acquired, admitted, *, physical_id=2):
    event = json.loads(queue_line("running", admitted))
    event.update(job_id=group_id, physical_ids=[physical_id], accounting_started_unix=acquired,
                 accounting_scope="one device; includes complete successful admission attempt")
    (root / "groups" / group_id / "stdout.log").write_text(json.dumps(event) + "\n")


def test_conservative_admission_overlap_is_separate_from_actual_lease_overlap(round_state):
    root, state = round_state
    add_shared_group(root, state, "g1", "npu", [("n1", 10, 101, 109)], allocation=(100, 110))
    add_shared_group(root, state, "g2", "npu", [("n2", 10, 111, 114)], allocation=(105, 115))
    shared_admission_log(root, "g1", 100, 101)
    shared_admission_log(root, "g2", 105, 111)
    report = report3.build_report(root, now=120)
    value = report["concurrency"]
    assert not report["observation_errors"]
    assert value["peak"] == 1 and value["device_seconds"] == 15
    assert value["conservative_accounting_overlap_seconds"] == 5
    assert value["recorded_group_device_seconds"] == value["ledger_device_seconds"] == 20
    assert value["admitted_leases"]["peak"] == 1
    assert value["admitted_leases"]["device_seconds"] == 13
    assert value["admitted_leases"]["timing_complete"]
    assert value["accounting_complete"]
    shared_admission_log(root, "g2", 105, 109)
    report = report3.build_report(root, now=120)
    assert not report["concurrency"]["accounting_complete"]
    assert any("overlapping admitted" in row["reason"] for row in report["observation_errors"])


@pytest.mark.parametrize("backend", ["cpu", "npu"])
@pytest.mark.parametrize("worker_status", ["pending", "starting", "running", "exited_waiting_cleanup"])
def test_worker_result_waits_for_live_owner_publication_without_claiming_corruption(round_state, monkeypatch, backend, worker_status):
    root, state = round_state
    group = add_shared_group(root, state, "g", backend, [("n1", 10, 101, 109)])
    record = state["tasks"]["n1"]
    record["status"] = record["attempts"][0]["status"] = "running"
    group.update(status="running", allocation_released_unix=None)
    group["workers"]["n1"].update(status=worker_status, return_code=None, finished_unix=None)
    group["workers"]["n1"].pop("result_sha256")
    write(root / "groups/g/group.json", group)
    write(root / "scheduler.json", state)
    monkeypatch.setattr(report3, "process_observation", lambda identity: {"alive": True})
    report = report3.build_report(root, now=120)
    assert not report["observation_errors"]
    assert report["execution"]["pending_result_bindings"] == 1
    assert report["execution"]["verified_completed_tasks"] == 0
    assert not report["exposure"]["physically_computed_loss_tokens_by_category"]
    assert not report["tasks"][0]["evidence_integrity_verified"]
    group["workers"]["n1"]["result_sha256"] = "wrong"
    write(root / "groups/g/group.json", group)
    report = report3.build_report(root, now=120)
    assert report["execution"]["pending_result_bindings"] == 0
    assert any("result binding mismatch" in row["reason"] for row in report["observation_errors"])


def test_dead_owner_or_terminal_worker_cannot_defer_missing_binding(round_state, monkeypatch):
    root, state = round_state
    group = add_shared_group(root, state, "g", "cpu", [("c1", 10, 101, 109)])
    record = state["tasks"]["c1"]
    record["status"] = record["attempts"][0]["status"] = "running"
    group.update(status="running")
    group["workers"]["c1"].update(status="running", return_code=None, finished_unix=None)
    group["workers"]["c1"].pop("result_sha256")
    write(root / "groups/g/group.json", group)
    write(root / "scheduler.json", state)
    monkeypatch.setattr(report3, "process_observation", lambda identity: {"alive": False})
    report = report3.build_report(root, now=120)
    assert report["execution"]["pending_result_bindings"] == 0
    assert any("result binding mismatch" in row["reason"] for row in report["observation_errors"])
    group["workers"]["c1"].update(status="completed", return_code=0, finished_unix=109)
    write(root / "groups/g/group.json", group)
    monkeypatch.setattr(report3, "process_observation", lambda identity: {"alive": True})
    report = report3.build_report(root, now=120)
    assert report["execution"]["pending_result_bindings"] == 0
    assert any("result binding mismatch" in row["reason"] for row in report["observation_errors"])


def test_selected_external_reference_keeps_original_identity_and_adds_no_new_exposure(round_state):
    root, state = round_state
    old = root.parent / "explicit-selected-evidence"
    task = {"id": "old-a3", "model_id": "selected-model", "phase": "a3", "source_identity": {"source": "old-frozen"},
            "batch_size": 2, "cpu_threads": 4, "lr": .0003}
    task_path = write(old / "task.json", task)
    result_path = write(old / "result.json", {"source_identity": task["source_identity"], "task_sha256": sha(task_path),
        "validation_final": {"nll": 4.1}, "initialization": {"resolved_recipe": "old-default"}})
    write(root / "selected.json", {"selected-model": {"recipe_id": "r2", "confirmation": [{
        "task_path": task_path, "result_path": result_path}]}})
    report = report3.build_report(root)
    assert not report["observation_errors"]
    assert report["selected_recipes"][0]["external_evidence"] is True
    assert report["selected_recipes"][0]["source_identity"] == task["source_identity"]
    assert report["selected_recipes"][0]["selection_nll"] == 4.1
    assert report["exposure"]["physically_computed_loss_tokens_by_category"] == {}


def test_positive_queue_cancellation_without_allocation_needs_no_worker_result(round_state):
    root, state = round_state
    task, record, results = add_task(root, state, "cancelled", "a1", 10, [(0, 0)], status="interrupted")
    task["device"] = "npu:0"
    write(record["task_path"], task)
    directory = Path(record["attempts"][0]["path"])
    Path(results[0][0]).unlink()
    write(directory / "queue.json", {"status": "cancelled", "job_id": "cancel-job"})
    events = [dict(json.loads(queue_line(status, stamp)), job_id="cancel-job")
              for status, stamp in (("queued", 100), ("cancelled", 101))]
    (directory / "stdout.log").write_text("".join(json.dumps(event) + "\n" for event in events))
    report = report3.build_report(root, now=120)
    assert not report["observation_errors"]
    assert report["tasks"][0]["attempts"][0]["never_started_verified"]
    assert report["concurrency"]["device_seconds"] == 0
