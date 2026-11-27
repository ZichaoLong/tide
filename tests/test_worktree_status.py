"""A scheduler receipt requires independent service and process observations."""
import json

import pytest

from pilotlab import report3
from scripts import status_runtime as runtime


def test_clean_checkout_stays_empty(tmp_path):
    view = runtime.observe(tmp_path)
    assert view["kind"] == "unavailable"
    assert view["pointer"]["status"] == "missing"
    assert not list(tmp_path.iterdir())


@pytest.fixture
def launch(tmp_path):
    directory = tmp_path / ".local/round"
    directory.mkdir(parents=True)
    pointer = tmp_path / runtime.LAUNCH
    pointer.parent.mkdir()
    pointer.write_text(json.dumps({"unit": "lab.service", "output": str(directory),
                                   "systemd": {"InvocationID": "first"}}))
    return pointer, directory


@pytest.mark.parametrize("bad", ["{", "[]", '{"unit": "lab.service"}', '{"output":NaN}'])
def test_bad_pointer_is_unavailable(tmp_path, launch, bad):
    pointer, _ = launch
    pointer.write_text(bad)
    assert runtime.observe(tmp_path)["kind"] == "unavailable"


def test_missing_scheduler_and_foreign_output(tmp_path, launch):
    pointer, directory = launch
    assert runtime.observe(tmp_path)["kind"] == "unavailable"
    assert not list(directory.iterdir())
    data = json.loads(pointer.read_bytes())
    data["output"] = str(tmp_path.parent / "another-checkout/.local/round")
    pointer.write_text(json.dumps(data))
    assert "另一工作树" in runtime.observe(tmp_path)["summary"]


@pytest.fixture
def host(launch, monkeypatch):
    pointer, directory = launch
    scheduler = directory / "scheduler.json"
    scheduler.write_text(json.dumps({"schema": "pilot-r3-scheduler-v1", "status": "running",
                                     "stage": "stage-a", "stage_b_gate": "closed",
                                     "controller_identity": {"pid": 77, "start_ticks": 7}}))
    controller = {"pid": 77, "alive": True, "same_start_ticks": True}
    properties = {"LoadState": "loaded", "ActiveState": "active", "MainPID": "77",
                  "ControlGroup": "/lab.service", "InvocationID": "first"}
    group = {"status": "populated", "members": [77, 81, 82]}
    monkeypatch.setattr(report3, "process_observation", lambda identity: controller)
    monkeypatch.setattr(runtime, "systemd_observation", lambda unit: {"status": "observed", "properties": properties})
    monkeypatch.setattr(runtime, "cgroup_observation", lambda service: group)
    monkeypatch.setattr(runtime, "driver_processes", lambda: {0: (81, 90), 1: (89,), 2: (77, 82)})
    return pointer, properties, group, controller


def test_live_filters_driver_pids_by_service(tmp_path, host):
    view = runtime.observe(tmp_path)
    assert view["kind"] == "live"
    assert view["scheduler"]["stage_b_gate"] == "closed"
    assert view["driver"]["service_pids_by_physical_card"] == {0: [81], 2: [77, 82]}


@pytest.mark.parametrize("mismatch", ["invocation", "main_pid", "cgroup", "start_time", "inactive", "bad_binding"])
def test_running_scheduler_with_unbound_service_is_a_snapshot(tmp_path, host, mismatch):
    pointer, properties, group, controller = host
    if mismatch == "invocation":
        properties["InvocationID"] = "restarted"
    elif mismatch == "main_pid":
        properties["MainPID"] = "81"
    elif mismatch == "cgroup":
        group["members"] = [81]
    elif mismatch == "start_time":
        controller["same_start_ticks"] = False
    elif mismatch == "inactive":
        properties["ActiveState"] = "inactive"
    else:
        data = json.loads(pointer.read_bytes())
        data["systemd"] = "invalid"
        pointer.write_text(json.dumps(data))
    view = runtime.observe(tmp_path)
    assert view["kind"] == "snapshot_only"
    assert view["scheduler"]["status"] == "running"
    assert "driver" not in view


def test_driver_unavailable_preserves_the_verified_controller(tmp_path, host, monkeypatch):
    def no_driver():
        raise FileNotFoundError("npu-smi")

    monkeypatch.setattr(runtime, "driver_processes", no_driver)
    view = runtime.observe(tmp_path)
    assert view["kind"] == "live"
    assert view["driver"] == {"status": "unavailable", "reason": "npu-smi"}


def test_pointer_replacement_during_sampling_downgrades_live(tmp_path, host, monkeypatch):
    def replace_pointer():
        host[0].write_text("{}")
        return {}

    monkeypatch.setattr(runtime, "driver_processes", replace_pointer)
    view = runtime.observe(tmp_path)
    assert view["kind"] == "snapshot_only"
    assert view["next_check"] == "python scripts/status.py"


def test_resume_and_review_counts_bind_the_two_manifests_to_their_queue(tmp_path):
    from pilotlab.prepare_resume3 import reference
    from pilotlab.runq_store import connect, submit
    from pilotlab.scheduler3 import freeze_json

    root = tmp_path/"queue"
    db = connect(root)
    spec = {"allowed":["cpu"], "ram":{"cpu":256}, "hbm":0, "owner":"campaign"}
    a = submit(db, "resume", {**spec, "task_id":"model-a3"})
    b = submit(db, "review", {**spec, "task_id":"early-review-old-model-cpu", "purpose":"deferred_source_review"})
    db.execute("UPDATE jobs SET state='running' WHERE id=?", (a,))
    db.close()
    resumed = freeze_json(tmp_path/"resumes.json", {"new_jobs":{"model-a3":a}})
    reviewed = freeze_json(tmp_path/"reviews.json", {"jobs":{"old-model/cpu":{"job":b}}})
    launch = {"output":"campaign", "runq_root":str(root), "last_signal_admission":reference(resumed),
              "deferred_source_review":{"submissions":reference(reviewed)}}
    out = {"summary":"", "sources":[]}
    runtime.observe_queue_additions(launch, out)
    assert out["queue_additions"] == {"signal_resumes":{"running":1}, "deferred_source_reviews":{"queued":1}}
    incorrect = {"summary":"", "sources":[]}
    runtime.observe_queue_additions({**launch, "output":"another campaign"}, incorrect)
    assert "queue_additions" not in incorrect and "不可核验" in incorrect["summary"]


def test_early_followup_keeps_unreleased_npu_requests_distinct_from_submitted_work(tmp_path, monkeypatch):
    from pilotlab.io import atomic_json
    from pilotlab.prepare_resume3 import reference
    from pilotlab.runq_store import connect, submit
    local = tmp_path / ".local"
    owner, primary = str(local / "early" / "round"), str(local / "main")
    queue = tmp_path / "queue"
    db = connect(queue)
    gate = dict(owner=primary, job_key="main/a/0", task_id="a")
    gate["job"] = submit(db, gate["job_key"], dict(owner=primary, task_id="a", allowed=["npu"], hbm=16, ram={"npu": 8}))
    review_file, plan_file = local / "review.json", local / "plan.json"
    atomic_json(review_file, dict(modes={"compare_finalists": 1}, phase_counts={"a1": 4}))
    config = dict(predecessor=gate, runq_root=str(queue))
    plan = dict(config=config, review=reference(review_file), unit="early.service", output=owner)
    atomic_json(plan_file, plan)
    link = dict(unit=plan["unit"], invocation="test", plan=reference(plan_file), output=owner,
                review=plan["review"], predecessor=gate)
    atomic_json(local / "early/round/scheduler.json", dict(config=config, status="running", stage_b_gate="closed",
                controller_identity=dict(pid=77, start_ticks=7),
                tasks={"cpu": dict(status="pending", device="cpu", runq_id=2), "npu": dict(status="pending", device="npu:0")}))
    submit(db, "early/cpu/0", dict(owner=owner, task_id="cpu", allowed=["cpu"], ram={"cpu": 8}))
    monkeypatch.setattr(runtime, "systemd_observation", lambda _: {"properties": {"ActiveState": "inactive"}})
    data = dict(summary="", sources=[])
    launch = dict(runq_root=str(queue), output=primary, deferred_followup=link)
    runtime.observe_deferred_followup(tmp_path, launch, data)
    view = data["deferred_followup"]
    assert view["queue_counts"] == {"queued": 1} and view["by_backend"]["npu"] == {}
    assert view["planned_ready_npu_without_queue_job"] == 1 and view["npu_release"] is None
    assert view["controller_live"] is False and view["predecessor"]["state"] == "queued"
