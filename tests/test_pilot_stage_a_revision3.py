"""A source repair cannot silently retry old learning or widen the catalog."""
from copy import deepcopy
import json

import pytest

from pilotlab.io import sha256
from pilotlab.stage_a_revision3 import verify_scope


@pytest.fixture
def plan(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path":str(path), "sha256":sha256(path)}
    config = {"budget_ledger":"original-budget", "deadline_unix":10000000000., "python":"original-python", "data":"original-data"}
    subjects = [{"id":f"m{i}", "model":{"fixture":i}} for i in range(344)]
    original = save("plan.json", {"subjects":subjects, "config":config})
    state = save("scheduler.json", {"status":"a_blocked_needs_repair", "stage_b_gate":"closed",
        "tasks":{"old":{"model_id":"m0", "prior_attempts":[{"path":"original"}]}}})
    selected = save("selected.json", {"m1":{"inherited_qualification":True}})
    audit = save("audit.json", {"status":"passed", "findings":[], "execution":{"scheduler":state},
        "lifecycle":{key:{"released":True} for key in ("processes","cgroup","queue")}})
    identity = {"source_files":{"/fixture/foundation/src/tide/attention.py":"a"*64}}
    screen_plan = save("screen-plan.json", {"source_identity":identity, "tasks":[{"id":"screen"}]})
    result = save("screen-result.json", {"status":"completed", "health":{"eligible":True},
        "recovery":{"trajectory":{"within_tolerance":True}}, "fp64":[]})
    screen = save("screen.json", {"status":"passed_mechanism_screen", "plan":screen_plan,
        "remaining_descendants":[], "result_files":{"screen":result}})
    return {"subjects":subjects[2:], "config":config, "source_identity":identity,
        "stage_a_revision":{"policy":"unstarted_original_subjects_only_no_stage_b",
            "original_plan":original, "original_scheduler":state, "original_audit":audit,
            "original_selected":selected,"deferred_subject_ids":["m0","m1"], "mechanism_screen":screen}}


def test_only_exact_previously_unstarted_catalog_is_admitted(plan):
    assert verify_scope(plan)["deferred_subject_ids"] == ["m0","m1"]
    bad = deepcopy(plan)
    bad["subjects"].append({"id":"m0", "model":{"fixture":0}})
    with pytest.raises(ValueError, match="previously unstarted"):
        verify_scope(bad)
    bad = deepcopy(plan)
    bad["subjects"].append({"id":"m1", "model":{"fixture":1}})
    with pytest.raises(ValueError, match="previously unstarted"):
        verify_scope(bad)
    bad = deepcopy(plan)
    bad["subjects"][0]["model"]["fixture"] = "changed"
    with pytest.raises(ValueError, match="previously unstarted"):
        verify_scope(bad)


def test_budget_and_frozen_worker_cannot_silently_change(plan):
    for key,value in (("deadline_unix",10000000001.), ("budget_ledger","reset-budget"), ("worker_source","old-source")):
        bad = deepcopy(plan)
        bad["config"][key] = value
        with pytest.raises(ValueError):
            verify_scope(bad)


def test_unscreened_source_and_old_verdict_transfer_are_rejected(plan):
    bad = deepcopy(plan)
    bad["source_identity"]["source_files"]["/fixture/foundation/src/tide/attention.py"] = "b"*64
    with pytest.raises(ValueError, match="screened source"):
        verify_scope(bad)
    bad = deepcopy(plan)
    bad["stage_a_transfer"] = {"path":"old-verdict"}
    with pytest.raises(ValueError, match="old qualification"):
        verify_scope(bad)


def test_controller_relay_preserves_original_worker_scope_and_budget(plan, tmp_path):
    def save(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return {"path":str(path), "sha256":sha256(path)}

    original = save("screened-plan.json", plan)
    prepared = save("worker.json", {"campaign_plan":original, "source":"/fixed/worker",
        "foundation":"/fixed/foundation", "source_identity":plan["source_identity"]})
    manifest = save("import.json", {"worker_source_identity":plan["source_identity"], "old_prepared":prepared})
    relay = deepcopy(plan)
    relay.update(worker_prepared=prepared, resume_import=manifest, controller_source_identity={"version":"new"})
    relay["config"].update(worker_source="/fixed/worker", worker_foundation="/fixed/foundation",
        resume_import=manifest, max_workers=None, max_cpu_threads=None, host_cpu_affinity=[0, 1])
    assert verify_scope(relay) == plan["stage_a_revision"]
    for key,value in (("worker_source","/other"), ("deadline_unix",0)):
        changed = deepcopy(relay)
        changed["config"][key] = value
        with pytest.raises(ValueError, match="worker, runtime contract or budget"):
            verify_scope(changed)
    changed = deepcopy(relay)
    changed["subjects"].pop()
    with pytest.raises(ValueError, match="A-only scope"):
        verify_scope(changed)
