import argparse
import importlib.util
from pathlib import Path
import sys
import time

import pytest

from pilotlab.data3 import DocumentStore3
from pilotlab.io import atomic_json
from pilotlab.preflight3 import claim_token_budget
from pilotlab.prepare_resume3 import reference
from pilotlab.scheduler3 import freeze_json, read
from scripts import deferred_source_probe as probe
from scripts.deferred_source_probe import compare, one_step
from test_pilot_resume_adapter3 import interrupted


def test_paired_update_restores_the_original_checkpoint_on_both_arms(interrupted):
    task = interrupted["task"]
    checkpoint = interrupted["result"]["checkpoint"]
    case = {"task":reference(interrupted["task_path"]), "model_id":task["model_id"], "key":task["id"],
            "start":"checkpoint_successor", "evidence":checkpoint, "checkpoint_task_id":task["id"], "sample":True}
    store = DocumentStore3(Path(interrupted["data"]))
    args = task["source_identity"], {}, store, "cpu"
    old = one_step(case, "historical", *args)
    new = one_step(case, "reference", *args)
    assert old["outcome"] == "updated" and old["from_step"] == checkpoint["steps"]
    assert old["checkpoint_state_sha256"] == checkpoint["state_sha256"]
    assert compare(old, new)["outcome"] == "sample_exact"
    assert compare(old, dict(new, input_sha256="other"))["outcome"] == "different_start"


@pytest.fixture
def source_plan(tmp_path, interrupted, monkeypatch):
    monkeypatch.setenv("RUNQ_DEVICE", "cpu")
    monkeypatch.setattr("pilotlab.prepare_campaign3.storage_observation", lambda *args, **kwargs: {})
    task = interrupted["task"]
    checkpoint = interrupted["result"]["checkpoint"]
    tokens = task["batch_size"] * task["sequence_length"]
    case = dict(task=reference(interrupted["task_path"]), model_id=task["model_id"], key="a",
                start="checkpoint_successor", evidence=checkpoint, checkpoint_task_id=task["id"],
                sample=True, tokens_per_arm=tokens)
    original = tmp_path / "original"
    ledger = tmp_path / "ledger.json"
    now = time.time()
    deadline = now + 30 * 86400
    atomic_json(ledger, dict(started_unix=now, deadline_unix=deadline, device_seconds=0, segments={},
                             max_device_seconds=7200*3600, preparation_max_device_seconds=48*3600,
                             preparation_max_processed_tokens=64*2**20))
    identity = reference(freeze_json(original / "identity.json", task["source_identity"]))
    tide = Path(next(iter(importlib.util.find_spec("tide").submodule_search_locations)))
    arm = dict(source=str(Path(probe.__file__).resolve().parents[1]), foundation=str(tide.parents[1]),
               model_policy={}, identity=identity)
    script = reference(probe.__file__)
    source = reference(freeze_json(original / "plan.json", dict(
        model=task["model_id"], device="cpu", ledger=str(ledger), deadline=deadline, max_run_seconds=0.001,
        probe_sha256=script["sha256"], arms=dict(historical=arm, reference=arm), data=str(interrupted["data"]),
        cases=[case, dict(case, key="b")])))
    return source


def test_frozen_attempt_merges_only_missing_points_and_charges_them_once(tmp_path, interrupted, source_plan):
    source = source_plan
    plan = probe.bound(source)
    task = interrupted["task"]
    case = plan["cases"][0]
    tokens = case["tokens_per_arm"]
    original = Path(source["path"]).parent
    ledger = Path(plan["ledger"])
    script = reference(probe.__file__)
    origin = one_step(case, "historical", task["source_identity"], {}, DocumentStore3(interrupted["data"]), "cpu")
    prior = {"historical": {"a": origin, "b": dict(origin, case="b")}, "reference": {"a": origin}}
    refs = {name: reference(freeze_json(original / "observations" / (name + ".json"), rows))
            for name, rows in prior.items()}
    path = freeze_json(tmp_path / "retry" / "attempt.json", dict(
        source_plan=source, source_probe=script, probe_sha256=script["sha256"], prior_observations=refs))
    claim_token_budget(ledger, str(path.parent), tokens, probe.checksum(path))
    args = argparse.Namespace(plan=path, sha256=probe.checksum(path), arm=None)
    probe.execute(args)
    out = path.parent / "observations"
    assert read(out/"historical.json") == prior["historical"]
    assert read(out/"reference.json")["a"] == origin
    comparison = read(out/"comparison.json")
    assert comparison["plan_sha256"] == source["sha256"]
    assert {key: value["outcome"] for key, value in comparison["comparisons"]["reference"].items()} == {
        "a": "sample_exact", "b": "sample_exact"}
    claim = read(ledger)["preparation_token_reservations"][str(path.parent)]
    assert claim["status"] == "observed" and claim["observed_completed_loss_tokens"] == tokens
    assert all(probe.bound(ref) == prior[name] for name, ref in refs.items())
    with pytest.raises(ValueError, match="reservation"):
        probe.execute(args)
    atomic_json(original / "observations" / "historical.json", {})
    with pytest.raises(ValueError, match="changed input"):
        probe.execute(args)


def test_unstarted_attempt_consumes_the_existing_claim(tmp_path, source_plan):
    plan = probe.bound(source_plan)
    directory = Path(source_plan["path"]).parent
    tokens = len(plan["arms"]) * sum(case["tokens_per_arm"] for case in plan["cases"])
    claim_token_budget(plan["ledger"], str(directory), tokens, source_plan["sha256"])
    script = reference(probe.__file__)
    path = freeze_json(tmp_path / "queued" / "attempt.json", dict(source_plan=source_plan, source_probe=script,
                       probe_sha256=script["sha256"], use_unstarted_reservation=True))
    probe.execute(argparse.Namespace(plan=path, sha256=probe.checksum(path), arm=None))
    claims = read(plan["ledger"])["preparation_token_reservations"]
    assert set(claims) == {str(directory)}
    assert claims[str(directory)]["observed_completed_loss_tokens"] == tokens
    assert read(path.parent / "observations/comparison.json")["status"] == "observations_recorded"
    assert not (directory / "observations").exists()


@pytest.mark.parametrize("exhausted", ["remaining_wall_seconds", "remaining_preparation_device_seconds",
                                       "remaining_device_seconds"])
def test_arm_stops_when_shared_budget_expires(tmp_path, monkeypatch, exhausted):
    budget = dict(deadline_unix=time.time()+60, remaining_wall_seconds=60, remaining_preparation_device_seconds=60,
                  remaining_device_seconds=60)
    readings = iter([budget, {**budget, exhausted: 0}])
    monkeypatch.setattr("pilotlab.prepare_campaign3.budget_remaining", lambda *args: next(readings))
    plan = dict(deadline=budget["deadline_unix"], device="npu:0", ledger="ledger")
    with pytest.raises(TimeoutError, match="campaign deadline or device-time budget"):
        probe.run_with_budget([sys.executable, "-c", "import time; time.sleep(60)"],
                              dict(source=str(tmp_path), foundation=str(tmp_path)), plan)
