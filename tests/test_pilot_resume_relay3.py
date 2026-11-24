"""Real CPU relays preserve origin, original dependencies and full state."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from pilotlab import resume_adapter3 as adapter
from pilotlab.io import sha256
from pilotlab.scheduler3 import evaluation_bound
from test_pilot_resume_adapter3 import arguments, interrupted, run_adapter
from test_pilot_worker3 import ROOT, run_worker


def ref(path):
    return {"path": str(path), "sha256": sha256(path)}


def relay_receipt(sample, results=None, reservation=None):
    return adapter.build_relay_receipt(ref(sample["task_path"]),
        results or [ref(sample["output"]/"result.json")],
        evaluation_bound(sample["task"]) if reservation is None else reservation)


def test_queue_relocation_reuses_pinned_job_and_charges_checkpoint_increment(tmp_path, interrupted):
    from pilotlab.runq_campaign import QueueSupervisor
    from pilotlab.runq_store import connect, submit
    from pilotlab.scheduler3 import Blocked

    task = dict(interrupted["task"], id="relocated", evaluation_budget_key="model", evaluation_token_limit=96)
    _, _, old_output = run_worker(tmp_path, interrupted["data"], task, interrupt=True)
    sample = dict(interrupted, task=task, task_path=tmp_path/"relocated.json", output=old_output)
    receipt = relay_receipt(sample)
    args = arguments(tmp_path, sample, receipt)
    queue = QueueSupervisor.__new__(QueueSupervisor)
    queue.registered, queue.owner = {}, "test-campaign"
    queue.config = {"run_id": "batch", "resume_adapter": ref(adapter.__file__)}
    queue.db = connect(tmp_path/"queue")
    previous = {"runq_id": 99, "physical_id": 9, "reason": "signal", "status": "stopped_budget", "path": str(old_output)}
    row = {"attempts": [previous], "task_path": str(sample["task_path"]), "resume_receipt": ref(args.receipt),
           "queue_relocation": {"from": 9, "to": 7, "previous_job": 99, "next_attempt": 1},
           "evaluation_budget_key": "model", "evaluation_reserved_tokens": evaluation_bound(task)}
    queue.state = {"tasks": {task["id"]: row}}
    spec = {"owner": queue.owner, "task_id": task["id"], "allowed": ["npu"], "hbm": 10, "ram": {"npu": 10}}
    job = submit(queue.db, "batch/relocated/1", spec, "npu:7")
    assert queue.register_job(task, row) == job
    row["runq_id"] = job
    lease = queue.db.execute("""INSERT INTO leases(card,role,state,trigger_job,grant,created)
                                VALUES(9,'manual','held',?,'null',0)""", (job,)).lastrowid
    queue.db.execute("UPDATE jobs SET state='reserved',lease=? WHERE id=?", (lease, job))
    with pytest.raises(Blocked, match="wrong card"):
        queue.reserve_task(task, row)
    queue.db.execute("UPDATE leases SET card=7 WHERE id=?", (lease,))
    queue.reserve_task(task, row)
    assert row["evaluation_reserved_tokens"] == evaluation_bound(task) + 32 == task["evaluation_token_limit"]

    run = run_adapter(args)
    assert run.returncode == 0, run.stderr
    result = ref(args.output/"result.json")
    row.update(result_path=result["path"], result_sha256=result["sha256"])
    row["attempts"].append({"runq_id": job, "physical_id": 7, "path": str(args.output), "return_code": 0})
    queue.verify_completed(task, row)
    row["attempts"][-1]["physical_id"] = 9
    with pytest.raises(Blocked, match="physical card"):
        queue.verify_completed(task, row)
    queue.db.close()


def test_resume_after_an_abandoned_attempt_never_recovers_its_evaluation_reservation(tmp_path, interrupted):
    from pilotlab import prepare_relay3 as relay
    old=copy.deepcopy(interrupted["result"])
    old.update(status="running",checkpoint=None)
    path=tmp_path/"abandoned-result.json";path.write_text(json.dumps(old))
    latest=ref(interrupted["output"]/"result.json")
    evidence=[{"result":ref(path)},{"result":latest,"worker_return_code":2}]
    limit=evaluation_bound(interrupted["task"])
    record={"task_path":str(interrupted["task_path"]),"status":"failed","evaluation_reserved_tokens":2*limit}
    row=relay.classify(interrupted["task"],record,evidence,interrupted["task"]["source_identity"])
    assert row["evaluation_reserved_tokens"]==2*limit
    assert row["abandoned_attempt_evaluation_upper_bound_tokens"]==limit
    assert row["receipt_value"]["prior_results"]==[latest]
    assert row["receipt_value"]["prior_evaluation_reserved_tokens"]==limit
    assert row["additional_evaluation_reserved_tokens"]==32


def test_repeated_a1_relay_keeps_initial_origin_and_charges_only_extra_work(tmp_path, interrupted):
    receipt = relay_receipt(interrupted)
    assert receipt["evaluation_additional_reservation_tokens"] == 32
    first, second = tmp_path/"first", tmp_path/"second"
    first.mkdir(); second.mkdir()
    args = arguments(first, interrupted, receipt)
    harness = first/"signal_adapter.py"
    harness.write_text('''
import os,signal
from pilotlab import worker3,resume_adapter3
original=worker3.optimizer_for
def factory(model,task):
 optimizer=original(model,task); step=optimizer.step; count=0
 def interrupted(*args,**kwargs):
  nonlocal count
  count+=1
  if count==2: os.kill(os.getpid(),signal.SIGTERM)
  return step(*args,**kwargs)
 optimizer.step=interrupted
 return optimizer
worker3.optimizer_for=factory
raise SystemExit(resume_adapter3.main())
''')
    process = run_adapter(args, harness)
    assert process.returncode == 2, (process.stdout, process.stderr)
    partial = json.loads((args.output/"result.json").read_text())
    assert partial["training"]["from_tokens"] == 32 and partial["training"]["tokens"] == 64
    assert partial["validation_initial"]["tokens"] == 0
    refs = [ref(interrupted["output"]/"result.json"), ref(args.output/"result.json")]
    reservation = evaluation_bound(interrupted["task"])+receipt["evaluation_additional_reservation_tokens"]
    repeated = relay_receipt(interrupted, refs, reservation)
    assert repeated["evaluation_additional_reservation_tokens"] == 32
    assert repeated["prior_observed_evaluation_tokens"] == 16
    assert repeated["initial_evaluation"]["result"] == refs[0]
    args = arguments(second, interrupted, repeated)
    process = run_adapter(args)
    assert process.returncode == 0, (process.stdout, process.stderr)
    result = json.loads((args.output/"result.json").read_text())
    baseline, expected, _ = run_worker(tmp_path, interrupted["data"], dict(interrupted["task"], id="baseline"))
    actual = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=False)
    assert actual["state_sha256"] == expected["state_sha256"]
    assert result["validation_final"]["nll"] == baseline["validation_final"]["nll"]
    assert result["validation_initial"]["exposure"] == 0
    assert result["validation_initial"]["inherited_from"]["path"] == refs[0]["path"]
    assert result["evaluation"]["cumulative_task_observed_tokens"] == 32
    assert result["recovery"]["numeric_trajectory"]["status"] == "not_tested"


@pytest.mark.parametrize("phase", ["a2", "a3"])
def test_stage_relay_preserves_parent_checkpoint_and_nonzero_initial_exposure(tmp_path, interrupted, phase, monkeypatch):
    base = dict(interrupted["task"], id="stage-parent", target_tokens=32, eval_interval_tokens=64)
    parent, _, _ = run_worker(tmp_path, interrupted["data"], base)
    origin = 32
    if phase == "a3":
        origin = 64
        base = dict(base, id="stage-parent-a2", phase="a2", target_tokens=origin, evaluation_start_tokens=32,
                    resume_checkpoint=parent["checkpoint"]["path"], resume_checkpoint_sha256=parent["checkpoint"]["sha256"])
        parent, _, _ = run_worker(tmp_path, interrupted["data"], base)
    task = dict(base, id="stage-"+phase, phase=phase, target_tokens=192, evaluation_start_tokens=origin,
                resume_checkpoint=parent["checkpoint"]["path"], resume_checkpoint_sha256=parent["checkpoint"]["sha256"])
    partial, _, output = run_worker(tmp_path, interrupted["data"], task, interrupt=True)
    sample = dict(interrupted, task=task, task_path=tmp_path/(task["id"]+".json"), result=partial, output=output)
    before = sample["task_path"].read_bytes()
    receipt = relay_receipt(sample)
    assert receipt["initial_evaluation"]["exposure"] == origin
    assert receipt["checkpoint"]["tokens"] == origin+32
    args = arguments(tmp_path, sample, receipt)
    process = run_adapter(args)
    assert process.returncode == 0, (process.stdout, process.stderr)
    result = json.loads((args.output/"result.json").read_text())
    assert sample["task_path"].read_bytes() == before
    assert result["training"]["from_tokens"] == origin+32
    assert result["validation_initial"]["exposure"] == origin
    assert result["validation_initial"]["nll"] == partial["validation_initial"]["nll"]
    assert result["validation_initial"]["tokens"] == 0
    assert result["evaluation"]["tokens"] == 32  # real intermediate + final evaluations
    assert len((args.output/"validation.jsonl").read_text().splitlines()) == 1
    baseline, expected, _ = run_worker(tmp_path, interrupted["data"],
        dict(interrupted["task"], id="baseline", target_tokens=192, eval_interval_tokens=64))
    actual = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=False)
    assert actual["state_sha256"] == expected["state_sha256"]
    assert result["validation_final"]["nll"] == baseline["validation_final"]["nll"]
    from pilotlab import gate3
    monkeypatch.setattr(gate3, "MI", origin if phase == "a3" else origin*4)
    assert gate3.eligible(task, result, require_resume=True) == []
    assert gate3.continuation_matches((base, parent), task, result)


@pytest.mark.parametrize("phase", ["init", "preflight"])
def test_interrupted_initial_evaluation_reruns_only_at_original_exposure(tmp_path, interrupted, phase):
    task = dict(interrupted["task"], id="initial-"+phase, phase=phase, target_tokens=64, batch_size=1)
    task_path, output = tmp_path/"task.json", tmp_path/"original"
    task_path.write_text(json.dumps(task))
    harness = '''
import os,signal
from pilotlab import worker3
original=worker3.evaluate
calls=0
def evaluate(model,*args,**kwargs):
 global calls
 calls+=1
 if calls!=1: return original(model,*args,**kwargs)
 forward=model.forward
 def partial(*a,**k):
  value=forward(*a,**k)
  os.kill(os.getpid(),signal.SIGTERM)
  return value
 model.forward=partial
 try: return original(model,*args,**kwargs)
 finally: model.forward=forward
worker3.evaluate=evaluate
raise SystemExit(worker3.main())
'''
    environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run([sys.executable, "-c", harness, "--task", str(task_path), "--output", str(output),
        "--data", str(interrupted["data"]), "--device", "cpu", "--deadline-unix", str(time.time()+120)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
    assert process.returncode == 2, (process.stdout, process.stderr)
    previous = json.loads((output/"result.json").read_text())
    assert previous["checkpoint"]["tokens"] == 0 and previous["validation_initial"]["tokens"] == 8
    sample = dict(interrupted, task=task, task_path=task_path, result=previous, output=output)
    receipt = relay_receipt(sample)
    assert receipt["initial_evaluation"] == {"mode": "rerun", "exposure": 0}
    assert receipt["evaluation_additional_reservation_tokens"] == 40
    args = arguments(tmp_path, sample, receipt)
    process = run_adapter(args)
    assert process.returncode == 0, (process.stdout, process.stderr)
    result = json.loads((args.output/"result.json").read_text())
    assert result["training"]["from_tokens"] == 0
    assert result["validation_initial"]["tokens"] == 16 and result["validation_initial"]["exposure"] == 0
    assert not result["execution_adapter"]["initial_evaluation_inherited"]
    assert result["evaluation"]["cumulative_task_observed_tokens"] == 40


@pytest.mark.parametrize("damage", ["causality", "missing_causality", "failure", "initial_incomplete", "origin"])
def test_v2_rejects_rehashed_scientific_failures_or_late_initial_recreation(tmp_path, interrupted, damage):
    result = copy.deepcopy(interrupted["result"])
    if damage == "causality":
        result["causality"]["within_tolerance"] = False
    elif damage == "missing_causality":
        result.pop("causality")
    elif damage == "failure":
        result["failure_class"] = "numerical_error"
    elif damage == "initial_incomplete":
        result["validation_initial"]["status"] = "budget_exhausted"
    else:
        result["training"]["from_tokens"] = 16
    path = tmp_path/"result.json"; path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        relay_receipt(interrupted, [ref(path)])


def test_v2_reservation_and_original_observation_cannot_change_after_freeze(tmp_path, interrupted):
    receipt = relay_receipt(interrupted)
    receipt["evaluation_additional_reservation_tokens"] = 0
    args = arguments(tmp_path, interrupted, receipt)
    with pytest.raises(ValueError, match="recomputed evidence"):
        adapter.validate_receipt(args, __import__("pilotlab.worker3", fromlist=["worker3"]))
