"""Append-bound admission with real CPU processes and a fake, flock-held card."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
import time

import pytest

from pilotlab import shared_npu3 as shared
from pilotlab.io import atomic_json, sha256
from test_pilot_shared_npu3 import FakeLease, manifest


WORKER = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
out,config=Path(sys.argv[1]),json.loads(sys.argv[2])
task_path=Path(sys.argv[sys.argv.index('--task')+1])
task=json.loads(task_path.read_text())
if config.get('orphan'):
    pid=os.fork()
    if pid==0:
        os.setsid()
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        time.sleep(20)
        os._exit(0)
    (out/'orphan.pid').write_text(str(pid))
tokens=0
end=time.monotonic()+config.get('sleep',.3)
finish_after=Path(config['finish_after']) if 'finish_after' in config else None
while time.monotonic()<end and (finish_after is None or not finish_after.exists()):
    tokens+=32
    p=out/'heartbeat.tmp'
    p.write_text(json.dumps({'task_id':task['id'],'pid':os.getpid(),'updated_unix':time.time(),
                            'status':'training','tokens':tokens}))
    p.replace(out/'heartbeat.json')
    time.sleep(.01)
if finish_after is not None and not finish_after.exists():
    raise TimeoutError('concurrent worker did not start before the fixture deadline')
result={'id':task['id'],'model_id':task['model_id'],'phase':task['phase'],
        'status':config.get('status','completed'),'source_identity':task['source_identity'],
        'task_sha256':hashlib.sha256(task_path.read_bytes()).hexdigest(),
        'training':{'tokens':tokens,'segment_tokens':tokens},'device_memory':{'npu_reserved_peak_bytes':1234}}
if config.get('failure_class'):result['failure_class']=config['failure_class']
(out/'result.json').write_text(json.dumps(result))
raise SystemExit(config.get('code',0))
'''


def request(tmp_path, configs=None):
    value = manifest(tmp_path, backend="npu", configs=configs or [{"sleep": .3}, {"sleep": .5}, {"sleep": .2}])
    (tmp_path / "fake_worker.py").write_text(WORKER)
    value.update(external_occupant_policy="continue_running", max_parallel=8)
    all_workers = value["workers"]
    for worker in all_workers:
        task_path = tmp_path / f'{worker["id"]}-task.json'
        atomic_json(task_path, {"id": worker["id"], "model_id": "fake", "phase": "a1", "cpu_threads": 1,
                               "source_identity": {"fixture": "cpu-only"}})
        worker.update(task_path=str(task_path), task_sha256=sha256(task_path))
        worker["argv"] += ["--task", str(task_path), "--output", worker["output"], "--device", "npu:0",
                           "--deadline-unix", str(value["deadline_unix"])]
    value["workers"] = all_workers[:1]
    value["dynamic_admission"] = {"inbox_dir": str(tmp_path / "inbox"), "max_total_workers": 16,
        "max_commands": 17, "controller_identity": shared.proc_identity(os.getpid(), strict=True),
        "idle_timeout_seconds": .03, "stable_window_seconds": .05, "heartbeat_max_age_seconds": 2,
        "integrity_interval_seconds": .1}
    path = tmp_path / "manifest.json"
    atomic_json(path, value)
    return value, path, all_workers


def publish(value, path, sequence, previous, kind, workers=()):
    inbox = Path(value["dynamic_admission"]["inbox_dir"])
    target = inbox / f"{sequence:08d}.json"
    assert not target.exists()
    command = {"schema": shared.ADMISSION_SCHEMA, "sequence": sequence, "previous_sha256": previous,
               "manifest_sha256": sha256(path), "job_id": value["job_id"], "deadline_unix": value["deadline_unix"],
               "kind": kind}
    if kind == "admit":
        command["workers"] = list(workers)
    atomic_json(target, command)
    return sha256(target)


def runner(value, path, tmp_path):
    instance = shared.GroupRunner(value, tmp_path / "group", manifest_path=path, lease_factory=FakeLease)
    def lease(*args):
        item = FakeLease(*args)
        item.runner = instance
        return item
    instance.lease_factory = lease
    return instance


def wait_for(path, predicate, *, timeout=8):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.exists():
            state = json.loads(path.read_text())
            if predicate(state):
                return state
        time.sleep(.005)
    raise AssertionError("group state did not reach the expected bounded condition")


def test_stop_is_reasserted_when_worker_resets_its_signal_flag_during_startup(tmp_path):
    value,path,workers=request(tmp_path,[{"sleep":.1}])
    value["stop_grace_seconds"]=3.
    atomic_json(path,value)
    (tmp_path/"fake_worker.py").write_text('''
import json,signal,sys,time
from pathlib import Path
out=Path(sys.argv[1]);count=0;ready=time.monotonic()
def stop(*args):
 global count
 if time.monotonic()-ready>=.2: count+=1
signal.signal(signal.SIGTERM,stop)
(out/'ready').write_text('ready')
while count<1: time.sleep(.01)
(out/'result.json').write_text(json.dumps({'status':'budget_exhausted','stop_reason':'signal'}))
raise SystemExit(2)
''')
    instance=runner(value,path,tmp_path)
    def stop_after_ready():
        until=time.monotonic()+5
        while not (Path(workers[0]["output"])/"ready").exists() and time.monotonic()<until:time.sleep(.01)
        instance.request_stop(15)
    thread=threading.Thread(target=stop_after_ready);thread.start()
    state=instance.run();thread.join(timeout=6)
    assert state["workers"]["w0"]["return_code"]==2
    assert state["workers"]["w0"]["descendant_cleanup_verified"]
    assert state["remaining_descendants"]==[] and state["queue_released_unix"]


def test_append_backfills_held_card_and_publishes_individual_completion_before_drain(tmp_path):
    later_heartbeat = tmp_path / "worker-2/heartbeat.json"
    value, path, workers = request(tmp_path, [
        {"sleep": .45}, {"sleep": 10, "finish_after": str(later_heartbeat)}, {"sleep": .15},
    ])
    instance = runner(value, path, tmp_path)
    observed, errors = [], []
    def producer():
        try:
            wait_for(instance.output / "group.json", lambda s: s["workers"]["w0"]["status"] == "running")
            digest = publish(value, path, 0, sha256(path), "admit", [workers[1]])
            current = wait_for(instance.output / "group.json", lambda s:
                s["workers"]["w0"].get("cleanup_verified") and s["workers"].get("w1", {}).get("status") == "running")
            observed.append(current)
            assert not instance.lease.released
            digest = publish(value, path, 1, digest, "admit", [workers[2]])
            publish(value, path, 2, digest, "close")
        except BaseException as error:
            errors.append(error)
            instance.request_stop()
    thread = threading.Thread(target=producer)
    thread.start()
    result = instance.run()
    thread.join(timeout=10)
    assert not thread.is_alive() and not errors, errors
    assert result["status"] == "completed", result
    assert observed[0]["allocation_released_unix"] is None
    assert result["admission"]["next_sequence"] == 3 and result["admission"]["closed"]
    assert result["peak_parallel_workers"] >= 2
    assert result["workers"]["w2"]["started_unix"] < result["workers"]["w1"]["finished_unix"]
    for record in result["workers"].values():
        assert record["cleanup_verified"] and record["descendant_cleanup_verified"]
        assert record["remaining_descendants"] == []
        assert record["result_sha256"] == sha256(record["result_path"])
        assert result["allocation_released_unix"] >= record["cleanup_verified_unix"]
    chain = shared.read_admission_chain(value, path, result)
    assert len(chain["published_workers"]) == len(chain["acknowledged_workers"]) == 3
    assert chain["acknowledged_closed"] and chain["closed"]
    assert instance.lease.samples >= 2*3 and instance.lease.released
    memory = result["lifecycle_memory_observations"]
    assert memory["samples"] and memory["final_result_device_memory"]["w0"]["npu_reserved_peak_bytes"] == 1234


def test_live_controller_idle_group_waits_for_explicit_close(tmp_path):
    value, path, workers = request(tmp_path, [{"sleep": .03}])
    instance = runner(value, path, tmp_path)
    errors = []
    def producer():
        try:
            idle = wait_for(instance.output / "group.json", lambda s: s["admission"].get("idle_since_unix"))
            time.sleep(.08)  # Longer than the configured hint, with a live controller.
            assert not instance.lease.released and not idle["admission"]["closed"]
            publish(value, path, 0, sha256(path), "close")
        except BaseException as error:
            errors.append(error)
            instance.request_stop()
    thread = threading.Thread(target=producer)
    thread.start()
    result = instance.run()
    thread.join(timeout=5)
    assert not errors and result["status"] == "completed", (errors, result)
    assert result["admission"]["close_reason"] == "controller_close"


def test_dead_controller_prevents_start_and_releases_only_after_empty_drain(tmp_path):
    value, path, workers = request(tmp_path)
    value["dynamic_admission"]["controller_identity"]["start_ticks"] += 1
    atomic_json(path, value)
    instance = runner(value, path, tmp_path)
    result = instance.run()
    assert result["admission"]["close_reason"] == "controller_lost"
    assert result["workers"]["w0"]["status"] == "not_started"
    assert result["remaining_descendants"] == [] and instance.lease.released


def test_real_queue_adapter_does_not_wait_for_a_dead_controller(tmp_path, monkeypatch):
    value, path, _ = request(tmp_path)
    value["dynamic_admission"]["controller_identity"]["start_ticks"] += 1
    helper = shared.load_queue_helper()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(helper, "run_npu_smi", lambda *args: "0 0 0 7 Ascend910B\n")
    monkeypatch.setattr(helper, "attempt_schedule", lambda **kwargs: pytest.fail("dead controller attempted device admission"))
    monkeypatch.setattr(shared, "load_queue_helper", lambda: helper)
    output = tmp_path / "queued-group"
    output.mkdir()
    lease = shared.QueueLease(value, output, threading.Event())
    try:
        with pytest.raises(shared.GroupBlocked, match="controller_lost_while_queued"):
            lease.acquire()
    finally:
        lease.release("failed")
    assert not (tmp_path / "runtime/ascend-job-queue/jobs/shared-test.json").exists()


def test_controller_loss_drains_running_worker_without_starting_pending_one(tmp_path, monkeypatch):
    value, path, workers = request(tmp_path, [{"sleep": .25}, {"sleep": .1}])
    value["dynamic_admission"].update(stable_window_seconds=30,
        controller_identity={"pid": 2**29, "start_ticks": 77})
    atomic_json(path, value)
    publish(value, path, 0, sha256(path), "admit", [workers[1]])
    instance = runner(value, path, tmp_path)
    original = shared.proc_identity
    lost = threading.Event()
    def process_identity(pid, **kwargs):
        if pid == 2**29:
            return None if lost.is_set() else {"pid": pid, "start_ticks": 77}
        return original(pid, **kwargs)
    monkeypatch.setattr(shared, "proc_identity", process_identity)
    def controller():
        wait_for(instance.output / "group.json", lambda s: s["workers"]["w0"]["status"] == "running")
        lost.set()
    thread = threading.Thread(target=controller)
    thread.start()
    result = instance.run()
    thread.join(timeout=5)
    assert result["status"] == "completed", result
    assert result["admission"]["close_reason"] == "controller_lost"
    assert result["workers"]["w0"]["return_code"] == 0 and result["workers"]["w0"]["cleanup_verified"]
    assert result["workers"]["w1"]["status"] == "not_started"
    assert instance.lease.released


def test_dynamic_worker_orphan_is_removed_before_completion_and_unlock(tmp_path):
    value, path, workers = request(tmp_path, [{"orphan": True, "sleep": .08}])
    publish(value, path, 0, sha256(path), "close")
    instance = runner(value, path, tmp_path)
    result = instance.run()
    assert result["status"] == "failed" and "orphan" in result["reason"], result
    child = int((tmp_path / "worker-0/orphan.pid").read_text())
    assert shared.proc_identity(child) is None
    assert result["workers"]["w0"]["cleanup_verified"]
    assert result["allocation_released_unix"] >= result["workers"]["w0"]["cleanup_verified_unix"]
    assert result["remaining_descendants"] == [] and instance.lease.released


def test_valid_unacknowledged_tail_is_not_a_worker_identity_failure(tmp_path):
    value, path, workers = request(tmp_path)
    instance = runner(value, path, tmp_path)
    publish(value, path, 0, sha256(path), "admit", [workers[1]])
    chain = shared.read_admission_chain(value, path, instance.state)
    assert len(chain["published_workers"]) == 2 and len(chain["acknowledged_workers"]) == 1
    assert chain["next_sequence"] == 1 and not chain["closed"]


@pytest.mark.parametrize("damage", ["predecessor", "manifest_hash", "deadline", "sequence_gap", "duplicate_id",
                                    "output_overwrite", "task_hash", "argv_task", "input_overwrite", "total_limit",
                                    "capacity", "after_close", "initial_manifest", "ack_hash"])
def test_bad_or_changed_append_evidence_is_rejected(tmp_path, damage):
    value, path, workers = request(tmp_path)
    instance = runner(value, path, tmp_path)
    worker = copy.deepcopy(workers[1])
    if damage == "duplicate_id":
        worker = copy.deepcopy(workers[0])
    elif damage == "output_overwrite":
        worker["output"] = workers[0]["output"]
        worker["argv"][worker["argv"].index("--output") + 1] = worker["output"]
    elif damage == "task_hash":
        Path(worker["task_path"]).write_text('{}')
    elif damage == "argv_task":
        worker["argv"][worker["argv"].index("--task") + 1] = workers[0]["task_path"]
    elif damage == "input_overwrite":
        worker["output"] = str(tmp_path)
        worker["argv"][worker["argv"].index("--output") + 1] = worker["output"]
    elif damage == "capacity":
        worker["allowed_failure_classes"] = ["capacity_error"]
    elif damage == "total_limit":
        value["dynamic_admission"]["max_total_workers"] = 1
        atomic_json(path, value)
    elif damage == "initial_manifest":
        value["workers"][0]["argv"].append("unexpected")
    digest = publish(value, path, 0, sha256(path), "admit", [worker])
    command_path = Path(value["dynamic_admission"]["inbox_dir"]) / "00000000.json"
    command = json.loads(command_path.read_text())
    if damage in {"predecessor", "manifest_hash", "deadline"}:
        key = {"predecessor": "previous_sha256", "manifest_hash": "manifest_sha256", "deadline": "deadline_unix"}[damage]
        command[key] = command[key]+1 if damage == "deadline" else "0"*64
        atomic_json(command_path, command)
    elif damage == "sequence_gap":
        command_path.rename(command_path.with_name("00000001.json"))
    elif damage == "after_close":
        command.update(kind="close", workers=[])
        atomic_json(command_path, command)
        publish(value, path, 1, sha256(command_path), "admit", [worker])
    elif damage == "ack_hash":
        instance.refresh_admissions()
        command["note"] = "changed after acknowledgement"
        atomic_json(command_path, command)
    with pytest.raises(ValueError):
        shared.read_admission_chain(value, path, instance.state if damage == "ack_hash" else None)


def test_only_bound_recipe_rejection_is_independently_nonfatal(tmp_path):
    value, path, workers = request(tmp_path, [
        {"sleep": .05, "code": 1, "status": "recipe_rejected", "failure_class": "declared_optimizer_instability"},
        {"sleep": .05}])
    digest = publish(value, path, 0, sha256(path), "admit", [workers[1]])
    publish(value, path, 1, digest, "close")
    instance = runner(value, path, tmp_path)
    result = instance.run()
    assert result["status"] == "completed", result
    assert result["workers"]["w0"]["status"] == "allowed_failure"
    assert result["workers"]["w0"]["return_code"] == 1
    assert result["workers"]["w1"]["status"] == "completed"


@pytest.mark.parametrize("cancel", [True, False])
def test_initially_empty_ticket_releases_or_accepts_after_real_allocation(tmp_path, cancel):
    value, path, workers = request(tmp_path, [{"sleep":.1}])
    value.update(workers=[], dispatch_lease=True)
    atomic_json(path, value)
    instance = runner(value, path, tmp_path)
    if cancel:
        class QueuedLease(FakeLease):
            physical_id = None

            def acquire(self):
                self.physical_id = None
                assert self.cancelled()

            def release(self, status):
                assert status == "completed" and not self.handle
        instance.lease_factory = QueuedLease
        publish(value, path, 0, sha256(path), "close")
        result = instance.run()
        assert result["status"] == "completed" and result["cancelled_before_allocation"]
        assert result["queue_released_unix"] and not result["allocation_released_unix"]
        assert result["physical_device_seconds"] == 0 and result["workers"] == {}
    else:
        errors = []
        def dispatch():
            try:
                wait_for(instance.output/"group.json", lambda g: g.get("allocation_acquired_unix") and not g["workers"])
                digest = publish(value,path,0,sha256(path),"admit",workers)
                publish(value,path,1,digest,"close")
            except BaseException as error:
                errors.append(error)
                instance.request_stop()
        thread = threading.Thread(target=dispatch)
        thread.start()
        result = instance.run()
        thread.join(timeout=5)
        assert not errors and result["status"] == "completed", (errors,result)
        assert result["workers"]["w0"]["started_unix"] >= result["allocation_acquired_unix"]
        assert result["workers"]["w0"]["cleanup_verified"] and result["physical_device_seconds"] > 0
