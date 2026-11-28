import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tide_cloud_relay as relay


def store(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def frozen(tmp_path):
    job, task, attempt = "tide-job", "model-task", "r3-digest-1"
    work, mount, obs = tmp_path / "work", tmp_path / "mount", tmp_path / "obs" / job
    share = mount / (".maflow-" + job)
    share.mkdir(parents=True)
    obs.mkdir(parents=True)
    (work / "payload").mkdir(parents=True)
    store(work / "job.json", {"name": job, "nodes": 1, "runq": {"commit": "frozen"},
                              "sfs_mounts": [str(mount)], "sfs_name": share.name})
    store(work / "payload/production.json", {"jobs": [job], "runq_commit": "frozen"})
    store(work / "queue.json", {"writable_areas": [str(share)]})
    descriptor = {"job": job, "task": task, "attempt": attempt, "fingerprint": "a" * 64, "canary": True}
    envelope = obs / "inputs/dispatch" / (attempt + ".json")
    store(envelope, descriptor)
    claim = {**descriptor, "digest": relay.digest(envelope)}
    cluster = share / "tide-r3/runq/.runq-cluster"
    token = hashlib.sha256(attempt.encode()).hexdigest() + ".json"
    store(cluster / "requests" / token, {"spec": {"key": attempt, "worker": 0, "options": {"argv": [
        "python", "-u", str(work / "payload/tide_cloud_node.py"), "worker", str(envelope), claim["digest"]]}}})
    q = cluster / "receipts" / token
    queued = {"key": attempt, "state": "cancelled", "reason": "cancelled before assignment"}
    store(q, queued)
    def run(**kwargs):
        return relay.reconcile(claim, work, obs, share, mount_check=lambda p: p in {mount, obs}, **kwargs)
    return claim, obs, share / "tide-r3/attempts" / attempt, q, run


def test_absence_requires_verified_terminal_without_a_start_and_the_exact_request(frozen):
    claim, obs, area, q, run = frozen
    assert run()["status"] == "never_started"
    store(q, {"key": claim["attempt"], "state": "unconfirmed"})
    assert run()["status"] == "incomplete"
    store(q, {"key": claim["attempt"], "state": "running", "pid": 5678})
    assert run()["status"] == "busy"
    store(q, {"key": claim["attempt"], "state": "failed", "reason": "out of memory", "started": 12})
    assert run()["status"] == "incomplete"
    assert not area.exists()
    (obs / "inputs/dispatch" / (claim["attempt"] + ".json")).write_text("{}")
    with pytest.raises(ValueError, match="dispatch bytes changed"):
        run()


def test_reconciliation_preserves_failure_and_uses_only_finished_native_outputs(frozen):
    claim, obs, area, q, run = frozen
    area.mkdir(parents=True)
    lock = area / "worker.lock"
    lock.touch()
    store(q, {"key": claim["attempt"], "state": "failed", "id": 9, "started": 100, "finished": 200,
              "pid": 9123, "ticks": 610, "generation": 1, "physical_card": 0, "chip": {"physical": 0},
              "usage": {"hbm_peak": 104000}, "worker": 0})
    path = area / "output/checkpoint.pt"
    path.parent.mkdir()
    path.write_bytes(b"model")
    checkpoint = {"path": str(path), "sha256": relay.digest(path), "size_bytes": 5}
    verdict = {"task": claim["task"], "fingerprint": claim["fingerprint"], "status": "completed",
               "worker_pid": 1234, "ticks": 405, "device": "npu:0", "checkpoint": checkpoint}
    store(area / "output/result.json", {"task_sha256": claim["fingerprint"], "checkpoint": checkpoint})
    store(area / "output/project-verdict.json", verdict)
    store(area / "output/runtime-first-update.json", {"task": claim["task"]})
    ident = {k: claim[k] for k in ("task", "job", "attempt", "fingerprint")}
    failed = {**ident, "schema": "tide-cloud-result/v1", "status": "failed", "runq": "frozen",
              "descriptor_sha256": claim["digest"], "artifacts": {}, "started": 105, "finished": 190}
    store(area / "receipt.json", failed)
    original = obs / "products/receipts" / (claim["attempt"] + ".json")
    store(original, failed)
    before = original.read_bytes()
    with lock.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert run()["status"] == "busy"
    result = run()
    assert result["status"] == "published"
    assert result["receipt_key"] == "products/reconciled/" + claim["attempt"] + ".json"
    assert original.read_bytes() == before and relay.read(area / "receipt.json") == failed
    rec = relay.read(obs / result["receipt_key"])
    assert (rec["status"], rec["started"], rec["finished"], rec["board"], rec["chip"]) == ("completed", 105, 190, 0, 0)
    assert rec["native_process"]["worker_pid"] == 1234 and rec["runq_observation"]["pid"] == 9123
    assert rec["runq_observation"]["usage"] == {"hbm_peak": 104000}
    assert (obs / rec["artifacts"]["checkpoint.pt"]["path"]).read_bytes() == b"model"
    assert (obs / "products/first-update" / (claim["attempt"] + ".json")).is_file()
    assert run() == result


def test_foreign_checkpoint_is_never_relayed(frozen):
    c, obs, area, q, run = frozen
    area.mkdir(parents=True)
    (area / "worker.lock").touch()
    saved = area.parent / "other.pt"
    saved.write_bytes(b"other")
    store(area / "output/result.json", {"task_sha256": c["fingerprint"], "checkpoint": {
        "path": str(saved), "sha256": relay.digest(saved)}})
    with pytest.raises(ValueError, match="outside the attempt"):
        run()
    assert not (obs / "products").exists()


def test_host_treats_ssh_failure_as_unknown(monkeypatch):
    def lost(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 1)
    monkeypatch.setattr(subprocess, "run", lost)
    c = {"job": "job", "task": "task", "attempt": "attempt", "fingerprint": "fp", "digest": "digest"}
    assert relay.recover({"name": "job", "id": "id"}, c, {})["status"] == "unavailable"


def test_republish_exact_sealed_receipt_after_sfs_outputs_have_been_collected(frozen):
    claim, obs, area, queue_path, run = frozen
    area.mkdir(parents=True)
    (area / "worker.lock").touch()
    store(queue_path, {"key": claim["attempt"], "state": "completed"})
    checkpoint = {"sha256": hashlib.sha256(b"already verified checkpoint").hexdigest()}
    bodies = {
        "checkpoint.pt": b"already verified checkpoint",
        "result.json": json.dumps({"task_sha256": claim["fingerprint"], "checkpoint": checkpoint}).encode(),
        "project-verdict.json": json.dumps({"task": claim["task"], "fingerprint": claim["fingerprint"],
                                             "status": "completed", "checkpoint": checkpoint}).encode(),
    }
    artifacts = {}
    for name, body in bodies.items():
        key = f"products/attempts/{claim['attempt']}/{name}"
        file = obs / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(body)
        artifacts[name] = {"path": key, "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                           "content_md5": hashlib.md5(body, usedforsecurity=False).hexdigest()}
    identity = {k: claim[k] for k in ("task", "job", "attempt", "fingerprint")}
    store(area / "receipt.json", {**identity, "descriptor_sha256": claim["digest"], "runq": "frozen",
                                  "status": "completed", "started": 10, "finished": 12, "artifacts": artifacts})
    result = run()
    assert result["status"] == "published"
    assert (obs / result["receipt_key"]).read_bytes() == (area / "receipt.json").read_bytes()
