import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tide_cloud_node as node_worker
import tide_cloud_submit as submit


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(submit.wire(obj))


class Region:
    def __init__(self):
        self.objects = {}

    def head(self, key):
        return {"exists": True} if key in self.objects else None

    def get_bytes(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def put_bytes(self, key, data):
        if self.objects.setdefault(key, data) != data:
            raise FileExistsError(key)


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    node = {"name": "tide-fixture", "id": "job-37", "prefix": "jobs/frozen"}
    work, mount, obs = tmp_path / "work", tmp_path / "mount", tmp_path / "obs" / node["name"]
    shared = mount / ".maflow-tide-fixture"
    native = shared / "tide-r3"
    queue = native / "runq"
    obs.mkdir(parents=True)
    sha, commit = "3" * 64, "frozen"
    release = {"commit": commit, "sha256": sha, "bytes": 101}
    dump(work / "tools/runq/.maflow-release.json", release)
    dump(work / "payload/tide_cloud_node.py", "synthetic worker bytes")
    dump(work / "job.json", {"name": node["name"], "nodes": 1, "runq": release,
                             "sfs_mounts": [str(mount)], "sfs_name": shared.name})
    dump(work / "payload/production.json", {"jobs": [node["name"]], "runq_commit": commit, "runq_sha256": sha,
                                           "files":{"tide_cloud_node.py": submit.relay.digest(work / "payload/tide_cloud_node.py")}})
    dump(work / "queue.json", {"writable_areas": [str(shared)], "deadline": time.time() + 7200})
    dump(work / "runq-project.json", {"mode": "project", "shared": str(queue), "deadline": time.time() + 7200})
    dump(queue / ".runq-cluster/allocation.json", {"workers": 1})
    exe = work / "bin/runq"
    exe.parent.mkdir()
    python = exe.parent / "original python"
    python.symlink_to(sys.executable)
    exe.write_text("#!/bin/sh\nexec " + shlex.join([str(python), "-I", "-B", str(work / "tools/runq/launcher.py")]) + ' "$@"\n')
    for k, v in {"MAFLOW_RUNQ": exe, "MAFLOW_RUNQ_COMMIT": commit, "MAFLOW_RUNQ_SHA256": sha}.items():
        monkeypatch.setenv(k, str(v))

    def entry(backend, attempt=None, **changes):
        desc = {"job": node["name"], "task": "model-" + backend, "attempt": attempt or ("r3-" + backend),
                "fingerprint": "a" * 64, "backend": backend, "threads": 3, "ram_mib": 20003, "priority": 10201,
                "hbm_mib": 59000 if backend == "npu" else 0, "reserve_chip": backend == "npu",
                "estimated_seconds": 17001, **changes}
        raw = submit.wire(desc)
        claim = {**{k: desc[k] for k in ("job", "task", "attempt", "fingerprint")},
                 "digest": hashlib.sha256(raw).hexdigest(), "transport": "ssh"}
        aid = claim["attempt"] + ".json"
        dump(obs / "inputs/authority" / aid, desc)
        dump(obs / "inputs/routes" / aid, submit.ticket(claim, node["id"]))
        return claim

    def batch(claims, run):
        return submit.remote_batch(node, claims, work, obs, shared, mount_check=lambda p: p in (obs, mount), runner=run)

    return SimpleNamespace(**locals())


def test_remote_batch_caches_exact_authority_and_reuses_the_frozen_argv(allocation):
    a = allocation
    gpu, cpu = a.entry("npu"), a.entry("cpu")
    requests = {}
    def run(cmd, **kwargs):
        key = cmd[cmd.index("--key") + 1]
        if key in requests:
            assert requests[key] == cmd
        requests[key] = cmd
        return SimpleNamespace(returncode=0, stdout=json.dumps({"key": key, "worker": 0}))
    for _ in range(2):
        result = a.batch([cpu, gpu], run)
        assert [result[c["attempt"]]["status"] for c in (cpu, gpu)] == ["submitted", "submitted"]
    for c in (cpu, gpu):
        argv = requests[c["attempt"]]
        cached = a.native / "envelopes" / (c["attempt"] + ".json")
        assert argv[:8] == [str(a.exe), "--json", "cluster", "submit", "--shared", str(a.queue), "--cwd", str(a.work / "payload")]
        assert argv[argv.index("--"):] == ["--", str(a.python), "-B", "-u", str(a.work / "payload/tide_cloud_node.py"),
                                            "worker", str(cached), c["digest"]]
        assert cached.read_bytes() == (a.obs / "inputs/authority" / cached.name).read_bytes()
        assert cached.stat().st_mode & 0o777 == 0o400
        assert [argv[argv.index(k) + 1] for k in ("--threads", "--ram-mib", "--priority", "--estimated-seconds")] == [
            "3", "20003", "10201", "17001"]
        if c == gpu:
            assert argv[argv.index("--hbm-mib") + 1] == "59000" and "--reserve-chip" in argv
        else:
            assert "--hbm-mib" not in argv and "--reserve-chip" not in argv
    assert not (a.obs / "inputs/dispatch").exists()


def test_mount_lag_invalid_resources_and_rejections_are_isolated(allocation, monkeypatch):
    a = allocation
    lag = a.entry("cpu", "lag")
    (a.obs / "inputs/authority/lag.json").write_bytes(b"{")
    bad = a.entry("npu", "invalid", hbm_mib=-1)
    conflict = a.entry("cpu", "conflict")
    ok = a.entry("npu", "good")
    def run(cmd, **kwargs):
        key = cmd[cmd.index("--key") + 1]
        if key == "conflict":
            return SimpleNamespace(returncode=2, stderr="key already used for a different cluster request", stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"key": key, "worker": 0}))
    result = a.batch([lag, bad, conflict, ok], run)
    assert {k: v["status"] for k, v in result.items()} == {
        "lag": "retry", "invalid": "rejected", "conflict": "rejected", "good": "submitted"}
    monkeypatch.setenv("MAFLOW_RUNQ_SHA256", "other-release")
    assert a.batch([ok], pytest.fail)[ok["attempt"]]["status"] == "retry"


def test_route_probe_checks_exact_worker_and_release_and_authenticates_the_returned_job(allocation, monkeypatch):
    a = allocation
    assert submit.verified_queue(a.node, a.work, a.obs, a.shared, lambda p: p in (a.mount, a.obs))[3] == str(a.python)
    returned = dict(job_id="other-job", node=0)
    env = {**os.environ, "MAFLOW_WORK": str(a.work), "MAFLOW_OBS": str(a.obs), "MAFLOW_SHARED": str(a.shared),
           "MAFLOW_RUN": a.node["name"], "MAFLOW_WORKER": "worker-0"}
    native_run = subprocess.run
    def run(cmd, **kwargs):
        assert kwargs["timeout"] == 35
        mounts = [str(a.mount), str(a.obs)]
        bootstrap = "import os;os.path.ismount=lambda p:os.fspath(p) in " + repr(mounts) + ";" + cmd[-2]
        actual = native_run([sys.executable, "-I", "-B", "-c", bootstrap, cmd[-1]], env=env, **kwargs)
        return SimpleNamespace(returncode=actual.returncode, stderr="", stdout=json.dumps({
            **returned, "exit_code": actual.returncode, "stdout": actual.stdout, "stderr": actual.stderr}))
    monkeypatch.setattr(subprocess, "run", run)
    assert not submit.ssh_ready(a.node)
    returned.update(job_id=a.node["id"])
    assert submit.ssh_ready(a.node)
    returned.update(node=1)
    assert not submit.ssh_ready(a.node)
    returned.update(node=0)
    (a.work / "payload/tide_cloud_node.py").write_text("different worker")
    assert not submit.ssh_ready(a.node)
    result = a.batch([a.entry("cpu")], pytest.fail)
    assert result["r3-cpu"]["status"] == "retry" and "frozen worker" in result["r3-cpu"]["reason"]


def test_one_authenticated_host_call_per_batch_and_lost_response_replays_same_route(allocation, monkeypatch):
    from tide_cloud_allocation import Store
    a, region = allocation, Region()
    claims = [a.entry("npu"), a.entry("cpu")]
    monkeypatch.setattr(Store, "named", lambda site: region)
    sent = []
    def run(cmd, **kwargs):
        sent.append(cmd)
        assert cmd[3:11] == ["job", "exec", "--json", "--node", "0", "guiyang/job-37", "--", "python3"]
        if len(sent) == 1:
            raise subprocess.TimeoutExpired("ssh", 5)
        response = {c["attempt"]: submit.reply("submitted", "accepted") for c in claims}
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "job_id": a.node["id"], "node": 0, "exit_code": 0, "stdout": json.dumps(response)}))
    monkeypatch.setattr(subprocess, "run", run)
    assert {v["status"] for v in submit.submit_batch(a.node, claims).values()} == {"retry"}
    assert len(region.objects) == 2 and all("/routes/" in key for key in region.objects)
    assert {v["status"] for v in submit.submit_native(a.node, [dict(c, transport="obs") for c in claims], region).values()} == {"rejected"}
    assert {v["status"] for v in submit.submit_batch(a.node, claims).values()} == {"submitted"}
    assert len(sent) == 2 and sent[0] == sent[1]


def test_native_fallback_is_idempotent_handles_uncertain_put_and_prior_legacy_envelope(allocation):
    a = allocation
    native = dict(a.entry("cpu"), transport="obs")
    class FlakyRegion(Region):
        def put_bytes(self, key, data):
            super().put_bytes(key, data)
            raise ConnectionResetError("lost conditional write reply")
    region = FlakyRegion()
    raw = (a.obs / "inputs/authority/r3-cpu.json").read_bytes()
    prefix = a.node["prefix"] + "/inputs/"
    region.objects[prefix + "authority/r3-cpu.json"] = raw
    for _ in range(2):
        assert submit.submit_native(a.node, [native], region)["r3-cpu"]["status"] == "submitted"
    assert region.objects[prefix + "dispatch/r3-cpu.json"] == raw
    ssh = dict(native, transport="ssh")
    assert submit.prepare(a.node, [ssh], region, "ssh")[1]["r3-cpu"]["status"] == "rejected"
    del region.objects[prefix + "routes/r3-cpu.json"]
    assert submit.prepare(a.node, [ssh], region, "ssh")[1]["r3-cpu"]["status"] == "submitted"
    assert prefix + "routes/r3-cpu.json" not in region.objects


def test_native_node_preserves_its_queue_through_transient_mount_submit_status_and_rejection_errors(allocation, monkeypatch):
    a = allocation
    monkeypatch.setenv("MAFLOW_RUN", a.node["name"])
    monkeypatch.setenv("MAFLOW_WORK", str(a.work))
    monkeypatch.setenv("MAFLOW_RUNQ_QUEUE_FILE", str(a.work / "runq-project.json"))
    monkeypatch.setattr(node_worker, "ROOT", a.work / "payload")
    release = {"runq_commit": a.commit, "runq_sha256": a.sha, "deadline_unix": time.time() + 3600}
    dump(node_worker.ROOT / "production.json", release)
    dispatch = a.obs / "inputs/dispatch"
    for claim in (a.entry("cpu", "good"), a.entry("npu", "flaky"), a.entry("npu", "invalid", priority=None)):
        aid = claim["attempt"] + ".json"
        dump(dispatch / aid, json.loads((a.obs / "inputs/authority" / aid).read_bytes()))
    (dispatch / "partial.json").write_text("{")
    observed = {"list": 0, "good": 0, "flaky": 0, "status": 0, "rejection": 0, "finish": 0}
    original = Path.iterdir
    def listdir(path):
        if path == dispatch:
            observed["list"] += 1
            if observed["list"] == 1:
                raise OSError("stale handle")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", listdir)
    agent = SimpleNamespace(code=None)
    agent.poll = lambda: agent.code
    def wait(timeout):
        agent.code = 0
        return 0
    agent.wait = wait
    agent.terminate = lambda: pytest.fail("a courier error stopped the queue")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: agent)
    clock = SimpleNamespace(at=time.time())
    def sleep(n):
        clock.at += n
        assert clock.at < release["deadline_unix"] - 900, observed
    monkeypatch.setattr(node_worker, "time", SimpleNamespace(time=lambda: clock.at, sleep=sleep))
    def call(cmd, **kwargs):
        if "submit" in cmd:
            key = cmd[cmd.index("--key") + 1]
            observed[key] += 1
            if key == "flaky" and observed[key] == 1:
                raise subprocess.CalledProcessError(1, cmd)
            assert cmd[-1] == hashlib.sha256((dispatch / (key + ".json")).read_bytes()).hexdigest()
            return json.dumps({"key": key, "worker": 0})
        if "finish" in cmd:
            assert (a.obs / "control/drain-tide.json").is_file()
            observed["finish"] += 1
            if observed["finish"] == 1:
                raise subprocess.TimeoutExpired(cmd, 30)
            return "{}"
        observed["status"] += 1
        if observed["status"] == 1:
            return '{"tasks":[{}]}'
        if observed["rejection"] >= 2 and observed["flaky"] == 2:
            dump(a.obs / "control/drain-tide.json", {})
        return '{"tasks":[],"workers":[0]}'
    monkeypatch.setattr(subprocess, "check_output", call)
    rejections = []
    def deliver(path, obs, name):
        if "/receipts/" in name:
            observed["rejection"] += 1
            rejections.append(path.read_bytes())
            if len(rejections) == 1:
                raise OSError("OBS mount temporarily unavailable")
        (obs / name).parent.mkdir(parents=True, exist_ok=True)
        (obs / name).write_bytes(path.read_bytes())
    monkeypatch.setattr(node_worker, "deliver", deliver)
    monkeypatch.setattr(node_worker, "export", deliver)
    assert node_worker.serve(a.obs, a.native) == 0
    assert observed["good"] == 1 and observed["flaky"] == observed["finish"] == 2
    assert len(rejections) == 2 and rejections[0] == rejections[1]
    assert json.loads(rejections[0])["status"] == "failed" and (a.obs / "products/queue-receipt.json").is_file()
