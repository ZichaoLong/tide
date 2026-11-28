"""Recover the native output of one frozen ModelArts attempt through its own allocation."""

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

NATIVE_FILES = (
    "result.json", "project-verdict.json", "runtime-first-update.json", "steps.jsonl",
    "checkpoint.json", "health.jsonl", "work.jsonl", "validation.jsonl",
    "startup-memory.json", "failure.json",
)
END = {"cancelled", "blocked", "failed", "completed", "unconfirmed"}
QUEUE_FIELDS = (
    "key", "state", "id", "pid", "ticks", "generation", "rc", "reason", "backend", "physical_card",
    "chip", "worker", "instance", "submitted", "started", "finished", "usage", "runtime_version", "at",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def optional(path):
    try:
        return read(path)
    except FileNotFoundError:
        return None


def member(root, name):
    """Restrict both lexical and resolved paths to this exact attempt or mounted job."""
    dest = root / name
    require(dest == dest.resolve() and dest.is_relative_to(root), "unexpected path or symbolic link")
    return dest


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bound(rec, identity):
    return all(rec.get(k) == v for k, v in identity.items())


def existing_body(path, expected):
    try:
        return path.stat().st_size == expected["size"] and digest(path) == expected["sha256"]
    except FileNotFoundError:
        return False


def publish(source, obs, name, expected=None):
    # The original worker and every relay share the same SFS lock. Stage new bytes
    # before publishing the final key so an interrupted transfer can be retried.
    dest = member(obs, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    stage = None
    try:
        if not dest.exists():
            stage = tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".tide-relay-", suffix=".part", delete=False)
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                h.update(block)
                md5.update(block)
                size += len(block)
                if stage:
                    stage.write(block)
        spec = {"path": name, "size": size, "sha256": h.hexdigest(), "content_md5": md5.hexdigest()}
        require(expected is None or spec["sha256"] == expected, "native artifact checksum differs: " + name)
        if stage:
            stage.close()
            if not dest.exists():
                try:
                    os.rename(stage.name, dest)
                except OSError as exc:
                    if exc.errno not in {errno.ENOTSUP, errno.ENOSYS, errno.EXDEV, errno.EPERM}:
                        raise
                    # An object mount may only support create/write. We own any
                    # destination opened with x and can retract a failed copy.
                    if not dest.exists():
                        with dest.open("xb") as target:
                            try:
                                with source.open("rb") as src:
                                    shutil.copyfileobj(src, target, 2**20)
                            except BaseException:
                                target.close()
                                dest.unlink(missing_ok=True)
                                raise
        require(existing_body(dest, spec), "conflicting published artifact: " + name)
        return spec
    finally:
        if stage:
            stage.close()
            Path(stage.name).unlink(missing_ok=True)


def native_paths(area, descriptor):
    output = member(area, "output")
    files, checksums = {}, {}
    for name in NATIVE_FILES:
        p = member(output, name)
        if p.is_file():
            files[name] = p
    log = member(area, "worker.log")
    if log.is_file():
        files["worker.log"] = log
    result = optional(member(output, "result.json"))
    verdict = optional(member(output, "project-verdict.json"))
    if result:
        require(result.get("task_sha256") == descriptor["fingerprint"], "native result fingerprint differs")
        for field, name in (("checkpoint", "checkpoint.pt"), ("failure_capture", "failure.pt")):
            obj = result.get(field) or {}
            if obj.get("path"):
                inn = Path(obj["path"])
                require(inn.is_relative_to(output), "native artifact outside the attempt")
                p = member(output, inn.relative_to(output))
                if name == "checkpoint.pt" or p.is_file():
                    checksums[name] = obj["sha256"]
                    if p.is_file():
                        files[name] = p
    if verdict:
        require(bound(verdict, {"task": descriptor["task"], "fingerprint": descriptor["fingerprint"]})
                and verdict["status"] in {"failed", "completed", "interrupted"}, "native verdict identity differs")
        if verdict["status"] in {"completed", "interrupted"}:
            require("checkpoint.pt" in checksums and result and verdict.get("checkpoint", {}).get("sha256")
                    == result["checkpoint"]["sha256"], "native verdict lacks its checkpoint")
    return files, checksums, verdict if result else None


def relay_locked(area, obs, claim, queue, identity):
    source = optional(member(area, "receipt.json"))
    if source is not None:
        require(bound(source, identity) and source.get("status") in {"completed", "interrupted", "failed"},
                "SFS terminal receipt identity differs")
    files, checksums, verdict = native_paths(area, claim)
    if not source and not verdict:
        return {"status": "incomplete", "reason": "SFS attempt has no terminal receipt or native verdict/result"}
    status = verdict["status"] if verdict else source["status"]
    name = claim["attempt"]
    products = {}
    for key, path in files.items():
        prior = (source or {}).get("artifacts", {}).get(key, {})
        products[key] = publish(path, obs, f"products/attempts/{name}/{key}", checksums.get(key) or prior.get("sha256"))
    if "runtime-first-update.json" in files and claim.get("canary"):
        publish(files["runtime-first-update.json"], obs, f"products/first-update/{name}.json")
    # A retained output can outlive its SFS source. Accept only the exact OBS bytes bound by the receipt.
    for key, artifact in (source or {}).get("artifacts", {}).items():
        if key in products:
            require(products[key] == artifact, "existing receipt artifact differs: " + key)
        else:
            path = f"products/attempts/{name}/{key}"
            require(key in {*NATIVE_FILES, "worker.log", "failure.pt", "checkpoint.pt"} and artifact["path"] == path
                    and existing_body(member(obs, path), artifact), "missing receipt artifact: " + key)
            products[key] = artifact
    if status in {"completed", "interrupted"}:
        require({"project-verdict.json", "result.json", "checkpoint.pt"} <= products.keys(),
                "terminal claim lacks a native verdict, result or checkpoint")
        # A sealed SFS receipt may outlive its local output directory. Verify the
        # retained original artifact bodies and their native cross-references.
        native_verdict = verdict or read(member(obs, products["project-verdict.json"]["path"]))
        native_result = read(member(obs, products["result.json"]["path"]))
        expected = products["checkpoint.pt"]["sha256"]
        require(bound(native_verdict, {"task": claim["task"], "fingerprint": claim["fingerprint"], "status": status})
                and native_result.get("task_sha256") == claim["fingerprint"]
                and native_verdict.get("checkpoint", {}).get("sha256") == native_result.get("checkpoint", {}).get("sha256") == expected,
                "retained terminal evidence is inconsistent")
    if source and source["status"] == status and source.get("artifacts") == products:
        data = (area / "receipt.json").read_bytes()
    else:
        observed = {k: queue[k] for k in QUEUE_FIELDS if k in queue}
        rec = {**identity, "schema": "tide-cloud-result/v1", "status": status, "artifacts": products,
               "runq_observation": observed, "recovered_from": "native verdict and result"}
        for k in ("started", "finished"):
            value = (source or {}).get(k)
            rec[k] = value if value is not None else queue.get(k)
        rec["device"] = (source or {}).get("device") or (verdict or {}).get("device")
        rec["board"] = (source or {}).get("board") if (source or {}).get("board") is not None else queue.get("physical_card")
        chip = queue.get("chip")
        if isinstance(chip, dict):
            chip = chip.get("physical")
        rec["chip"] = (source or {}).get("chip") if (source or {}).get("chip") is not None else chip
        if verdict:
            rec["native_process"] = {k: verdict[k] for k in ("worker_pid", "ticks", "device") if k in verdict}
        if source:
            rec["prior_sfs_receipt"] = {"status": source["status"], "sha256": digest(area / "receipt.json")}
        data = (json.dumps(rec, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    for directory in ("receipts", "reconciled"):
        key = f"products/{directory}/{name}.json"
        dest = member(obs, key)
        prior = optional(dest)
        if prior is not None:
            if bound(prior, identity) and prior.get("status") == status and prior.get("artifacts") == products:
                return {"status": "published", "receipt_key": key, "sha256": digest(dest)}
            require(directory == "receipts" and bound(prior, identity) and prior.get("status") == "failed",
                    "conflicting original receipt bound to other work or a different scientific outcome")
            continue
        with tempfile.NamedTemporaryFile() as f:
            f.write(data)
            f.flush()
            item = publish(Path(f.name), obs, key)
        return {"status": "published", "receipt_key": key, "sha256": item["sha256"]}
    raise ValueError("conflicting reconciliation receipt")


def reconcile(claim, work, obs, shared, mount_check=os.path.ismount):
    """Inspect verified mounts and the queue's actual durable request before taking the worker lock."""
    name, attempt = claim["job"], claim["attempt"]
    require(all(isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,190}", s)
                for s in (name, attempt, claim["task"])), "invalid attempt identity")
    cfg = read(member(work, "job.json"))
    release = read(member(work, "payload/production.json"))
    qcfg = read(member(work, "queue.json"))
    require(name == cfg["name"] and cfg["nodes"] == 1 and name in release["jobs"]
            and cfg["runq"]["commit"] == release["runq_commit"], "incorrect allocation")
    writable = [Path(p) for p in cfg["sfs_mounts"] if p not in cfg.get("sfs_read_only", [])]
    require(writable and shared == writable[0] / cfg["sfs_name"] and shared == shared.resolve()
            and str(shared) == qcfg["writable_areas"][0] and mount_check(writable[0])
            and obs == obs.resolve() and obs.name == name and mount_check(obs), "unverified job OBS or writable SFS mount")
    require(shared.is_dir() and shared.stat().st_uid == os.getuid(), "unexpected shared directory")
    native = member(shared, "tide-r3")
    cluster = member(native, "runq/.runq-cluster")
    token = hashlib.sha256(attempt.encode()).hexdigest() + ".json"
    req = read(member(cluster, "requests/" + token))["spec"]
    envelope = member(obs, "inputs/dispatch/" + attempt + ".json")
    payload = envelope.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == claim["digest"], "OBS dispatch bytes changed")
    d = json.loads(payload)
    require(bound(d, {k: claim[k] for k in ("job", "task", "attempt", "fingerprint")}), "incorrect dispatch envelope")
    argv = req["options"]["argv"]
    require(req["key"] == attempt and req.get("worker") == 0 and argv[-4:] == [
        str(member(work, "payload/tide_cloud_node.py")), "worker", str(envelope), claim["digest"]
    ], "queue request does not match the frozen descriptor")
    queue = optional(member(cluster, "receipts/" + token))
    if not queue or queue.get("state") not in END:
        return {"status": "busy", "reason": "local runq has not sealed a terminal receipt"}
    require(queue.get("key") == attempt and queue.get("worker", 0) == 0, "local runq key changed")
    observed = claim.get("queue_task") or {}
    require(all(observed.get(k) is None or queue.get(k) == observed[k] for k in ("key", "worker", "instance", "id", "generation")),
            "queue generation or process identity changed")
    identity = {"job": name, "task": claim["task"], "attempt": attempt, "fingerprint": claim["fingerprint"],
                "descriptor_sha256": claim["digest"], "runq": release["runq_commit"]}
    area = member(native, "attempts/" + attempt)
    if not area.is_dir():
        require(not area.exists(), "unexpected SFS attempt path")
        prelaunch = {"cancelled before assignment", "cancelled before admission", "deadline before assignment",
                     "cancelled by user", "worker outside allocation", "dependency failed"}
        absent = (queue["state"] in {"cancelled", "failed", "blocked"} and queue.get("reason") in prelaunch
                  and all(queue.get(k) is None for k in ("started", "pid", "ticks")))
        return {"status": "never_started" if absent else "incomplete", "reason": "SFS attempt path absent on the verified mount",
                "queue": {k: queue[k] for k in QUEUE_FIELDS if k in queue}}
    lock = member(area, "worker.lock")
    if not lock.is_file():
        return {"status": "incomplete", "reason": "SFS attempt exists without its original worker lock"}
    with lock.open("rb") as fp:
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "reason": "original worker holds its SFS lock"}
        return relay_locked(area, obs, d, queue, identity)


def recover(node, claim, queue_task):
    """Return a single bounded SSH observation for this job and attempt."""
    from tide_cloud_allocation import MAFLOW
    try:
        keys = {k: claim[k] for k in ("job", "task", "attempt", "fingerprint", "digest")}
        require(keys["job"] == node["name"], "claim belongs to a different ModelArts job")
        keys.update(job_id=node["id"], queue_task={k: (queue_task or {})[k] for k in QUEUE_FIELDS if k in (queue_task or {})})
        cmd = [sys.executable, "-B", str(MAFLOW), "job", "exec", "--json", "--node", "0", "guiyang/" + node["id"], "--",
               "python3", "-I", "-B", "-c", Path(__file__).read_text(), json.dumps(keys)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        require(r.returncode == 0, "ModelArts SSH command unavailable: " + r.stderr[-400:])
        answer = json.loads(r.stdout)
        require(answer["job_id"] == node["id"] and answer["exit_code"] == 0, "verified job execution did not complete")
        value = json.loads(answer["stdout"])
        require(value["status"] in {"published", "never_started", "incomplete", "busy", "unavailable"}, "unexpected relay reply")
        return value
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


if __name__ == "__main__":
    try:
        claim = json.loads(sys.argv[1])
        require(os.environ.get("MAFLOW_RUN") == claim["job"] and os.environ.get("MAFLOW_WORKER") == "worker-0",
                "SSH session belongs to another allocation")
        for name in ("MA_JOB_ID", "MODELARTS_JOB_ID"):
            value = os.environ.get(name, "")
            if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
                require(value.lower() == claim["job_id"].lower(), "platform identity differs")
        result = reconcile(claim, Path(os.environ["MAFLOW_WORK"]), Path(os.environ["MAFLOW_OBS"]), Path(os.environ["MAFLOW_SHARED"]))
    except (ValueError, TypeError, KeyError) as exc:
        result = {"status": "incomplete", "reason": str(exc)}
    except (OSError, IndexError) as exc:
        result = {"status": "unavailable", "reason": str(exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
