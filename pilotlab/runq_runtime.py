"""Detached units hold their own device locks and execute one immutable argv."""
from __future__ import annotations

import array
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

from .io import atomic_json
from .runq_devices import admission, external, install_predicate, process_unit
from .runq_place import decoded
from .runq_store import connect, event, lookup, packed, prefix, setting, sockpath, unit, wake, write
from .scheduler import child_subreaper, proc_identity
from .shared_npu3 import QueueLease


def service(root, name, args, *, source=None, deadline=None, restart=False):
    source = Path(source or Path(__file__).resolve().parents[1])
    log = root / (name + ".log")
    cmd = ["systemd-run", "--user", "--collect", "--quiet", "--service-type=exec", "--slice=background.slice",
           "--unit="+name, "--working-directory="+str(source), "--setenv=PYTHONPATH="+str(source),
           "--setenv=PYTHONDONTWRITEBYTECODE=1", "--property=KillMode=control-group", "--property=TimeoutStopSec=90",
           "--property=StandardOutput=append:"+str(log), "--property=StandardError=append:"+str(log)]
    if deadline:
        cmd += ["--property=RuntimeMaxSec="+str(max(1, int(deadline-time.time()+1)))]
    if restart:
        cmd += ["--property=Restart=on-failure", "--property=RestartSec=1"]
    return subprocess.run([*cmd, sys.executable, "-m", "pilotlab.runq", "--root", str(root), *args],
                          capture_output=True, text=True, timeout=20)


def state_of(name):
    result = subprocess.run(["systemctl", "--user", "show", name, "-p", "ActiveState", "--value"],
                            capture_output=True, text=True, timeout=8)
    if result.stdout.strip():
        return result.stdout.strip()
    if "not found" in result.stderr:
        return "inactive"
    raise RuntimeError(result.stderr.strip())


def change(db, ident, state, reason, **fields):
    values = dict(state=state, reason=reason, **fields)
    db.execute("UPDATE jobs SET "+",".join(k+"=?" for k in values)+" WHERE id=?", (*values.values(),ident))
    event(db, ident, state, reason)


def defer(db, job, reason):
    change(db, job["id"], "queued", reason, lease=None, backend=None, assigned=None, tried=None,
           wrapper=None, pid=None, ticks=None)


def allowed(db, root, job, lease):
    others = [decoded(r, root) for r in db.execute("""SELECT * FROM jobs WHERE lease=? AND id!=?
                     AND state IN ('reserved','starting','running')""", (lease["id"], job["id"]))]
    # Its own wrapper has no driver allocation until it receives the lock.
    permits = [json.loads(job["grant"] or "null")] if job["manual"] else [
        j["permission"] for j in others if j["manual"] and j["fill"] and j["state"] == "running"]
    if not job["manual"] and lease["role"] != "auto" and not any(j["manual"] and j["state"] == "running" for j in others):
        return "manual device is draining"
    samples = setting(db,"samples",[])
    reason = admission(samples, lease["card"], others, json.loads(job["spec"])["hbm"], permits)
    if not reason:
        card = samples[-1]["cards"][str(lease["card"])]
        event(db,job["id"],"device_admitted",packed({"card":lease["card"],"lease":lease["id"],
              "sample_times":[s["at"] for s in samples[-2:]],"used_bytes":card["used"],
              "external":external(card,{j["unit"] for j in others})}))
    return reason


def serve_fd(db, root, lease_id, listener, resource):
    peer, _ = listener.accept()
    with peer:
        peer.settimeout(2)
        pid, uid, _ = struct.unpack("3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid():
            return
        req = json.loads(peer.recv(1024))
        with write(db):
            job = lookup(db, req["job"])
            lease = db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
            if (job["state"] != "starting" or job["wrapper"] != pid or job["lease"] != lease_id
                    or job["generation"] != req["generation"] or lease["state"] != "held"
                    or process_unit(pid) != unit(root, "job", job["id"], job["generation"])):
                peer.send(b"stale allocation")
                return
            reason = allowed(db, root, job, lease)
            if reason:
                defer(db, job, reason)
                peer.send(reason.encode())
                return
            peer.sendmsg([b"ok"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i",resource.pass_fds()))])


def hold_card(root, ident):
    db = connect(root)
    row = db.execute("SELECT * FROM leases WHERE id=?", (ident,)).fetchone()
    if not row or row["state"] != "opening":
        return
    job = lookup(db, row["trigger_job"])
    end = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: end.set())
    directory = root / f"lease-{ident}"
    directory.mkdir(mode=0o700, exist_ok=True)
    # A pinned ticket that loses capacity yields its place in the cooperative
    # queue. It is resubmitted on fresh measurements, never bypasses the head.
    manifest = dict(job_id=f"{prefix(root)}-lease-{ident}", candidate_ids=[row["card"]], priority=0,
                    deadline_unix=json.loads(job["spec"]).get("deadline") or time.time()+86400)
    lease = QueueLease(manifest, directory, end)
    helper = lease.helper
    samples = install_predicate(helper, row["card"], json.loads(row["grant"]), json.loads(job["spec"])["hbm"], bool(job["manual"]))
    def emit(state, **fields):
        helper.emit(state,directory/"queue.json",job_id=manifest["job_id"],project="runq",
                    priority=0,npu_count=1,helper=lease.helper_identity,**fields)

    reason = None
    path = Path(sockpath(root, "lease-"+str(ident)))
    try:
        lease.job_lock, _ = helper.register_job(lease.queue_dir, job_id=manifest["job_id"], project="runq",
                        priority=0, npu_count=1, candidate_ids={row["card"]})
        while not end.is_set():
            if time.time() >= manifest["deadline_unix"]:
                reason = "trigger deadline"
                break
            trigger = lookup(db, row["trigger_job"])
            if (trigger["state"] != "queued" or trigger["target"] not in ("auto", "npu:"+str(row["card"]))
                    or trigger["manual"] != job["manual"] or trigger["grant"] != job["grant"]):
                reason = "trigger changed"
                break
            samples.clear()
            before = time.time()
            decision = helper.attempt_schedule(queue_dir=lease.queue_dir, device_lock_dir=lease.device_dir,
                job_id=manifest["job_id"], mapping=lease.mapping, candidate_ids={row["card"]}, npu_count=1,
                max_hbm_mib=4096, max_aicore_percent=5, sample_seconds=3, stop_event=end)
            if decision.physical_ids:
                lease.device_locks, lease.physical_id = decision.device_locks, decision.physical_ids[0]
                break
            reason = decision.reason
            emit("queued", reason=reason, queue_position=decision.queue_position, head_job_id=decision.head_job_id)
            if reason != "waiting_turn":
                break
            end.wait(1)
        if not lease.device_locks:
            return
        path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(64)
            listener.settimeout(.5)
            with write(db):
                db.execute("UPDATE leases SET state='held',acquired=?,pid=?,ticks=?,reason=NULL WHERE id=?",
                           (before, os.getpid(),proc_identity(os.getpid())["start_ticks"],ident))
            emit("running", physical_ids=[row["card"]], accounting_started_unix=before, admission_samples=samples)
            wake(root)
            while True:
                current = db.execute("SELECT state FROM leases WHERE id=?", (ident,)).fetchone()[0]
                if end.is_set() or current == "closing":
                    with write(db):
                        db.execute("UPDATE leases SET state='closing' WHERE id=?", (ident,))
                        active = db.execute("SELECT 1 FROM jobs WHERE lease=? AND state IN ('reserved','starting','running') LIMIT 1", (ident,)).fetchone()
                    if not active or end.is_set():
                        break
                try:
                    serve_fd(db,root,ident,listener,lease)
                except (socket.timeout, ConnectionError, ValueError):
                    pass
        reason = None
    finally:
        path.unlink(missing_ok=True)
        # close() preserves the advisory lock while independent worker units
        # hold duplicates. LOCK_UN on a shared open file description would not.
        for handle in lease.device_locks:
            handle.close()
        if lease.job_lock is not None:
            if not end.is_set() or not lease.device_locks:
                helper.remove_job(lease.queue_dir, manifest["job_id"])
            lease.job_lock.close()
        with write(db):
            status = "released" if lease.device_locks else "missed"
            if lease.device_locks and end.is_set():
                status = "lost"
            db.execute("UPDATE leases SET state=?,released=?,reason=? WHERE id=?", (status,time.time(),reason,ident))
        emit(status, physical_ids=[row["card"]] if lease.device_locks else [], reason=reason)
        wake(root)


def get_fds(root, job):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(6)
        peer.connect(sockpath(root, "lease-"+str(job["lease"])))
        peer.sendall(packed({"job":job["id"],"generation":job["generation"]}).encode())
        msg, meta, _, _ = peer.recvmsg(512, socket.CMSG_SPACE(8))
        fds = array.array("i")
        for level, kind, data in meta:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(data)
        if msg != b"ok" or len(fds) != 2:
            for fd in fds:
                os.close(fd)
            raise OSError(msg.decode())
        return tuple(fds)


def run_job(root, ident, generation):
    db = connect(root)
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    child_subreaper(True)
    with write(db):
        job = lookup(db, ident)
        if job["generation"] != generation or job["state"] != "reserved" or not job["command"]:
            return
        change(db, ident, "starting", None, wrapper=os.getpid())
    spec, cmd = json.loads(job["spec"]), json.loads(job["command"])
    fds = ()
    proc = None
    code, reason = -1, None
    try:
        if job["backend"] == "npu":
            try:
                fds = get_fds(root, job)
            except OSError as error:
                with write(db):
                    cur = lookup(db, ident)
                    if cur["state"] == "starting" and cur["generation"] == generation:
                        defer(db, job, str(error))
                return
        tag = "cpu" if job["backend"] == "cpu" else "npu:0"
        env = {**os.environ, **cmd["env"], "RUNQ_DEVICE":tag}
        for name in ("NPU_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES","ASCEND_VISIBLE_DEVICES","ASCEND_RT_VISIBLE_DEVICES"):
            env.pop(name, None)
        card = None
        if fds:
            card = db.execute("SELECT card FROM leases WHERE id=?", (job["lease"],)).fetchone()[0]
            env["ASCEND_RT_VISIBLE_DEVICES"] = env["RUNQ_PHYSICAL_DEVICE"] = str(card)
        if cmd.get("affinity"):
            os.sched_setaffinity(0, cmd["affinity"])
        log = Path(cmd.get("log") or root / f"job-{ident}.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as stream, write(db):
            if stopping.is_set() or (spec.get("deadline") and time.time() >= spec["deadline"]):
                raise RuntimeError("stopped before execution")
            if spec.get("owner") and setting(db,"permit."+spec["owner"],0) < time.time():
                defer(db, job, "waiting for campaign controller")
                return
            proc = subprocess.Popen(cmd["argv"], cwd=cmd["cwd"], env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    pass_fds=fds, stdin=subprocess.DEVNULL)
            started = time.time()
            change(db,ident,"running",None,started=started,pid=proc.pid,ticks=proc_identity(proc.pid,strict=True)["start_ticks"])
        wake(root)
        code = proc.wait()
        # A successful launcher which forks grandchildren still occupies the
        # device until every descendant in this subreaper's service has exited.
        while True:
            try:
                os.waitpid(-1, 0)
            except ChildProcessError:
                break
    except BaseException as error:
        reason = f"{type(error).__name__}: {error}"
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
    finally:
        for fd in fds:
            os.close(fd)
        with write(db):
            cur = lookup(db, ident)
            if cur["generation"] == generation and cur["state"] in ("running","starting"):
                cancelled = db.execute("SELECT 1 FROM events WHERE job=? AND event='stop_requested' LIMIT 1",(ident,)).fetchone()
                outcome = "cancelled" if cancelled else "completed" if code == 0 and not reason else "failed"
                change(db, ident, outcome, "cancelled by user" if cancelled else reason, rc=code, finished=time.time())
        wake(root)
