"""Durable, account-local dispatch state. SQLite serializes placement and claims."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import time

ACTIVE = ("reserved", "starting", "running")
TERMINAL = ("completed", "failed", "cancelled")
SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leases (
 id INTEGER PRIMARY KEY, card INTEGER NOT NULL, role TEXT NOT NULL, state TEXT NOT NULL,
 trigger_job INTEGER NOT NULL, grant TEXT NOT NULL, created REAL NOT NULL, acquired REAL,
 released REAL, pid INTEGER, ticks INTEGER, tried REAL, reason TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS live_card ON leases(card) WHERE state IN ('opening','held','closing');
CREATE TABLE IF NOT EXISTS jobs (
 id INTEGER PRIMARY KEY, job_key TEXT NOT NULL UNIQUE, spec TEXT NOT NULL, original TEXT NOT NULL,
 target TEXT NOT NULL, manual INTEGER NOT NULL, fill INTEGER NOT NULL, state TEXT NOT NULL,
 created REAL NOT NULL, requested REAL NOT NULL, assigned REAL, started REAL, finished REAL,
 backend TEXT, lease INTEGER REFERENCES leases(id), generation INTEGER NOT NULL DEFAULT 0,
 grant TEXT, command TEXT, pid INTEGER, ticks INTEGER, wrapper INTEGER, tried REAL,
 rc INTEGER, reason TEXT, first_step REAL);
CREATE INDEX IF NOT EXISTS active_jobs ON jobs(state, manual, requested);
CREATE INDEX IF NOT EXISTS jobs_by_lease ON jobs(lease);
CREATE TABLE IF NOT EXISTS bindings (name TEXT PRIMARY KEY, backend TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, job INTEGER, at REAL NOT NULL, event TEXT NOT NULL, detail TEXT);
"""


def default_root():
    return Path(os.environ.get("RUNQ_HOME") or Path(os.environ.get("XDG_STATE_HOME", Path.home()/".local/state"))/"tide-runq").resolve()


def connect(root):
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(root/"queue.sqlite3", isolation_level=None, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='events'").fetchone():
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
    return db


@contextmanager
def write(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.commit()
    except BaseException:
        db.rollback()
        raise


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def setting(db, key, default=None):
    row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, key, value):
    db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, packed(value)))


def event(db, job, name, detail=None):
    db.execute("INSERT INTO events(job,at,event,detail) VALUES (?,?,?,?)", (job, time.time(), name, detail))


def prefix(root):
    return "tide-runq-" + hashlib.sha256(os.fsencode(Path(root).resolve())).hexdigest()[:10]


def sockpath(root, suffix="wake"):
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return str(runtime/(prefix(root)+"-"+suffix+".sock"))


def wake(root):
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.setblocking(False)
        try:
            sock.sendto(b"!", sockpath(root))
        except OSError:
            pass


def device(target):
    if target in ("auto", "cpu"):
        return target
    if not re.fullmatch(r"npu:(0|[1-9][0-9]*)", target):
        raise ValueError("expected auto, cpu or npu:PHYSICAL_ID")
    return "npu"


def submit(db, key, spec, target="auto", fill=False, grant=None):
    kind = device(target)
    allowed = spec["allowed"]
    if not allowed or set(allowed) - {"cpu", "npu"} or kind != "auto" and kind not in allowed:
        raise ValueError("device is outside the job's supported backends")
    if any(type(spec["ram"].get(k)) is not int or spec["ram"][k] < 0 for k in allowed):
        raise ValueError("expected host-memory reservations in bytes")
    if "npu" in allowed and (type(spec.get("hbm")) is not int or spec["hbm"] <= 0):
        raise ValueError("NPU jobs require a positive peak HBM estimate")
    now = time.time()
    original = packed([spec, target, bool(fill)])
    with write(db):
        row = db.execute("SELECT id,original FROM jobs WHERE job_key=?", (key,)).fetchone()
        if row:
            if row["original"] != original:
                raise ValueError(f"existing job key {key!r} has a different submission")
            return row["id"]
        ident = db.execute("""INSERT INTO jobs(job_key,spec,original,target,manual,fill,state,created,requested,command)
            VALUES(?,?,?,?,?,?,'queued',?,?,?)""", (key, packed(spec), original, target, int(kind != "auto"),
            int(fill), now, now, packed(spec["command"]) if spec.get("command") else None)).lastrowid
        db.execute("UPDATE jobs SET grant=? WHERE id=?", (packed(grant), ident))
        event(db, ident, "submitted", target)
        return ident


def lookup(db, ident):
    if str(ident).isdigit():
        row = db.execute("SELECT * FROM jobs WHERE id=?", (int(ident),)).fetchone()
    else:
        row = db.execute("SELECT * FROM jobs WHERE job_key=?", (ident,)).fetchone()
    if not row:
        raise ValueError("unknown job: " + str(ident))
    return dict(row)


def pin(db, ident, target, fill=False, grant=None):
    kind = device(target)
    with write(db):
        row = lookup(db, ident)
        if row["target"] == target and bool(row["fill"]) == fill and grant is None:
            return row
        if row["state"] != "queued":
            raise ValueError(f"job {row['id']} is {row['state']} on {row['backend']}; its placement is committed")
        spec = json.loads(row["spec"])
        fixed = db.execute("SELECT backend FROM bindings WHERE name=?", (spec.get("binding"),)).fetchone()
        if kind != "auto" and (kind not in spec["allowed"] or fixed and fixed[0] != kind):
            raise ValueError("target conflicts with the job's backend or an already bound recipe")
        if kind != "auto" and spec.get("binding"):
            for other in db.execute("SELECT id,target,spec FROM jobs WHERE state='queued' AND manual=1 AND id!=?", (row["id"],)):
                if json.loads(other["spec"]).get("binding") == spec["binding"] and device(other["target"]) != kind:
                    raise ValueError("a pending partner of this recipe is pinned to another backend")
        db.execute("UPDATE jobs SET target=?,manual=?,fill=?,requested=?,grant=?,reason=NULL WHERE id=?",
                   (target, int(kind != "auto"), int(fill), time.time(), packed(grant), row["id"]))
        event(db, row["id"], "placement_requested", target)
        return lookup(db, row["id"])


def unit(root, kind, ident, generation=None):
    suffix = f"-{generation}" if generation is not None else ""
    return f"{prefix(root)}-{kind}-{ident}{suffix}.service"


def command_env():
    # Carry the submitting software stack, without exporting unrelated account credentials.
    names = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
             "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_DEVICE_BACKEND_AUTOLOAD",
             "PYTORCH_NPU_ALLOC_CONF", "VIRTUAL_ENV", "TOOLCHAIN_HOME", "TBE_IMPL_PATH"}
    return {k:v for k,v in os.environ.items() if (k in names or k.startswith(("ASCEND_", "CANN_", "HCCL_")))
            and "VISIBLE_DEVICES" not in k}


def active_units(db, root):
    return {unit(root, "job", row["id"], row["generation"]) for row in
            db.execute("SELECT id,generation FROM jobs WHERE state IN ('reserved','starting','running')")}


def manual_grant(db, root, target, until):
    from .runq_devices import FRESH_SECONDS, grant_for, observe
    if device(target) != "npu":
        return None
    samples = setting(db, "samples", [])
    sample = samples[-1] if samples and time.time()-samples[-1]["at"] < FRESH_SECONDS else observe()
    return grant_for(sample, target, active_units(db, root), until)
