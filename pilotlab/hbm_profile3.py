"""Measured HBM by immutable worker source and model/input shape.

One row per attempt keeps failed allocations, startup and driver high waters
distinct. CPU results never supply evidence for an NPU reservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from .io import sha256
from .source_migration3 import digest, read_bound

GIB = 2**30
SNAPSHOT = "pilot-r3-hbm-profile-v1"
DEFAULT_POLICY = (1.5, GIB // 2)
SHAPE_FIELDS = ("model", "precision", "deterministic", "sequence_length", "stream_chunks", "batch_size")


def hbm_shape(task):
    return digest({key: task.get(key) for key in ("source_identity", *SHAPE_FIELDS)})


def source_content(identity):
    """Compare identical frozen trees installed at different absolute paths."""
    if not identity or not identity.get("source_files"):
        return None
    files = {}
    for name, checksum in identity["source_files"].items():
        parts = Path(name).parts
        roots = [i for i, part in enumerate(parts) if part == "pilotlab" or parts[i:i+2] == ("src", "tide")]
        if len(roots) != 1:
            return None
        relative = "/".join(parts[roots[0]:])
        if relative in files:
            return None
        files[relative] = checksum
    return digest({**identity, "source_files":files})


def _amount(error, pattern):
    number = re.search(pattern + r"\s*([\d.]+)\s*(bytes?|[KMGT]iB)\b", error, re.I)
    if not number:
        return None
    units = {"kib": 2**10, "mib": 2**20, "gib": GIB, "tib": 2**40}
    return math.ceil(float(number[1]) * units.get(number[2].lower(), 1))


def oom_kind(result, log):
    if result.get("status") != "failed":
        return None
    error = result.get("error", "")
    if result.get("failure_class") == "capacity_error" and error.lower().startswith("npu out of memory"):
        return "allocator"
    if (result.get("failure_class") is None and "ERR00100" in error and "aclnn" in error and Path(log).is_file()):
        text = Path(log).read_text(errors="replace")
        if "Memory_Allocation_Failure(EL0004)" in text and "driver error:out of memory" in text:
            return "driver"
    return None


def reservation(peak, factor, padding):
    return max(1, math.ceil((peak * factor + padding) / GIB)) * GIB if peak else 0


class HbmProfile:
    def __init__(self, directory, *, readonly=False):
        self.path = Path(directory).resolve() / "hbm-profile.sqlite3"
        uri = self.path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc")
        self.db = sqlite3.connect(uri, uri=True, timeout=15)
        self.db.row_factory = sqlite3.Row
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    path TEXT PRIMARY KEY, shape TEXT NOT NULL, model TEXT NOT NULL, batch INTEGER NOT NULL,
                    status TEXT, result_sha256 TEXT, allocated INTEGER, reserved INTEGER,
                    startup_allocated INTEGER, startup_reserved INTEGER,
                    driver INTEGER DEFAULT 0, before_first_update INTEGER DEFAULT 0,
                    oom TEXT, oom_active INTEGER, oom_reserved INTEGER, requested INTEGER);
                CREATE INDEX IF NOT EXISTS attempts_by_shape ON attempts(shape);
                CREATE TABLE IF NOT EXISTS trials (shape TEXT PRIMARY KEY, leader TEXT NOT NULL, basis INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY CHECK(id=1), factor REAL NOT NULL, padding INTEGER NOT NULL);
            """)

    def policy(self):
        row = self.db.execute("SELECT factor,padding FROM policy WHERE id=1").fetchone()
        return tuple(row) if row else DEFAULT_POLICY

    def set_policy(self, factor, padding):
        if not math.isfinite(factor) or factor < 1 or type(padding) is not int or padding < 0:
            raise ValueError("HBM factor must be at least one and padding nonnegative bytes")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO policy VALUES(1,?,?)", (factor, padding))

    def add_trial(self, task, basis):
        if task.get("device") != "npu:0" or not task.get("source_identity") or not task.get("id") or basis <= 0:
            raise ValueError("a trial needs a frozen NPU task and a positive sizing basis")
        shape = hbm_shape(task)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO trials VALUES(?,?,?)", (shape, task["id"], basis))
            stored = self.db.execute("SELECT leader,basis FROM trials WHERE shape=?", (shape,)).fetchone()
            if tuple(stored) != (task["id"], basis):
                raise ValueError("shape already has a different representative or sizing basis")
        return shape

    def record_result(self, task, task_hash, path, result, result_hash, *, outcome=None):
        if (not str(result.get("device", "")).startswith("npu") or result.get("status") == "running"
                or not task.get("source_identity") or result.get("source_identity") != task["source_identity"]
                or result.get("id") != task["id"] or result.get("task_sha256") != task_hash):
            return False
        directory = Path(path).resolve().parent
        mem = result.get("device_memory") or {}
        oom = oom_kind(result, directory / "stdout.log")
        error = result.get("error", "") if oom else ""
        requested = _amount(error, r"tried to allocate")
        active = _amount(error, r"total capacity;") if oom else None
        reserved_at_failure = _amount(error, r"(?:free;|free,)") if oom else None
        sh = hbm_shape(task)
        args = (str(directory), sh, task["model_id"], task["batch_size"], outcome or result["status"], result_hash,
                mem.get("npu_peak_allocated_bytes"), mem.get("npu_peak_reserved_bytes"),
                mem.get("npu_startup_peak_allocated_bytes"), mem.get("npu_startup_peak_reserved_bytes"),
                oom, active, reserved_at_failure, requested)
        cur = self.db.execute("""INSERT INTO attempts(path,shape,model,batch,status,result_sha256,allocated,reserved,
            startup_allocated,startup_reserved,oom,oom_active,oom_reserved,requested) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET status=excluded.status,result_sha256=excluded.result_sha256,
            allocated=excluded.allocated,reserved=excluded.reserved,startup_allocated=excluded.startup_allocated,
            startup_reserved=excluded.startup_reserved,oom=excluded.oom,oom_active=excluded.oom_active,
            oom_reserved=excluded.oom_reserved,requested=excluded.requested
            WHERE shape=excluded.shape AND (result_sha256 IS NULL OR result_sha256=excluded.result_sha256)""", args)
        if not cur.rowcount:
            raise ValueError(f"HBM evidence identity changed: {directory}")
        return True

    def record_driver(self, directory, shape, model, batch, amount, before_update):
        if type(amount) is not int or amount <= 0:
            return
        self.db.execute("""INSERT INTO attempts(path,shape,model,batch,driver,before_first_update)
            VALUES (?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET driver=MAX(driver,excluded.driver),
            before_first_update=MAX(before_first_update,excluded.before_first_update)
            WHERE shape=excluded.shape AND (driver<excluded.driver OR before_first_update<excluded.before_first_update)""",
            (str(Path(directory).resolve()), shape, model, batch, amount, amount if before_update else 0))

    def summaries(self):
        # Adding a refused allocation to the pool is a conservative sizing hint.
        # The separate raw fields preserve what was actually observed.
        rows = self.db.execute("""WITH measurements AS (
            SELECT *, MAX(COALESCE(allocated,0),COALESCE(reserved,0),COALESCE(startup_allocated,0),
                          COALESCE(startup_reserved,0),driver,COALESCE(oom_active,0),COALESCE(oom_reserved,0)) AS observed,
                   MAX(COALESCE(oom_reserved,0),COALESCE(oom_active,0))+COALESCE(requested,0) AS oom_hint
            FROM attempts)
            SELECT shape, model, batch, COUNT(result_sha256) AS terminal,
                   SUM(CASE WHEN status='completed' AND observed>0 THEN 1 ELSE 0 END) AS successful,
                   MAX(CASE WHEN status='completed' THEN observed ELSE 0 END) AS success_peak_bytes,
                   MAX(observed) AS observed_peak_bytes, MAX(oom_hint) AS oom_sizing_bytes,
                   MAX(MAX(observed,oom_hint)) AS peak FROM measurements GROUP BY shape""")
        summaries = {r["shape"]: dict(r) for r in rows}
        for row in self.db.execute("SELECT shape,leader,basis FROM trials"):
            info = summaries.setdefault(row["shape"], dict(peak=0, successful=0, terminal=0))
            if not info["successful"]:
                info["leader"] = row["leader"]
                info["trial_basis_bytes"] = row["basis"]
                info["peak"] = max(info["peak"], row["basis"])
        factor, padding = self.policy()
        for info in summaries.values():
            size = reservation(info["peak"], factor, padding)
            info["reservation_bytes"] = max(size, 8*GIB) if size and not info["successful"] else size
        return summaries

    def snapshot(self):
        self.db.execute("SAVEPOINT hbm_export")
        try:
            return self._snapshot()
        finally:
            self.db.execute("RELEASE hbm_export")

    def _snapshot(self):
        factor, padding = self.policy()
        shapes = self.summaries()
        portable = {}
        for shape, directory in self.db.execute("SELECT shape,MIN(path) FROM attempts GROUP BY shape"):
            path = Path(directory) / "task.json"
            if not path.is_file():
                continue
            task = json.loads(path.read_bytes())
            if hbm_shape(task) != shape:
                raise ValueError(f"changed task in HBM snapshot: {path}")
            content = source_content(task["source_identity"])
            if content:
                portable[shape] = dict(source=content, inputs={k:task.get(k) for k in SHAPE_FIELDS})
        return dict(schema=SNAPSHOT, factor=factor, padding_bytes=padding, shapes=shapes, portable_shapes=portable)


def frozen_profile(reference, worker_source=None):
    if not reference:
        return {}
    snapshot = read_bound(reference)
    if snapshot.get("schema") != SNAPSHOT:
        raise ValueError("invalid frozen HBM profile")
    values = {k: row["reservation_bytes"] for k, row in snapshot["shapes"].items()}
    if any(type(size) is not int or size < 0 for size in values.values()):
        raise ValueError("invalid HBM reservation in frozen snapshot")
    content = source_content(worker_source)
    if content:
        for previous, item in snapshot.get("portable_shapes", {}).items():
            if item["source"] == content:
                shape = hbm_shape({**item["inputs"], "source_identity":worker_source})
                values[shape] = max(values.get(shape, 0), values[previous])
    return values


def backfill(profile, plan, output):
    """Import only terminal results and tasks bound to this campaign."""
    refs = []
    if plan.get("stage_a_migration"):
        for row in read_bound(plan["stage_a_migration"])["completed"].values():
            refs.append((row["task"], row["result"], None))
    scheduler = json.loads((output / "scheduler.json").read_bytes())
    for row in scheduler["tasks"].values():
        for attempt in row.get("attempts", []):
            path = Path(attempt["path"]) / "result.json"
            if attempt.get("status") not in ("completed", "failed", "stopped_budget", "recipe_rejected", "interrupted") or not path.is_file():
                continue
            argv = attempt.get("command") or []
            task = Path(argv[argv.index("--task")+1]) if "--task" in argv else path.parent / "task.json"
            if task.is_file():
                refs.append(({"path":str(task)}, {"path":str(path), "sha256": row.get("result_sha256")
                             if row.get("result_path") == str(path) else None}, attempt["status"]))
    loaded = 0
    with profile.db:
        for tref, ref, outcome in refs:
            path = Path(ref["path"])
            data = path.read_bytes()
            result = json.loads(data)
            if result.get("device") != "npu:0":
                continue
            result_hash = hashlib.sha256(data).hexdigest()
            if ref.get("sha256") and ref["sha256"] != result_hash:
                raise ValueError(f"changed HBM result: {path}")
            raw = Path(tref["path"]).read_bytes()
            task_hash = hashlib.sha256(raw).hexdigest()
            if tref.get("sha256") and tref["sha256"] != task_hash:
                raise ValueError("changed HBM task: " + tref["path"])
            if task_hash != result.get("task_sha256") and (path.parent/"task.json").is_file():
                raw = (path.parent/"task.json").read_bytes()
                task_hash = hashlib.sha256(raw).hexdigest()
            if not profile.record_result(json.loads(raw), task_hash, path, result, result_hash, outcome=outcome):
                raise ValueError(f"HBM result and frozen task disagree: {path}")
            loaded += 1
    return loaded


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage A/B HBM measurements, reservations and frozen handoff")
    parser.add_argument("--round", required=True, type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("backfill").add_argument("--plan", required=True, type=Path)
    policy = commands.add_parser("policy")
    policy.add_argument("--factor", required=True, type=float)
    policy.add_argument("--padding-gib", required=True, type=float)
    trial = commands.add_parser("trial")
    trial.add_argument("--task", required=True, type=Path)
    trial.add_argument("--basis-gib", required=True, type=float, help="Observed sizing basis before policy margins")
    commands.add_parser("snapshot").add_argument("--output", required=True, type=Path)
    commands.add_parser("estimate").add_argument("--task", required=True, type=Path)
    args = parser.parse_args(argv)
    profile = HbmProfile(args.round, readonly=args.action in ("snapshot", "estimate"))
    if args.action == "policy":
        profile.set_policy(args.factor, math.ceil(args.padding_gib*GIB))
        print(profile.policy())
    elif args.action == "backfill":
        print(backfill(profile, json.loads(args.plan.read_bytes()), args.round.resolve()))
    elif args.action == "trial":
        print(profile.add_trial(json.loads(args.task.read_bytes()), math.ceil(args.basis_gib*GIB)))
    elif args.action == "snapshot":
        from .scheduler3 import freeze_json
        freeze_json(args.output, profile.snapshot())
        print(sha256(args.output))
    else:
        task = json.loads(args.task.read_bytes())
        info = profile.summaries().get(hbm_shape(task), {})
        print(json.dumps({**info, "reservation_bytes": max(task.get("reserved_hbm_bytes",0),
            info.get("reservation_bytes") or 8*GIB)}))
    profile.db.close()


if __name__ == "__main__":
    main()
