"""Reviewable one-shot executor for one fixed plan; dry validation is default.

Run only under the root agent's resource/lifecycle supervision after reviewing
the fixed plan and recording a fresh live-use audit. No recursive removal and
no resume: any partial execution requires explicit receipt reconciliation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "cleanup-plan.json"
PREPARE = HERE / "prepare_cleanup.py"
EXPECTED_PLAN_SHA256 = "e8c280818e7402f1390ede8a58ef75b67bbedda5610babb54a8c49019feb93f0"
EXPECTED_PREPARE_SHA256 = "55564d4c12f84faa68e397a01032adf7bc4c73f0db3c7f40e30a3e604eebc174"
EXPECTED_COUNT = 1609
EXPECTED_ALLOCATED_BYTES = 37_304_471_552


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_record(path, value):
    require(path.parent == HERE, "execution record outside cleanup directory")
    descriptor, temporary = tempfile.mkstemp(prefix=".execution.", dir=HERE)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this executor's incomplete record.


def live_audit(path):
    path = Path(path)
    require(path.is_absolute() and path.parent == HERE and not path.is_symlink(),
            "live audit must be a regular JSON file directly in the cleanup directory")
    data = path.read_bytes()
    value = json.loads(data)
    require(value.get("schema") == "tide-cleanup-live-audit-v1" and value.get("state") == "passed",
            "live-use audit is not explicitly passed")
    require(value.get("plan_sha256") == EXPECTED_PLAN_SHA256,
            "live-use audit does not cover this exact plan")
    require(value.get("candidate_count") == EXPECTED_COUNT and value.get("live_candidate_references") == [],
            "live-use audit has uncovered paths or live references")
    observed = datetime.fromisoformat(value["checked_utc"])
    require(observed.tzinfo is not None, "live-use audit time must be timezone-aware")
    age = (datetime.now(timezone.utc) - observed).total_seconds()
    require(0 <= age <= 300, "live-use audit is stale or in the future")
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(data), checked_utc=value["checked_utc"])


def stat_fields(info):
    return dict(bytes=info.st_size, allocated_bytes=info.st_blocks * 512,
                device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                ctime_ns=info.st_ctime_ns, mode=info.st_mode, links=info.st_nlink)


def append_receipt(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def guarded_unlink(root_descriptor, row, stream):
    parts = Path(row["path"]).parts
    require(parts[0] == "artifacts" and ".." not in parts,
            f"candidate outside exact artifact scope: {row['path']}")
    parent = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
            os.close(parent)
            parent = following
        current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        require(stat_fields(current) == row["stat_guard"],
                f"per-file guard changed immediately before removal: {row['path']}")
        append_receipt(stream, dict(event="before_unlink", utc=utc(), path=row["path"],
                                    group=row["group"], stat_guard=row["stat_guard"]))
        os.unlink(parts[-1], dir_fd=parent)
        os.fsync(parent)
        append_receipt(stream, dict(event="unlinked", utc=utc(), path=row["path"],
                                    group=row["group"], allocated_bytes=row["stat_guard"]["allocated_bytes"]))
    finally:
        os.close(parent)


def verify_retained(plan, helpers):
    counts = {}
    for group in ("retained_tensors", "failed_ffn_profiler"):
        for row in plan["protected"][group]:
            helpers["same_guard"](row["path"], row["stat_guard"])
        counts[group] = len(plan["protected"][group])
    for row in plan["retained_records"]:
        _, identity = helpers["read_small_record"](row["path"])
        require(identity["sha256"] == row["sha256"], f"retained result changed: {row['path']}")
        helpers["same_guard"](row["path"], row["stat_guard"])
    for row in plan["preserved_records"]:
        _, identity = helpers["read_small_record"](row["preserved_path"])
        require(identity["sha256"] == row["sha256"], f"preserved copy changed: {row['preserved_path']}")
    counts.update(retained_records=len(plan["retained_records"]),
                  preserved_records=len(plan["preserved_records"]))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--activity-audit")
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    require(1 <= args.max_seconds <= 600, "finite execution budget must be 1..600 seconds")
    require(sha(PREPARE.read_bytes()) == EXPECTED_PREPARE_SHA256, "preparation code changed")
    require(sha(PLAN.read_bytes()) == EXPECTED_PLAN_SHA256, "fixed manifest changed")
    helpers = runpy.run_path(str(PREPARE), run_name="cleanup_preparation_library")
    validation = helpers["validate_existing"](EXPECTED_PLAN_SHA256)
    plan = json.loads(PLAN.read_bytes())
    require(plan["totals"]["count"] == EXPECTED_COUNT and
            plan["totals"]["allocated_bytes"] == EXPECTED_ALLOCATED_BYTES,
            "fixed candidate budget changed")
    if not args.execute:
        print(json.dumps(dict(state="validated_no_deletion", validation=validation), sort_keys=True))
        return
    require(args.activity_audit is not None, "execution requires the root agent's fresh live-use audit")
    audit = live_audit(args.activity_audit)
    result_path = HERE / "cleanup-execution-result.json"
    receipt_path = HERE / "cleanup-retirement.jsonl"
    require(not result_path.exists() and not receipt_path.exists(),
            "one-shot execution records already exist; reconcile before any further action")
    start = time.monotonic()
    result = dict(schema="tide-cleanup-execution-v1", state="running", started_utc=utc(),
                  plan_sha256=EXPECTED_PLAN_SHA256, executor_sha256=sha(Path(__file__).read_bytes()),
                  activity_audit=audit, max_seconds=args.max_seconds,
                  expected_count=EXPECTED_COUNT, expected_allocated_bytes=EXPECTED_ALLOCATED_BYTES,
                  removed_count=0, removed_allocated_bytes=0,
                  directories_removed=False, payloads_rehashed=False)
    atomic_record(result_path, result)
    root_descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for row in plan["candidates"]:
                require(time.monotonic() - start < args.max_seconds,
                        "finite cleanup deadline reached; preserve partial receipts")
                guarded_unlink(root_descriptor, row, stream)
                result["removed_count"] += 1
                result["removed_allocated_bytes"] += row["stat_guard"]["allocated_bytes"]
        require(all(not os.path.lexists(ROOT / row["path"]) for row in plan["candidates"]),
                "candidate still present or recreated after removal")
        result["after_verification"] = verify_retained(plan, helpers)
        result["state"] = "completed"
    except BaseException as error:
        result["state"] = "failed_partial_or_unstarted"
        result["error"] = repr(error)
        raise
    finally:
        os.close(root_descriptor)
        result["finished_utc"] = utc()
        result["elapsed_seconds"] = time.monotonic() - start
        atomic_record(result_path, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
