#!/usr/bin/env python3
"""Explicit, expiring sharing policy for a pinned site cooperative NPU queue.

This optional site adapter does not belong to the installed tidegraph runtime.
It changes only the in-process availability predicate: the supplied scheduler
still owns priority/FIFO, two samples, all-or-nothing locks and child lifetime.
The site scheduler file and global device configuration are never modified.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def process_identity(pid, proc_root=Path("/proc")):
    """PID alone is insufficient after reuse; unreadable identities fail closed."""
    folder = proc_root / str(pid)
    fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "uid": folder.stat().st_uid}


def validate_policy(policy):
    if policy.get("schema") != "tide-npu-sharing-v1":
        raise ValueError("policy.schema must be tide-npu-sharing-v1")
    if not isinstance(policy.get("reason"), str) or not policy["reason"].strip():
        raise ValueError("policy.reason must explain explicit sharing authorization")
    if (type(policy.get("expires_unix")) not in (int, float)
            or not math.isfinite(policy["expires_unix"])):
        raise ValueError("policy.expires_unix must be a timestamp")
    if not isinstance(policy.get("cards"), dict) or not policy["cards"]:
        raise ValueError("policy.cards must name explicitly authorized physical cards")
    for card, rule in policy["cards"].items():
        if not card.isdecimal() or str(int(card)) != card:
            raise ValueError("policy.cards keys must be physical nonnegative integers")
        for key in ("max_hbm_mib", "max_aicore_percent", "required_free_mib"):
            if type(rule.get(key)) is not int or rule[key] < 0:
                raise ValueError(f"policy.cards.{card}.{key} must be nonnegative int")
        if rule["required_free_mib"] <= 0 or rule["max_aicore_percent"] > 100:
            raise ValueError(f"policy.cards.{card}: require headroom and valid utilization")
        if not isinstance(rule.get("processes"), list) or not rule["processes"]:
            raise ValueError(f"policy.cards.{card}.processes must bind allowed owners")
        ids = []
        for owner in rule["processes"]:
            if set(owner) != {"pid", "start_ticks", "uid"} or any(
                type(v) is not int or v < 0 for v in owner.values()
            ) or owner["pid"] == 0:
                raise ValueError(f"policy.cards.{card}: invalid process identity")
            ids.append(owner["pid"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"policy.cards.{card}: duplicate process identity")


def shareable(device, policy, *, identity=process_identity, now=time.time):
    rule = policy["cards"].get(str(device.physical_id))
    if not rule or now() >= policy["expires_unix"]:
        return False
    if (device.health != "OK" or device.hbm_used_mib > rule["max_hbm_mib"]
            or device.aicore_percent > rule["max_aicore_percent"]
            or device.hbm_total_mib - device.hbm_used_mib < rule["required_free_mib"]):
        return False
    owners = {x["pid"]: x for x in rule["processes"]}
    try:
        return all(pid in owners and identity(pid) == owners[pid]
                   for pid in device.process_ids)
    except (OSError, ValueError, IndexError):
        return False


def load_helper(path, expected_hash):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError("site scheduler hash changed; review before sharing")
    spec = importlib.util.spec_from_file_location("gef_site_npu_queue", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    exec(compile(source, str(path), "exec"), helper.__dict__)
    for name in ("DeviceStatus", "snapshot", "main", "attempt_schedule"):
        if not hasattr(helper, name):
            raise ValueError(f"site scheduler lacks reviewed interface {name}")
    return helper


def install_policy(helper, policy, record):
    validate_policy(policy)
    original_free = helper.DeviceStatus.is_free
    original_snapshot = helper.snapshot

    def available(device, *, max_hbm_mib, max_aicore_percent):
        # Expiry also prevents a stale shared request from becoming an idle run.
        if time.time() >= policy["expires_unix"]:
            return False
        if str(device.physical_id) in policy["cards"]:
            return shareable(device, policy)
        return original_free(device, max_hbm_mib=max_hbm_mib,
                             max_aicore_percent=max_aicore_percent)

    def snapshot(mapping):
        devices = original_snapshot(mapping)
        record("snapshot", devices={str(k): dataclasses.asdict(v) for k, v in devices.items()})
        return devices

    helper.DeviceStatus.is_free = available
    helper.snapshot = snapshot
    return original_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True, help="site-owned scheduler, read only")
    parser.add_argument("--helper-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True, help="new persistent JSONL")
    parser.add_argument("queue_args", nargs=argparse.REMAINDER, help="scheduler arguments after --")
    args = parser.parse_args(argv)
    policy = json.loads(args.policy.read_text())
    validate_policy(policy)
    if time.time() >= policy["expires_unix"]:
        parser.error("sharing policy expired; inspect devices and issue a new policy")
    helper = load_helper(args.helper, args.helper_sha256)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("x") as stream:
        def record(event, **fields):
            stream.write(json.dumps(dict(event=event, unix=time.time(), **fields), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        record("policy", policy=policy, helper_sha256=args.helper_sha256,
               interpretation="explicit shared-resource observation; not exclusive timing")
        snapshot = install_policy(helper, policy, record)
        queue_args = args.queue_args[1:] if args.queue_args[:1] == ["--"] else args.queue_args
        try:
            code = helper.main(queue_args)
            record("terminal", exit_code=code)
            return code
        finally:
            # Observation only: no new selection, process signalling or lock changes.
            try:
                mapping = helper.parse_mapping(helper.run_npu_smi("info", "-m"))
                record("after", devices={str(k): dataclasses.asdict(v) for k, v in snapshot(mapping).items()})
            except Exception as error:
                record("after-unavailable", error=repr(error))


if __name__ == "__main__":
    raise SystemExit(main())
