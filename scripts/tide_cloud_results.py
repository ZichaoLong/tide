"""Read native TIDE evidence and replicate the verified cloud output objects to East OBS."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import subprocess
import time

from tide_cloud_allocation import PREFIX, PROJECT, wire
from tide_cloud_dispatch import object_key
from tide_portable import read, require, sha, write
from pilotlab.runtime_migration3 import TARGET, valid_allocation, verify_result, history_for


def canary(area, state, region, prefix, claim):
    a = claim["attempt"]
    old = str(claim["row"]["legacy_id"])
    if old in state["canaries"]:
        return
    key = prefix + "/products/first-update/" + a + ".json"
    head = region.head(key)
    if not head:
        return
    path = area / "attempts" / a / "first-update.json"
    if not path.exists():
        region.download(key, path)
    proof = read(path)
    plan_path = area / "attempts" / a / "runtime-plan.json"
    origin = claim["row"]["resume"]
    require(proof["task"] == claim["task"] and proof["fingerprint"] == claim["fingerprint"]
            and proof["runtime_plan_sha256"] == sha(plan_path) and proof["runtime"] == TARGET
            and proof["status"] == "passed" and proof["from_steps"] == origin["steps"]
            and proof["steps"] == origin["steps"] + 1 and proof["restored"]["checkpoint_sha256"] == origin["sha256"]
            and proof["checkpoint_load"]["status"] == "exact" and proof["checkpoint_load"]["state_sha256"] == proof["state_sha256"]
            and all(math.isfinite(proof[k]) for k in ("loss", "grad_norm")), "invalid first update for " + old)
    valid_allocation(proof["allocation"], claim["job"])
    state["canaries"][old] = {"at": time.time(), "job": claim["job"], "attempt": a,
                             "sha256": sha(path), "steps": proof["steps"], "state_sha256": proof["state_sha256"]}


def readback(region, key, h, size, node):
    before = region.head(key)
    require(before and before["size"] == size and before.get("sha256") in {None, h},
            "regional object metadata differs from native evidence")
    if node:
        from tide_cloud_submit import verify_object
        try:
            proof = verify_object(node, region, key, h, size)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            pass
        else:
            require(proof == {"sha256":h, "size":size}, "allocated node returned a different readback")
            after = region.head(key)
            require(after and (after["size"], after["etag"], after.get("version")) ==
                    (size, before["etag"], before.get("version")), "regional object changed during node readback")
            return {**after, "verified_on":"guiyang/" + node["id"]}
    return {**region.stream(key, expected=h, etag=before["etag"]), "verified_on":"client"}


def checkpoint_object(region, key, target, artifact, node):
    h, size = artifact["sha256"], artifact["size"]
    if region.head(target):
        return
    src = region.head(key)
    require(src and src["size"] == size and src.get("sha256") in (None,h), "published checkpoint metadata differs")
    if src.get("sha256") == h or src["etag"].lower() == artifact["content_md5"].lower():
        region.copy_verified(key, target, h, size, content_md5=artifact["content_md5"])
        return
    source = readback(region, key, h, size, node)
    try:
        region.copy_verified(key, target, None, size, service={
            "store":region.side, "etag":source["etag"], "version":source.get("version"),
            "operation":"tide-native-checkpoint-" + h})
    except FileExistsError:
        # Another attempt can publish identical weights. The caller verifies the actual destination body.
        if not region.head(target):
            raise


def ingest(area, cache, region, east, prefix, claim, state):
    a, fp = claim["attempt"], claim["fingerprint"]
    out = area / "attempts" / a
    canonical = "products/receipts/" + a + ".json"
    selected = claim.get("receipt_key", canonical)
    require(selected in {canonical, "products/reconciled/" + a + ".json"}, "invalid receipt recovery scope")
    receipt = out / ("receipt.json" if selected == canonical else "receipt-reconciled.json")
    location_suffix = ".location.json" if selected == canonical else ".reconciled.location.json"
    expected = claim.get("receipt_sha256")
    if not receipt.exists():
        region.download(prefix + "/" + selected, receipt, sha256=expected)
    if expected:
        require(sha(receipt) == expected, "recovered receipt does not match the node's publication")
    rec = read(receipt)
    frozen = read(area / "payload/production.json")
    require(rec["fingerprint"] == fp and rec["attempt"] == a and rec["task"] == claim["task"] and
            rec["job"] == claim["job"] and rec["descriptor_sha256"] == claim["digest"]
            and rec["runq"] == frozen["runq_commit"], "mismatched cloud receipt")
    east.put_file(PREFIX + "/attempts/" + a + "/" + receipt.name, receipt)
    platform = area / "allocations" / claim["job"] / "submitted.json"
    node = {"id":read(platform)["id"], "name":claim["job"]} if platform.is_file() else None
    results = {}
    for name, artifact in rec["artifacts"].items():
        key = prefix + "/" + artifact["path"]
        require(artifact["path"] == "products/attempts/" + a + "/" + name and Path(name).name == name,
                "attempt published an artifact outside its scope")
        h, size = artifact["sha256"], artifact["size"]
        require(len(h) == 64 and all(c in "0123456789abcdef" for c in h), "invalid output digest")
        if name.endswith(".pt"):
            target = object_key(h)
            path = cache / h
            location = out / (name + location_suffix)
            if not location.exists():
                checkpoint_object(region, key, target, artifact, node)
                observed = readback(region, target, h, size, node)
                require(observed["size"] == size, "regional checkpoint body length differs")
                record = {"schema":"tide-cloud-checkpoint/v1", "key": target, "sha256": h, "size": size,
                          "path": str(path), "uri": region.uri(target), "etag":observed["etag"], "version":observed.get("version"),
                          "verified_sha256":h, "verified_unix": time.time(), "verified_on":observed["verified_on"],
                          "producer":{"job":claim["job"], "attempt":a, "fingerprint":fp,
                                      "receipt_sha256":sha(receipt)}}
                write(location, record)
            record = read(location)
            require(record["sha256"] == h and record["producer"]["receipt_sha256"] == sha(receipt),
                    "checkpoint readback belongs to different evidence")
            east.put_file(PREFIX + "/attempts/" + a + "/" + location.name, location)
            write(area / "regional-objects" / (h + ".json"), record)
        else:
            path = out / name
            if not path.exists():
                region.download(key, path, sha256=h)
            require(sha(path) == h, "native evidence changed")
            east.put_file(PREFIX + "/attempts/" + a + "/" + name, path, sha256=h)
        results[name] = str(path)
    status = rec["status"]
    require(status in ("completed", "interrupted", "failed"), "unrecognized project verdict")
    if "project-verdict.json" in results:
        verdict = read(results["project-verdict.json"])
        require(verdict["fingerprint"] == fp and verdict["task"] == claim["task"] and verdict["status"] == status,
                "transport does not match native project verdict")
    if status == "failed" and not {"result.json", "project-verdict.json"}.issubset(results):
        claim.update(status="failed", needs_relay=True,
                     reason=rec.get("error") or "worker exited without a native result")
        return
    native = read(results["result.json"])
    require(native["execution_runtime"]["adapter_sha256"] == frozen["files"]["runtime_migration3.py"],
            "result was produced by a different runtime adapter")
    path = out / "runtime-plan.json"
    task = read(claim["row"]["path"])
    plan = read(path)
    if status in ("completed", "interrupted"):
        require("checkpoint.pt" in results and native["checkpoint"]["sha256"] == Path(results["checkpoint.pt"]).name,
                "native result missing the exact terminal checkpoint")
        verify_result(task, native, plan, history_for(plan, task, fp), sha(path))
    evidence = {"result_path": results["result.json"], "verdict_path": results["project-verdict.json"],
                "runtime_plan_path": str(path), "receipt_path": str(receipt), "checkpoint_path": results.get("checkpoint.pt"),
                "task_path": claim["row"]["path"], "runq_status": status}
    if "checkpoint.pt" in results:
        location = out / ("checkpoint.pt" + location_suffix)
        evidence["checkpoint_receipt"] = {"path": str(location), "sha256": sha(location)}
    if status == "interrupted":
        claim["status"] = "interrupted"
        old = claim["row"]
        new = {**old, "prior_results": [*old["prior_results"], evidence["result_path"]],
               "reserved_evaluation_tokens": old.get("reserved_evaluation_tokens", 0) + claim["reserved_evaluation_tokens"],
               "resume": {"sha256": native["checkpoint"]["sha256"], "size": native["checkpoint"]["size_bytes"],
                          "steps": native["checkpoint"]["steps"]}}
        state["retries"][fp] = new
    else:
        # The native campaign can consume qualified negative recipe/capacity observations too.
        state["accepted"][fp] = evidence
        claim["status"] = "accepted" if status == "completed" else "failed"
        if status == "failed":
            claim["reason"] = native.get("failure_class") or native.get("stop_reason") or "failed native verdict"
    claim["finished"] = rec.get("finished")


def archive_outputs(area, east):
    import ma_oms
    from concurrent.futures import ThreadPoolExecutor
    source = area / "regional-objects"
    archive = area / "east-objects"
    archive.mkdir(parents=True, exist_ok=True)
    records = {p.stem: read(p) for p in source.glob("*.json") if not (archive / p.name).is_file()}
    if not records:
        return
    # Journal the exact membership before submit. New arrivals cannot change an OMS retry.
    pending = area / "output-transfers"
    pending.mkdir(parents=True, exist_ok=True)
    assigned = set()
    groups = []
    for path in sorted(pending.glob("*.json")):
        hashes = read(path)["hashes"]
        assigned.update(hashes)
        if any(h in records for h in hashes):
            groups.append((path.stem, hashes))
    fresh = []
    for h in sorted(set(records) - assigned):
        key = object_key(h)
        previous = east.head(key)
        if previous:
            verified = east.stream(key, expected=h, etag=previous["etag"])
            require(verified["size"] == records[h]["size"], "existing authority object size differs")
            write(archive / (h + ".json"), {"uri": east.uri(key), "sha256":h, "size":verified["size"],
                                             "verified":time.time(), "source":"existing_east_readback"})
        else:
            fresh.append(h)
    for index in range(0, len(fresh), 16):
        hashes = fresh[index:index+16]
        name = "tide-r3-out-" + hashlib.sha256(wire(hashes)).hexdigest()[:36]
        write(pending / (name + ".json"), {"hashes": hashes})
        groups.append((name, hashes))
    def transfer(item):
        name, hashes = item
        keys = [object_key(h) for h in hashes]
        receipt = ma_oms.submit("guiyang", "east", keys, project=PROJECT, operation_id=name,
                                hashes=dict(zip(keys, hashes)))
        verified = ma_oms.wait(receipt, timeout=60)
        for row in verified["objects"]:
            h = row["sha256"]
            write(archive / (h + ".json"), {"uri": east.uri(row["destination_key"]), "sha256": h,
                  "size": row["size"], "verified": verified["verified_at"], "oms": verified["task_id"]})
    with ThreadPoolExecutor(max_workers=3) as pool:
        for task in [pool.submit(transfer, g) for g in groups[:3]]:
            try:
                task.result()
            except TimeoutError:
                pass
