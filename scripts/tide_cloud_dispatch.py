"""Reserve evaluation exposure and issue immutable, content-verified R3 work to one queue."""

from __future__ import annotations

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

from tide_cloud_allocation import PREFIX, PROJECT, REPO, wire
from tide_cloud_inputs import data_files, digest, make_base, reference_file
from tide_portable import read, require, sha, write
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilotlab.runtime_migration3 import build_runtime_plan

CANARIES = (2237, 2907)


def starting_budget():
    pause = read(REPO / "docs/evidence/pilot-r3-stage-a-checkpoint-pause-20261127.json")
    used, original_preparation = {}, {}
    # The adjudication's historical exposure for an early subject already includes
    # its primary history. Its own scheduler is authoritative for that subject.
    for group in ("primary", "deferred"):
        ref = pause["groups"][group]["scheduler"]
        require(sha(ref["path"]) == ref["sha256"], "paused scheduler digest changed")
        sched = read(ref["path"])
        totals = Counter(sched["config"].get("prior_evaluation_reservations", {}))
        for key, row in sched["tasks"].items():
            if row.get("evaluation_budget_key"):
                totals[row["evaluation_budget_key"]] += row.get("evaluation_reserved_tokens", 0)
            if row.get("preparation_reserved_tokens"):
                original_preparation[key] = row["preparation_reserved_tokens"]
        used.update(totals)
    budget = read(REPO / ".local/pilot-r3/budget.json")
    claims = budget.get("preparation_token_reservations", {})
    preparation = sum(v["reserved_tokens"] for v in claims.values()) + sum(
        row.get("preparation_reserved_tokens", 0) for key, row in budget["segments"].items() if key not in claims)
    return {"evaluation": used, "preparation": preparation, "preparation_prior": original_preparation,
            "preparation_limit": budget["preparation_max_processed_tokens"]}


def priority(row):
    if row.get("priority") is not None:
        return row["priority"]
    # A ready dependency gets a modest head start; duration and scarce chip
    # memory discriminate among jobs at similar points on the training path.
    bonus = {"preflight": 3600, "init": 2400, "a1": 1800, "a2": 2400, "a3": 1200}.get(row["phase"], 0)
    seconds = row.get("estimated_seconds") or 0
    return bonus + min(7 * 86400, seconds) + (12000 if row.get("hbm_mib", 0) >= 48 * 1024 else 0)


def known_objects(area, verified):
    import ma_oms
    storage = area.parent / "storage"
    single = storage / "receipts/guiyang-object"
    service = {}
    delivery = storage / "oms-service-delivery.json"
    if delivery.exists():
        for group in read(delivery)["operations"]:
            if (group["status"] != 5 or group["n"] != group["total_num"] or group["bytes"] != group["complete_size"]
                    or group["failed_num"] or group["skipped_num"]):
                continue
            operation = read(ma_oms.receipt_path(group["id"]))
            require(operation["source"] == "east" and operation["destination"] == "guiyang"
                    and str(operation["task_id"]) == str(group["task"]), "unrecognized incoming OMS route")
            for obj in operation["objects"]:
                h = obj["sha256"]
                source = read(storage / "receipts/east" / (h + ".json"))
                require(obj["source_key"] == obj["destination_key"] == object_key(h)
                        and source["sha256"] == h and source["size"] == obj["size"]
                        and source["etag"] == obj["source_etag"], "OMS transfer does not bind the verified East object")
                service[h] = {"size":obj["size"], "source_verified_sha256":h, "oms":group["task"]}
    return {**service, **verified, **{p.stem: read(p) for p in single.glob("*.json")},
            **{p.stem: read(p) for p in (area / "other-inputs").glob("*.json")},
            **{p.stem: read(p) for p in (area / "regional-objects").glob("*.json")}}


def object_key(digest):
    return PREFIX + "/objects/sha256/" + digest


@lru_cache(maxsize=4)
def hot_paths(file):
    return read(file)["paths"]


def original_size(area, old, h):
    path = Path(old)
    if path.is_file():
        return path.stat().st_size
    regional = area / "regional-objects" / (h + ".json")
    if regional.exists():
        location = read(regional)
        require(location["sha256"] == location["verified_sha256"] == h and location["path"] == old,
                "relocated parent has no matching readback")
        return location["size"]
    storage = area.parent / "storage"
    candidate = hot_paths(storage / "hot.json").get(old, {})
    if candidate.get("sha256") == h and candidate["key"] == object_key(h):
        proof = read(storage / "receipts/east" / (h + ".json"))
        require(proof["sha256"] == h and candidate["size"] == proof["size"], "archived hot parent differs")
        return proof["size"]
    from pilotlab.checkpoint_archive3 import archive_proofs, archive_record, checkpoint_reads
    refs = archive_proofs(storage)
    with checkpoint_reads(refs, authoritative=True):
        archived = archive_record(old, h)
    require(archived and archived["key"] == object_key(h), "missing authoritative original parent: " + old)
    return archived["size"]


def descriptor(area, row, job, attempt, spec):
    path = Path(row["path"]).resolve(strict=True)
    body = path.read_bytes()
    t = json.loads(body)
    fingerprint = hashlib.sha256(body).hexdigest()
    require(fingerprint == row["fingerprint"] and t["id"] == row["id"] and
            t["device"].split(":")[0] == row["backend"], "task, resource binding or digest changed")
    reserved = row.get("reserved_evaluation_tokens", 0)
    plan = build_runtime_plan(path, prior_results=row.get("prior_results", ()), reserved_evaluation_tokens=reserved)
    plan_bytes = wire(plan)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    references = {}
    if t.get("resume_checkpoint"):
        old, h = t["resume_checkpoint"], t["resume_checkpoint_sha256"]
        reference_file(references, old, h, original_size(area, old, h))
    gate = t.get("gate_file") or t.get("stage_b_gate")
    auxiliary = {}
    if gate:
        g = read(gate)
        auxiliary[gate] = sha(gate)
        for field in ("manifest", "report"):
            auxiliary[g[field + "_path"]] = g[field + "_sha256"]
    auth = t.get("confirmation_authorization") or {}
    if auth.get("manifest_path"):
        auxiliary[auth["manifest_path"]] = auth["manifest_sha256"]
    for original, h in auxiliary.items():
        raw = Path(original).read_bytes()
        # Large shared stage manifests travel once per node as regular blobs.
        reference_file(references, original, h, len(raw), inline=raw if len(raw) < 128 * 1024 else None)
    start = t.get("evaluation_start_tokens", 0)
    if row.get("resume"):
        start = row["resume"]["steps"] * t["batch_size"] * t["sequence_length"]
    d = {"schema": "tide-cloud-dispatch/v1", "job": job, "attempt": attempt, "task": t["id"],
         "source": digest(t["source_identity"]), "original_path": str(path), "fingerprint": fingerprint,
         "task_bytes": base64.b64encode(body).decode(), "references": references,
         "runtime_plan_bytes": base64.b64encode(plan_bytes).decode(), "runtime_plan_sha256": plan_sha,
         "resume": row.get("resume"), "deadline_unix": spec["deadline_unix"],
         "canary": row.get("legacy_id") in CANARIES, "backend": row["backend"], "threads": row["threads"],
         "ram_mib": row["ram_mib"], "hbm_mib": row["hbm_mib"], "priority": priority(row),
         "estimated_seconds": row.get("estimated_seconds"), "start_tokens": start,
         "closing_tokens": t.get("eval_tokens", 0) * (2 if t.get("allow_confirmation") else 1),
         "closing_seconds": t.get("closing_reserve_seconds", 0)}
    d["reserve_chip"] = d["backend"] == "npu" and (
        row.get("reserve_chip", False) or d["hbm_mib"] >= 48 * 1024 or (d["estimated_seconds"] or 0) >= 4 * 3600)
    require(row.get("source", d["source"]) == d["source"], "task source mismatch")
    base = area / "payload/bases" / d["source"] / "base.json"
    if not base.is_file():
        base = area / "additional-bases/bases" / d["source"] / "base.json"
        if not base.is_file():
            launch = read(REPO / ".local/pilot-r3/launch.json")
            data = data_files(Path(read(launch["plan_path"])["config"]["data"]).resolve())
            make_base(area / "additional-bases", t["source_identity"], data)
        d["base_manifest"] = read(base)
    return d, plan


def objects_needed(area, d):
    objects = {r["sha256"]: (r["size"], r["path"]) for r in read(area / "data-objects.json")}
    for old, item in d["references"].items():
        if "bytes" not in item:
            objects[item["sha256"]] = (item["size"], old)
    if d["resume"]:
        endpoint = d["resume"]
        objects[endpoint["sha256"]] = (endpoint["size"], None)
    if d.get("base_manifest"):
        b = d["base_manifest"]
        for old, name in b["paths"].items():
            ref = b["files"][name]
            objects[ref["sha256"]] = (ref["size"], old)
    return objects


def stage_objects(area, region, job_prefix, objects, job, verified):
    cache = area / "staged" / job
    cache.mkdir(parents=True, exist_ok=True)
    def copy(entry):
        h, (size, _) = entry
        if (cache / h).is_file():
            return
        dest = job_prefix + "/inputs/objects/" + h
        proof = verified.get(h, {})
        if proof.get("source_verified_sha256") == h and not proof.get("etag"):
            # OMS completed this exact East input. Pin its regional copy now.
            # The worker hashes every body before admitting it to the training process.
            obj = region.head(object_key(h))
            require(obj and obj["size"] == size and obj.get("sha256") in (None, h), "unexpected incoming object")
            proof = {**proof, "etag":obj["etag"], "version":obj.get("version")}
        previous = region.head(dest)
        if previous and previous["size"] == size and previous.get("sha256") == h:
            meta = previous
        elif proof.get("etag") and proof["size"] == size:
            # Both single-part and multipart ETags pin the source version
            # whose complete bytes the incoming SHA-256 receipt already verified.
            service = {"store":region.side, "etag":proof["etag"], "version":proof.get("version"),
                       "operation":"tide-verified-sha256-" + h}
            try:
                meta = region.copy_verified(object_key(h), dest, None, size, service=service)
            except FileExistsError:
                meta = region.stream(dest, expected=h)
                require(meta["size"] == size, "conflicting staged object size")
        else:
            meta = region.copy_verified(object_key(h), dest, h, size)
        write(cache / h, {"sha256": h, "size": size, "key": dest, "etag": meta["etag"]})
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(copy, objects.items()))


def publish_claim(area, region, east, job_prefix, state, row, d, plan, *, transport=None):
    attempt, fp = d["attempt"], d["fingerprint"]
    location = area / "attempts" / attempt
    location.mkdir(parents=True, exist_ok=True)
    payload = wire(d)
    h = hashlib.sha256(payload).hexdigest()
    task = read(row["path"])
    evaluation_key = task.get("evaluation_budget_key")
    extra = plan["evaluation"]["additional_reservation"] if evaluation_key else 0
    prep_charge = 0
    if (task["phase"] == "preflight" and not row.get("prior_results") and not row.get("preparation_already_reserved")
            and task["id"] not in state["budget"]["preparation_prior"]):
        prep_charge = int(task.get("budget_tokens", task["target_tokens"] + 2*task.get("eval_tokens", 16384) + 32))
    claim = state["attempts"].get(attempt)
    if claim is None:
        require(not any(x["fingerprint"] == fp and x["status"] in {"claimed", "awaiting_submission", "dispatched", "accepted"}
                        for x in state["attempts"].values()), "already claimed or accepted fingerprint")
        if evaluation_key and state["budget"]["evaluation"].get(evaluation_key, 0) + extra > task["evaluation_token_limit"]:
            raise ValueError("evaluation_token_limit:" + evaluation_key)
        if state["budget"]["preparation"] + prep_charge > state["budget"]["preparation_limit"]:
            raise ValueError("preparation_token_limit")
        claim = {"task": row["id"], "fingerprint": fp, "job": d["job"], "attempt": attempt,
                 "backend": row["backend"], "row": row, "digest": h, "status": "claimed",
                 "reserved_evaluation_tokens": extra, "reserved_preparation_tokens": prep_charge}
        if transport:
            claim["transport"] = transport
        # An immutable East receipt also fences another restored client choosing a different node.
        key = PREFIX + "/claims/" + attempt + ".json"
        try:
            east.put_bytes(key, wire(claim))
        except FileExistsError:
            winner = json.loads(east.get_bytes(key))
            require(winner["task"] == row["id"] and winner["fingerprint"] == fp and winner["status"] == "claimed"
                    and winner["attempt"] == attempt and winner["reserved_evaluation_tokens"] == extra
                    and winner["reserved_preparation_tokens"] == prep_charge, "incompatible canonical attempt")
            claim = winner
        if evaluation_key:
            state["budget"]["evaluation"][evaluation_key] = state["budget"]["evaluation"].get(evaluation_key, 0) + extra
        state["budget"]["preparation"] += prep_charge
        state["attempts"][attempt] = claim
        write(area / "state.json", state)
    if claim["job"] != d["job"] or claim.get("transport") != transport:
        return claim
    require(claim["digest"] == h and claim["job"] == d["job"], "attempt dispatch bytes changed")
    (location / "runtime-plan.json").write_bytes(wire(plan))
    publications = [(east, PREFIX + "/attempts/" + attempt + "/runtime-plan.json", wire(plan)),
                    (east, PREFIX + "/attempts/" + attempt + "/task.json", base64.b64decode(d["task_bytes"], validate=True)),
                    (east, PREFIX + "/dispatch/" + attempt + ".json", payload)]
    if transport:
        publications.append((region, job_prefix + "/inputs/authority/" + attempt + ".json", payload))
    with ThreadPoolExecutor(max_workers=len(publications)) as pool:
        futures = [pool.submit(store.put_bytes, key, body) for store,key,body in publications]
        for future in futures:
            future.result()
    if not transport:
        region.put_bytes(job_prefix + "/inputs/dispatch/" + attempt + ".json", payload)
    claim["status"] = "awaiting_submission" if transport else "dispatched"
    claim["submitted"] = time.time()
    write(area / "state.json", state)
    return claim
