"""Coordinate two independent ModelArts runq instances with one durable R3 task authority."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import http.client
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from tide_cloud_allocation import BOARDS, MAFLOW, PREFIX, PROJECT, REPO, Store, target_context, wire
from tide_cloud_evidence import preparation_budget, snapshot_campaign
from tide_cloud_dispatch import (CANARIES, descriptor, known_objects, object_key, objects_needed, priority,
                                 publish_claim_files, reserve_claim, stage_objects, starting_budget)
from tide_cloud_results import archive_outputs, canary, ingest
from tide_portable import read, require, sha, write
from ma_jobs import status_ref, stop_ref
from ma_obs import ObsError
import ma_oms

ACTIVE = {"claimed", "awaiting_submission", "dispatched"}
PUBLISHERS = 8


def stamp(value):
    if isinstance(value, (float, int)) and value > 0:
        return value / 1000 if value > 10**11 else value
    if isinstance(value, str) and value:
        try:
            return stamp(float(value))
        except ValueError:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return None


def snapshot(region, key):
    try:
        # A runq heartbeat can be replaced every 15 seconds. Read one atomic OBS
        # version; an initial HEAD followed by a conditional GET can race every update.
        with region.response("GET", key) as response:
            headers = region.metadata(response)
            body = response.read()
        require(len(body) == headers["size"], "incomplete operation snapshot")
        if headers["sha256"]:
            require(hashlib.sha256(body).hexdigest() == headers["sha256"], "operation snapshot digest differs")
        return json.loads(body)
    except ObsError as exc:
        if exc.status == 404:
            return None
        raise


def queue_over_ssh(name, platform_id):
    """Use maflow's already host-key-pinned access to read the actual job-local cluster."""
    from tide_cloud_submit import pinned_ssh
    pinned = pinned_ssh({"name":name, "id":platform_id})
    if pinned is None:
        return None
    code = """import json,os,pathlib,re,subprocess,sys,time
name, job_id = sys.argv[1:]
if os.environ.get('MAFLOW_RUN') != name or os.environ.get('MAFLOW_WORKER') != 'worker-0':
    raise ValueError('wrong job or worker')
for field in ('MA_JOB_ID','MODELARTS_JOB_ID'):
    value = os.environ.get(field,'')
    if re.fullmatch('[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value) and value.lower() != job_id.lower():
        raise ValueError('wrong platform allocation')
work = pathlib.Path(os.environ['MAFLOW_WORK'])
reg = json.loads((work/'runq-project.json').read_bytes())
shared = pathlib.Path(os.environ['MAFLOW_SHARED'])/'tide-r3/runq'
if reg['mode'] != 'project' or pathlib.Path(reg['shared']) != shared:
    raise ValueError('wrong project queue')
data = json.loads(subprocess.check_output([os.environ['MAFLOW_RUNQ'],'--json','cluster','status','--shared',str(shared)],timeout=15))
counts = {}
for row in data['tasks']:
    counts[row['state']] = counts.get(row['state'],0) + 1
print(json.dumps(dict(at=time.time(), counts=counts, workers=data['workers'], tasks=data['tasks'],
    job=name, runq=os.environ['MAFLOW_RUNQ_COMMIT'], deadline=reg['deadline'])))
"""
    command = [*pinned, shlex.join(["python3","-I","-B","-c",code,name,platform_id])]
    try:
        r = subprocess.run(command, capture_output=True, timeout=24)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def observe_allocations(area, spec, region):
    def observe(name):
        directory = area / "allocations" / name
        manifest = directory / "runtime-allocation.json"
        if not manifest.exists():
            return name, None
        m, receipt = read(manifest), read(directory / "submitted.json")
        previous = directory / "last-platform.json"
        error = None
        platform_seen = None
        queue, queue_source = None, None
        try:
            with target_context("guiyang"):
                platform = status_ref(receipt["id"])
            require(platform["name"] == name and platform["nodes"] == 1
                    and platform["flavor"] == "modelarts.pool.visual.16xlarge", "allocation identity differs")
            platform_seen = time.time()
            write(previous, platform)
        except (ValueError, OSError, RuntimeError, KeyError, http.client.HTTPException) as exc:
            platform = read(previous) if previous.is_file() else {"phase":"Unknown", "id":receipt["id"], "name":name}
            platform_seen = previous.stat().st_mtime if previous.is_file() else None
            error = str(exc)
        if platform["phase"].lower() == "running":
            try:
                queue = queue_over_ssh(name, receipt["id"])
                queue_source = "job-local runq via maflow SSH"
                if queue is None:
                    queue = snapshot(region, m["prefix"] + "/products/tide-status.json")
                    queue_source = "OBS heartbeat"
                if queue:
                    require(queue["job"] == name and queue["runq"] == spec["runq_commit"], "queue release differs")
            except (ValueError, OSError, RuntimeError, KeyError, http.client.HTTPException) as exc:
                error = str(exc)
                queue = None
        now = time.time()
        phase = platform["phase"].lower()
        started = stamp(platform.get("started"))
        terminal = phase in {"completed", "failed", "terminated", "stopped", "killed", "deleted", "succeeded", "rejected", "lost"}
        elapsed = max(0., (platform.get("duration") or 0) / 1000) if started else 0.
        if started and not terminal:
            elapsed = max(elapsed, now - started)
        return name, {"name": name, "id": receipt["id"], "prefix": m["prefix"], "platform": platform,
                      "platform_observed_unix":platform_seen,
                      "platform_project": m["project"], "queue": queue, "queue_source":queue_source, "elapsed_seconds": elapsed,
                      "observation_error":error, "terminal": terminal,
                      "ready":bool(phase == "running" and queue and -5 <= now - queue["at"] < 150
                                   and any(w.get("state") == "serving" and -5 <= now-w.get("at",0) < 150
                                           for w in queue["workers"]))}
    with ThreadPoolExecutor(max_workers=len(spec["jobs"])) as pool:
        return {name:row for name,row in pool.map(observe, spec["jobs"]) if row is not None}


def stage_resource_report(area, spec, nodes):
    spent = sum(n["elapsed_seconds"] * BOARDS for n in nodes.values())
    active = sum(n["ready"] for n in nodes.values())
    allocated = sum(bool(n["platform"].get("started")) and not n["terminal"] for n in nodes.values())
    data = {"schema": "tide-cloud-physical-budget/v1", "observed_unix": time.time(),
            "remaining_device_seconds": max(0, spec["remaining_device_seconds"] - spent),
            "cloud_device_seconds": spent, "physical_boards_per_node": BOARDS,
            "deadline_unix": spec["deadline_unix"], "active_nodes": active, "allocated_physical_boards": allocated*BOARDS,
            "local_budget": str(REPO / ".local/pilot-r3/budget.json"),
            "node_capacity": {k:{"job": v["id"], "phase": v["platform"]["phase"], "workers": (v["queue"] or {}).get("workers", [])}
                              for k,v in nodes.items()}}
    write(area / "campaign/resources.json", data)
    return data


def idle_review_reason(dag, state, nodes, expected_nodes):
    """Release an idle allocation only for a settled gate, with all authorized capacity accounted for."""
    if any(c.get("needs_relay") or c["status"] == "unconfirmed" for c in state["attempts"].values()):
        return None
    if len(state["canaries"]) < len(CANARIES):
        for c in state["attempts"].values():
            if (c["row"].get("legacy_id") in CANARIES and c["status"] == "failed"
                    and c["fingerprint"] in state["accepted"]):
                return "canary_needs_review"
        return None
    if state["blocked"]:
        return "awaiting_scientific_review"
    if dag.get("blocker_kind") == "scientific" and dag["blocked"]:
        return "awaiting_scientific_review"
    if len(nodes) != expected_nodes or any(n.get("observation_error") or not (n["ready"] or n["terminal"])
                                          for n in nodes.values()):
        return None
    blockers = dag["blocked"]
    if not blockers or dag["phase"] == "stage-a-or-b-evidence":
        return None
    transient = ("awaiting measured idle chip", "cloud allocation", "fresh runq chip maps",
                 "no observed execution capacity")
    if any(str(reason).startswith(transient) for reason in blockers):
        return None
    return "awaiting_scientific_review"


def normalized(item):
    row = dict(item)
    path = Path(row.get("task_path") or row["path"]).resolve(strict=True)
    t = read(path)
    backend = t["device"].split(":")[0]
    row.update(path=str(path), id=t["id"], fingerprint=sha(path), phase=t["phase"], backend=backend)
    row.setdefault("threads", t["cpu_threads"])
    row.setdefault("ram_mib", math.ceil(t.get("reserved_rss_bytes", 0) / 2**20))
    row.setdefault("hbm_mib", math.ceil(t.get("reserved_hbm_bytes", 0) / 2**20) if backend == "npu" else 0)
    row.setdefault("prior_results", [])
    require(row["ram_mib"] > 0 and row["threads"] > 0 and (backend == "cpu" or row["hbm_mib"] > 0),
            "missing required native resource reservation for " + t["id"])
    return row


def queued_window(node, backend, threads):
    workers = node["queue"]["workers"]
    if backend == "cpu":
        return max(1, sum(w["capacity"]["threads"] for w in workers) // threads)
    chips = sum(len(c.get("chips", {})) for w in workers for c in w.get("cards", {}).values())
    return max(1, chips)


def restore_input(area, source, h, size, east, region):
    key = object_key(h)
    if not region.head(key):
        head = east.head(key)
        if head:
            require(head["size"] == size and head.get("sha256") in (None, h), "East parent metadata differs")
            if not head.get("sha256"):
                east.stream(key, expected=h, etag=head["etag"])
        else:
            source = Path(source)
            require(source.stat().st_size == size and sha(source) == h, "late stage input does not match its native evidence")
            east.put_file(key, source, sha256=h)
        task = ma_oms.submit("east", "guiyang", [key], operation_id="tide-r3-in-" + h[:44], project=PROJECT, hashes={key:h})
        ma_oms.wait(task, timeout=3600)
    head = region.stream(key, expected=h)
    require(head["size"] == size, "late stage input size mismatch")
    write(area / "other-inputs" / (h + ".json"), {"key":key, "sha256":h, "size":size,
                                                   "etag":head["etag"], "version":head.get("version")})


class Controller:
    def __init__(self, area, cache):
        self.area, self.cache = area, cache
        self.spec = read(area / "payload/production.json")
        self.east, self.region = Store.named("east"), Store.named("guiyang")
        state = area / "state.json"
        if state.exists():
            self.state = read(state)
        else:
            remote = snapshot(self.east, PREFIX + "/control/state.json")
            self.state = remote or {"schema": "tide-cloud-authority/v1", "project": PROJECT, "attempts": {},
                                    "accepted": {}, "canaries": {}, "retries": {}, "blocked": {}, "budget": starting_budget()}
            write(state, self.state)
        binding = sha(area / "payload/production.json")
        require(self.state["project"] == PROJECT and self.state.setdefault("production_sha256", binding) == binding,
                "restored authority state belongs to a different campaign")
        self.initial = read(area / "initial-tasks.json")
        self.hot = read(area.parent / "storage/hot.json")["objects"]
        self.background = ThreadPoolExecutor(max_workers=5)
        self.input_futures, self.archive_future = {}, None
        self.prestage_futures = {}
        self.stage_pool = ThreadPoolExecutor(max_workers=4)
        self.stage_futures = {}
        self.collect_pool = ThreadPoolExecutor(max_workers=6)
        self.collections = {}
        self.relay_work = {}
        self.science_future = None
        self.transport_cache = {}
        self.publication_pool = ThreadPoolExecutor(max_workers=PUBLISHERS)
        self.publications = {}
        self.submission_pool = ThreadPoolExecutor(max_workers=6)
        self.submissions = {}
        self.dag = {"phase": "stage-a-subjects", "finished":False, "blocked": [], "tasks": [], "selected_subjects":262}
        self.dag_version = None
        self.state.setdefault("drained", {})

    def receipts(self, nodes):
        state, area = self.state, self.area
        for attempt, future in list(self.collections.items()):
            if not future.done():
                continue
            del self.collections[attempt]
            try:
                updated, evidence = future.result()
            except Exception as exc:
                self.log_error("collect:" + attempt, exc)
                if isinstance(exc, (KeyError, FileNotFoundError, ValueError)):
                    state["attempts"][attempt]["needs_relay"] = True
                continue
            state["attempts"][attempt] = updated
            state["accepted"].update(evidence["accepted"])
            state["retries"].update(evidence["retries"])
            state.get("errors", {}).pop("collect:" + attempt, None)
            write(area / "state.json", state)

        for attempt, future in list(self.relay_work.items()):
            if future.done():
                claim = state["attempts"][attempt]
                self.reconcile(attempt, {"name":claim["job"]}, claim)
        for name, n in nodes.items():
            try:
                self.node_receipts(n)
                state.get("errors", {}).pop("receipts:" + name, None)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                self.log_error("receipts:" + name, exc)

    def collect(self, prefix, claim):
        updated = dict(claim)
        evidence = {"accepted": {}, "retries": {}}
        ingest(self.area, self.cache, self.region, self.east, prefix, updated, evidence)
        return updated, evidence

    def node_receipts(self, node):
        prefix = node["prefix"]
        claims = {a:c for a,c in self.state["attempts"].items() if c["job"] == node["name"]}
        for attempt, claim in claims.items():
            old = claim["row"].get("legacy_id")
            if old not in CANARIES or claim["status"] in {"claimed", "awaiting_submission"}:
                continue
            proof = self.state["canaries"].get(str(old))
            if proof and (proof.get("uri") or proof["attempt"] != attempt):
                continue
            try:
                canary(self.area, self.state, self.region, prefix, claim)
                proof = self.state["canaries"].get(str(old))
                if proof:
                    key = PREFIX + "/first-updates/" + attempt + ".json"
                    self.east.put_file(key, self.area / "attempts" / attempt / "first-update.json", sha256=proof["sha256"])
                    proof["uri"] = self.east.uri(key)
                self.state.get("errors", {}).pop("canary:" + attempt, None)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                self.log_error("canary:" + attempt, exc)
        if not any(c["status"] in ACTIVE | {"unconfirmed"} or c.get("needs_relay") for c in claims.values()):
            return
        published = {Path(o["key"]).stem for o in self.region.list(prefix + "/products/receipts/") if o["size"]}
        recovered = {Path(o["key"]).stem for o in self.region.list(prefix + "/products/reconciled/") if o["size"]}
        by_key = {t["key"]:t for t in (node["queue"] or {}).get("tasks", [])}
        for attempt,c in claims.items():
            recovery = "products/reconciled/" + attempt + ".json"
            if attempt in recovered and c.get("receipt_key") != recovery and c["status"] not in {"accepted","interrupted","not_started"}:
                c.update(receipt_key=recovery, status="dispatched", needs_relay=False)
            if attempt in self.collections:
                continue
            if attempt in published | recovered and not c.get("needs_relay"):
                if c["status"] in ACTIVE | {"unconfirmed"} and len(self.collections) < 12:
                    self.collections[attempt] = self.collect_pool.submit(self.collect, prefix, dict(c))
                continue
            q = by_key.get(attempt)
            if node["ready"] and q and q["state"] in {"completed", "failed", "blocked", "cancelled", "unconfirmed"}:
                self.reconcile(attempt, node, c, q)
            elif node["terminal"] and c["status"] in ACTIVE:
                c.update(status="unconfirmed", reason="platform terminal without verified native receipt")

    def reconcile(self, attempt, node, claim, queue_task=None):
        if claim["status"] in {"accepted", "interrupted", "not_started"}:
            self.relay_work.pop(attempt, None)
            return
        now = time.time()
        evidence = claim.setdefault("reconciliation", {"first_terminal":now, "tries":0})
        future = self.relay_work.get(attempt)
        if future is None:
            if (queue_task is None or now-evidence["first_terminal"] < 90 or now < evidence.get("check_after", 0)
                    or len(self.relay_work) >= 2):
                return
            from tide_cloud_relay import recover
            self.relay_work[attempt] = self.background.submit(recover, node, dict(claim), queue_task)
            return
        if not future.done():
            return
        del self.relay_work[attempt]
        try:
            result = future.result()
        except Exception as exc:
            result = {"status":"unavailable", "reason":str(exc)}
        evidence.update(tries=evidence["tries"] + 1, last=result, at=now)
        evidence["check_after"] = now + min(600, 30 * evidence["tries"])
        status = result["status"]
        if status == "published":
            claim.update(status="dispatched", needs_relay=False, receipt_key=result["receipt_key"],
                         receipt_sha256=result.get("sha256"))
        elif status == "never_started":
            row = claim["row"]
            prior = list(row.get("unstarted_on", []))
            if node["name"] in prior:
                claim.update(status="unconfirmed", needs_relay=False, reason="repeated pre-start failure on the same allocation")
                self.state["blocked"][row["fingerprint"]] = claim["reason"]
                return
            claim.update(status="not_started", needs_relay=False)
            self.state["retries"][row["fingerprint"]] = {
                **row, "reserved_evaluation_tokens":row.get("reserved_evaluation_tokens", 0) + claim["reserved_evaluation_tokens"],
                "preparation_already_reserved":True, "unstarted_on":[*prior, node["name"]]}
        elif status == "incomplete" or (status == "unavailable" and evidence["tries"] >= 3):
            claim.update(status="unconfirmed", reason="native attempt requires evidence reconciliation")
            self.log_error("relay:" + attempt, RuntimeError(str(result)))

    def log_error(self, key, exc):
        message = f"{type(exc).__name__}: {exc}"
        if self.state.setdefault("errors", {}).get(key) != message:
            print(key, message, flush=True)
        self.state["errors"][key] = message

    def candidates(self, nodes):
        rows = {r["fingerprint"]:r for r in self.initial}
        try:
            from pilotlab.cloud_campaign3 import advance
        except ModuleNotFoundError as exc:
            if exc.name != "pilotlab.cloud_campaign3":
                raise
            advance = None
        version = (len(self.state["accepted"]), tuple((k,n["ready"],n["terminal"]) for k,n in nodes.items()),
                   int(time.time()//300))
        if advance and self.dag_version != version:
            self.dag = advance(self.area / "campaign", self.area / "initial-tasks.json", self.state["accepted"])
            self.dag_version = version
        for t in self.dag["tasks"]:
            row = normalized(t)
            rows.setdefault(row["fingerprint"], row)
        rows.update(self.state["retries"])
        if len(self.state["canaries"]) < len(CANARIES):
            rows = {k:r for k,r in rows.items() if r.get("legacy_id") in CANARIES}
        skip = {c["fingerprint"] for c in self.state["attempts"].values() if c["status"] not in {"interrupted","not_started"}}
        skip.update(self.state["blocked"])
        return sorted([r for k,r in rows.items() if k not in skip], key=lambda r: (-priority(r),r["id"]))

    def inputs(self, expected, available):
        for h, (size, path) in expected.items():
            if h in available:
                continue
            if h in self.hot and self.hot[h]["priority"] < 3:
                return False
            if (self.area / "other-inputs" / (h + ".json")).exists():
                continue
            f = self.input_futures.get(h)
            if f and f.done():
                try:
                    f.result()
                except Exception as exc:
                    self.log_error("input:" + h, exc)
                self.input_futures.pop(h)
            if h not in self.input_futures:
                self.input_futures[h] = self.background.submit(restore_input, self.area, path, h, size, self.east, self.region)
            return False
        return True

    def transport(self, node):
        if (node.get("queue_source") == "job-local runq via maflow SSH" or any(
                c["job"] == node["name"] and c.get("transport") == "ssh" and
                (c.get("last_ingress") or {}).get("status") == "submitted" for c in self.state["attempts"].values())):
            return "ssh"
        cached = self.transport_cache.get(node["id"])
        if cached and time.monotonic() < cached[1]:
            return cached[0]
        # Only a positive machine-authenticated probe selects the private SFS route.
        # A node reachable solely through its job mount can accept native OBS descriptors.
        from tide_cloud_submit import ssh_ready
        mode = "ssh" if ssh_ready(node) else "obs"
        self.transport_cache[node["id"]] = mode, time.monotonic() + 300
        return mode

    def deliver(self, nodes):
        from tide_cloud_submit import refresh_native, submit_batch, submit_native
        self.finish_publications()
        for slot,future in list(self.submissions.items()):
            if not future.done():
                continue
            del self.submissions[slot]
            try:
                reply = future.result()
            except Exception as exc:
                self.log_error("ingress:" + slot[0], exc)
                continue
            self.state.get("errors", {}).pop("ingress:" + slot[0], None)
            for attempt, result in reply.items():
                c = self.state["attempts"][attempt]
                if slot[1] == "refresh":
                    c["last_native_refresh"] = result
                    c["retry_native_after"] = time.time() + 90
                    continue
                if c["status"] != "awaiting_submission":
                    continue
                c["last_ingress"] = result
                if result["status"] == "submitted":
                    c.update(status="dispatched", submitted=time.time())
                elif result["status"] == "rejected":
                    c.update(status="failed", reason=result["reason"])
                else:
                    c["retry_ingress_after"] = time.time() + 20
                    self.transport_cache.pop(nodes[slot[0]]["id"], None)
            write(self.area / "state.json", self.state)
        in_flight = {attempt for _, _, batch in self.submissions for attempt in batch}
        for name,node in nodes.items():
            if not node["ready"]:
                continue
            visible = {task["key"] for task in node["queue"]["tasks"]}
            per_mode = {"ssh":[], "obs":[], "refresh":[]}
            for attempt,c in self.state["attempts"].items():
                if c["job"] != name or attempt in in_flight:
                    continue
                if c.get("transport") == "obs" and c["status"] == "dispatched" and attempt not in visible:
                    if (node.get("queue_source") == "job-local runq via maflow SSH" and
                            time.time() >= c.get("retry_native_after", c["submitted"] + 60)):
                        per_mode["refresh"].append(dict(c))
                    continue
                if c["status"] != "awaiting_submission":
                    continue
                if attempt in visible:
                    c.update(status="dispatched", submitted=time.time())
                elif time.time() >= c.get("retry_ingress_after", 0):
                    per_mode[c["transport"]].append(dict(c))
            for mode,items in per_mode.items():
                slots = (2 if mode == "ssh" else 1) - sum(k[:2] == (name,mode) for k in self.submissions)
                if not items or slots <= 0:
                    continue
                # One slow checkpoint or regional request should hold only its own small batch.
                ordered = sorted(items, key=lambda c:-priority(c["row"]))
                func, args = {"ssh":(submit_batch, (self.area,)), "obs":(submit_native, (self.region,)),
                              "refresh":(refresh_native, (self.area,))}[mode]
                for start in range(0, min(len(ordered), slots*6), 6):
                    batch = ordered[start:start+6]
                    slot = name, mode, tuple(c["attempt"] for c in batch)
                    self.submissions[slot] = self.submission_pool.submit(func, node, batch, *args)

    def finish_publications(self):
        changed = False
        for attempt, future in list(self.publications.items()):
            if not future.done():
                continue
            del self.publications[attempt]
            claim = self.state["attempts"][attempt]
            try:
                publication = future.result()
            except Exception as exc:
                self.log_error("publish:" + attempt, exc)
                continue
            self.state.get("errors", {}).pop("publish:" + attempt, None)
            if claim["status"] == "claimed":
                claim.update(publication)
                self.state["retries"].pop(claim["fingerprint"], None)
                changed = True
        if changed:
            write(self.area / "state.json", self.state)

    def send(self, row, node, available, existing=None):
        error = "dispatch:" + node["name"] + ":" + row["fingerprint"]
        try:
            result = self.send_ready(row, node, available, existing)
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            self.log_error(error, exc)
            return False
        self.state.get("errors", {}).pop(error, None)
        return result

    def send_ready(self, row, node, available, existing):
        fp, name = row["fingerprint"], node["name"]
        if existing in self.publications or len(self.publications) >= PUBLISHERS:
            return False
        count = sum(c["fingerprint"] == fp for c in self.state["attempts"].values())
        attempt = existing or f"r3-{fp}-{count+1}"
        d, plan = descriptor(self.area, row, name, attempt, self.spec)
        objs = objects_needed(self.area, d)
        if not self.inputs(objs, available):
            return False
        if not all((self.area / "staged" / name / h).is_file() for h in objs):
            key = name, fp
            if key not in self.stage_futures and len(self.stage_futures) < 8:
                self.stage_futures[key] = self.stage_pool.submit(
                    stage_objects, self.area, self.region, node["prefix"], objs, name, available)
            return False
        mode = self.state["attempts"][attempt].get("transport") if existing else self.transport(node)
        try:
            claim = reserve_claim(self.area, self.east, self.state, row, d, plan, transport=mode)
        except ValueError as exc:
            if str(exc).startswith(("evaluation_token_limit", "preparation_token_limit")):
                self.state["blocked"][fp] = str(exc)
                return False
            raise
        if claim["status"] != "claimed" or claim["job"] != name or claim.get("transport") != mode:
            return False
        self.publications[attempt] = self.publication_pool.submit(
            publish_claim_files, self.area, self.region, self.east, node["prefix"], dict(claim), d, plan)
        return True

    def dispatch(self, nodes, rows):
        for (name,fp), future in list(self.stage_futures.items()):
            if future.done():
                del self.stage_futures[name,fp]
                try:
                    future.result()
                except Exception as exc:
                    self.log_error("staging:" + name + ":" + fp, exc)
                else:
                    self.state.get("errors", {}).pop("staging:" + name + ":" + fp, None)
        available = known_objects(self.area, (read(self.area.parent / "storage/hot-guiyang.json")
                                              if (self.area.parent / "storage/hot-guiyang.json").exists() else {"objects":{}})["objects"])
        if len(self.state["canaries"]) < len(CANARIES):
            common = {}
            for r in self.initial:
                if r.get("legacy_id") in CANARIES:
                    d, _ = descriptor(self.area, r, "prestage", "prestage", self.spec)
                    common.update({h:x for h,x in objects_needed(self.area,d).items() if h in available})
            for name,n in nodes.items():
                active = self.prestage_futures.get(name)
                if active is not None and active.done():
                    try:
                        active.result()
                    except Exception as exc:
                        self.log_error("prestage:" + name, exc)
                    self.prestage_futures.pop(name)
                missing = any(not (self.area / "staged" / name / h).is_file() for h in common)
                if not n["terminal"] and missing and name not in self.prestage_futures:
                    self.prestage_futures[name] = self.background.submit(
                        stage_objects, self.area, self.region, n["prefix"], common, name, available)
        ready = {k:v for k,v in nodes.items() if v["ready"] and k not in self.state["drained"]}
        if not ready:
            return
        next_observation = time.monotonic() + 40
        for a,c in list(self.state["attempts"].items()):
            if c["status"] == "claimed" and c["job"] in ready:
                self.send(c["row"], ready[c["job"]], available, a)
        counts, loads = Counter(), Counter()
        running = Counter((k,t["backend"]) for k,n in ready.items() for t in n["queue"]["tasks"]
                          if t["state"] == "running" and t.get("backend"))
        for c in self.state["attempts"].values():
            if c["status"] in ACTIVE:
                slot = c["job"], c["backend"]
                counts[slot] += 1
                loads[slot] += self.cost(c["row"])
        for row in rows:
            if time.monotonic() >= next_observation:
                break
            backend = row["backend"]
            # Keep at most one hardware-width wave waiting behind active workers.
            # Longer-running attempts stay on their owner while late nodes can claim unassigned work.
            usable = [k for k,n in ready.items()
                      if counts[k,backend] - running[k,backend] < queued_window(n, backend, row["threads"])]
            if not usable:
                continue
            usable = [k for k in usable if k not in row.get("unstarted_on", ())] or usable
            if len(self.state["canaries"]) < len(CANARIES):
                started = [c["job"] for c in self.state["attempts"].values() if c["row"].get("legacy_id") in CANARIES]
                chosen = started[0] if started and started[0] in usable else min(usable, key=lambda k:nodes[k]["platform"].get("started") or 0)
            else:
                chosen = min(usable, key=lambda k:(loads[k,backend], counts[k,backend], k))
            if self.send(row, ready[chosen], available):
                counts[chosen,backend] += 1
                loads[chosen,backend] += self.cost(row)
                self.deliver(nodes)

    @staticmethod
    def cost(row):
        fraction = max(1/16, row.get("hbm_mib", 0)/(64*1024)) if row["backend"]=="npu" else row["threads"]
        return (row.get("estimated_seconds") or 1800) * fraction

    def tick(self):
        nodes = observe_allocations(self.area, self.spec, self.region)
        resources = stage_resource_report(self.area, self.spec, nodes)
        self.receipts(nodes)
        self.deliver(nodes)
        if self.archive_future is None or self.archive_future.done():
            if self.archive_future is not None:
                try:
                    self.archive_future.result()
                except Exception as exc:
                    self.log_error("archive-outputs", exc)
            self.archive_future = self.background.submit(archive_outputs, self.area, self.east)
        rows = self.candidates(nodes)
        if self.science_future is None or self.science_future.done():
            if self.science_future is not None:
                try:
                    self.science_future.result()
                except Exception as exc:
                    self.log_error("publish-science", exc)
            self.science_future = self.background.submit(snapshot_campaign, self.area, self.east)
        prep = preparation_budget(self.state, nodes, time.time())
        if prep["remaining_seconds"] <= 0:
            for row in rows:
                if row["phase"] == "preflight" and row["backend"] == "npu":
                    self.state["blocked"][row["fingerprint"]] = "preparation_device_hour_limit"
            rows = [r for r in rows if r["fingerprint"] not in self.state["blocked"]]
            # The platform continues to serve unrelated workers. Address exactly the
            # preparation cluster key that owns the exhausted experiment budget.
            for registration in prep["registrations"]:
                key = registration["key"]
                if self.state.setdefault("preparation_stops", {}).get(key):
                    continue
                code = ("import json,os,subprocess,sys; "
                        "p=json.load(open(os.environ['MAFLOW_RUNQ_QUEUE_FILE'])); "
                        "subprocess.run([os.environ['MAFLOW_RUNQ'],'cluster','cancel','--shared',p['shared'],"
                        "sys.argv[1]],check=True)")
                ref = "guiyang/" + nodes[registration["job"]]["id"]
                result = subprocess.run([sys.executable, "-B", str(MAFLOW), "job", "exec", ref, "--",
                                         "python3", "-c", code, key], capture_output=True, timeout=90)
                if result.returncode == 0:
                    self.state["preparation_stops"][key] = time.time()
        deadline = resources["deadline_unix"] - time.time()
        exhausted = deadline <= 360 or resources["remaining_device_seconds"] <= 0
        if not exhausted and not self.dag["finished"]:
            self.dispatch(nodes, rows)
        counts = Counter(c["status"] for c in self.state["attempts"].values())
        terminal = not any(counts[k] for k in ACTIVE)
        reason = "completed" if self.dag["finished"] else "budget" if exhausted else None
        if not reason and terminal and not rows and not self.relay_work and not self.collections:
            reason = idle_review_reason(self.dag, self.state, nodes, len(self.spec["jobs"]))
        for k,n in nodes.items():
            if n["terminal"] or k in self.state["drained"]:
                continue
            tail = (self.dag["phase"] == "stage-b-confirmation" and n["ready"] and not rows
                    and not self.dag["blocked"] and not self.relay_work and not self.collections
                    and counts["dispatched"] > 0 and not any(
                        c["job"] == k and (c["status"] in ACTIVE | {"unconfirmed"} or c.get("needs_relay"))
                        for c in self.state["attempts"].values()))
            if reason or tail:
                self.region.put_bytes(n["prefix"]+"/control/drain-tide.json", wire({"reason":reason or "confirmation_tail"}))
                if n["platform"]["phase"].lower() in {"pending", "waiting", "queuing"}:
                    with target_context("guiyang"):
                        stop_ref(n["id"])
                self.state["drained"][k]=reason or "confirmation_tail"
        samples = [n["platform_observed_unix"] for n in nodes.values() if n["platform_observed_unix"]]
        snapshot = {"observed_unix":min(samples, default=time.time()), "published_unix":time.time(),
                    "phase":self.dag["phase"], "finished": self.dag["finished"],
                    "selected_subjects":self.dag.get("selected_subjects"), "selected_controls":self.dag.get("selected_controls"),
                    "nodes": nodes, "physical_budget": resources, "counts":dict(counts), "canaries":self.state["canaries"],
                    "pending_candidates":len(rows), "initial_known_tasks":len(self.initial),
                    "native_ready_tasks":len(self.dag["tasks"]), "original_signals":sum(bool(r["resume"]) for r in self.initial), "gate_blockers": self.dag["blocked"], "budget_blockers":self.state["blocked"],
                    "preparation_budget":prep, "errors":self.state.get("errors",{}), "drained":self.state["drained"]}
        write(self.area / "state.json", self.state)
        write(self.area / "status.json", snapshot)
        for name in ("state", "status"):
            self.east.put_file(PREFIX + "/control/" + name + ".json", self.area / (name + ".json"), overwrite=True)
        settled = (len(nodes) == len(self.spec["jobs"]) and all(n["terminal"] for n in nodes.values()) and
                   terminal and not self.collections and not self.relay_work and
                   all((self.area / "east-objects" / p.name).exists() for p in (self.area / "regional-objects").glob("*.json")))
        if settled:
            self.science_future.result()
            if self.dag["finished"]:
                final = self.area / "campaign/campaign/report.json"
                require(read(self.area / "science-snapshot.json")["files"].get("report.json") == sha(final),
                        "the final scientific report has not yet been archived")
        return settled


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--checkpoint-cache", type=Path, required=True)
    cli.add_argument("--once", action="store_true")
    args=cli.parse_args(); area=args.control.resolve(); cache=args.checkpoint_cache.resolve()
    cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (area / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ctl=Controller(area, cache)
        while True:
            try:
                end = ctl.tick()
                if args.once or end:
                    break
            except Exception:
                traceback.print_exc()
                if args.once:
                    raise
            time.sleep(15)
