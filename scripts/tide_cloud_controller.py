"""Coordinate two independent ModelArts runq instances with one durable R3 task authority."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import traceback

from tide_cloud_allocation import BOARDS, PREFIX, PROJECT, REPO, Store, target_context, wire
from tide_cloud_dispatch import (CANARIES, descriptor, known_objects, object_key, objects_needed, priority,
                                 publish_claim, stage_objects, starting_budget)
from tide_cloud_results import archive_outputs, canary, ingest
from tide_portable import read, require, sha, write
from ma_jobs import status_ref, stop_ref
import ma_oms


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
        return json.loads(region.get_bytes(key))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def observe_allocations(area, spec, region):
    nodes, now = {}, time.time()
    for name in spec["jobs"]:
        directory = area / "allocations" / name
        manifest = directory / "runtime-allocation.json"
        if not manifest.exists():
            continue
        m, receipt = read(manifest), read(directory / "submitted.json")
        with target_context("guiyang"):
            platform = status_ref(receipt["id"])
        require(platform["name"] == name and platform["nodes"] == 1 and platform["flavor"] == "modelarts.pool.visual.16xlarge",
                "an allocation no longer matches its request")
        status = snapshot(region, m["prefix"] + "/products/tide-status.json")
        if status:
            require(status["job"] == name and status["runq"] == spec["runq_commit"], "unexpected active queue")
        phase = platform["phase"].lower()
        started = stamp(platform.get("started"))
        terminal = phase in {"completed", "failed", "terminated", "stopped", "killed", "deleted", "succeeded", "rejected", "lost"}
        # ModelArts duration is in milliseconds and can lag while the job is running.
        elapsed = max(0., (platform.get("duration") or 0) / 1000) if started else 0.
        if started and not terminal:
            elapsed = max(elapsed, now - started)
        nodes[name] = {"name": name, "id": receipt["id"], "prefix": m["prefix"], "platform": platform,
                       "platform_project": m["project"], "queue": status, "elapsed_seconds": elapsed,
                       "terminal": terminal, "ready": bool(status and status["workers"] and now-status["at"]<150
                                                          and phase == "running" and not terminal)}
    return nodes


def stage_resource_report(area, spec, nodes):
    spent = sum(n["elapsed_seconds"] * BOARDS for n in nodes.values())
    active = sum(n["ready"] for n in nodes.values())
    data = {"schema": "tide-cloud-physical-budget/v1", "observed_unix": time.time(),
            "remaining_device_seconds": max(0, spec["remaining_device_seconds"] - spent),
            "cloud_device_seconds": spent, "physical_boards_per_node": BOARDS,
            "deadline_unix": spec["deadline_unix"], "active_nodes": active, "allocated_physical_boards": active*BOARDS,
            "local_budget": str(REPO / ".local/pilot-r3/budget.json"),
            "node_capacity": {k:{"job": v["id"], "phase": v["platform"]["phase"], "workers": (v["queue"] or {}).get("workers", [])}
                              for k,v in nodes.items()}}
    write(area / "campaign/resources.json", data)
    return data


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


def restore_input(area, source, h, size, east, region):
    key = object_key(h)
    if not region.head(key):
        source = Path(source)
        require(source.stat().st_size == size and sha(source) == h, "late stage input does not match its native evidence")
        east.put_file(key, source, sha256=h)
        task = ma_oms.submit("east", "guiyang", [key], operation_id="tide-r3-in-" + h[:44], project=PROJECT, hashes={key:h})
        ma_oms.wait(task, timeout=3600)
    head = region.stream(key, expected=h)
    require(head["size"] == size, "late stage input size mismatch")
    write(area / "other-inputs" / (h + ".json"), {"key":key, "sha256":h, "size":size})


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
        self.initial = read(area / "initial-tasks.json")
        self.hot = read(area.parent / "storage/hot.json")["objects"]
        self.background = ThreadPoolExecutor(max_workers=5)
        self.input_futures, self.archive_future = {}, None
        self.dag = {"phase": "stage-a-subjects", "finished":False, "blocked": [], "tasks": [], "selected_subjects":262}
        self.dag_version = None
        self.state.setdefault("drained", {})

    def receipts(self, nodes):
        state, area = self.state, self.area
        for name, n in nodes.items():
            if not any(c["job"] == name and c["status"] in {"dispatched", "claimed"} for c in state["attempts"].values()):
                continue
            prefix = n["prefix"]
            for claim in list(state["attempts"].values()):
                if claim["job"] == name and claim["row"].get("legacy_id") in CANARIES and claim["status"] != "claimed":
                    canary(area, state, self.region, prefix, claim)
                    proof = state["canaries"].get(str(claim["row"]["legacy_id"]))
                    if proof and not proof.get("uri"):
                        key = PREFIX + "/first-updates/" + claim["attempt"] + ".json"
                        self.east.put_file(key, area / "attempts" / claim["attempt"] / "first-update.json", sha256=proof["sha256"])
                        proof["uri"] = self.east.uri(key)
            published = set()
            for obj in self.region.list(prefix + "/products/receipts/"):
                a = Path(obj["key"]).stem
                published.add(a)
                c = state["attempts"].get(a)
                if not c or c["status"] not in {"claimed", "dispatched"} or not obj["size"]:
                    continue
                try:
                    ingest(area, self.cache, self.region, self.east, prefix, c, state)
                except Exception as exc:
                    self.log_error("collect:" + a, exc)
                    continue
                write(area / "state.json", state)
            # No task wrapper can publish a receipt after the node has terminated.
            if n["terminal"]:
                for a,c in state["attempts"].items():
                    if c["job"] == name and a not in published and c["status"] in {"claimed", "dispatched"}:
                        c.update(status="unconfirmed", reason="platform terminal without verified native receipt")

    def log_error(self, key, exc):
        message = f"{type(exc).__name__}: {exc}"
        if self.state.setdefault("errors", {}).get(key) != message:
            print(key, message, flush=True)
        self.state["errors"][key] = message

    def candidates(self):
        rows = {r["fingerprint"]:r for r in self.initial}
        try:
            from pilotlab.cloud_campaign3 import advance
        except ModuleNotFoundError as exc:
            if exc.name != "pilotlab.cloud_campaign3":
                raise
            advance = None
        version = (len(self.state["accepted"]), int(time.time()//300))
        if advance and self.dag_version != version:
            self.dag = advance(self.area / "campaign", self.area / "initial-tasks.json", self.state["accepted"])
            self.dag_version = version
        for t in self.dag["tasks"]:
            row = normalized(t)
            rows.setdefault(row["fingerprint"], row)
        if len(self.state["canaries"]) < len(CANARIES):
            rows = {k:r for k,r in rows.items() if r.get("legacy_id") in CANARIES}
        rows.update(self.state["retries"])
        skip = {c["fingerprint"] for c in self.state["attempts"].values() if c["status"] != "interrupted"}
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

    def send(self, row, node, available, existing=None):
        fp, name = row["fingerprint"], node["name"]
        count = sum(c["fingerprint"] == fp for c in self.state["attempts"].values())
        attempt = existing or f"r3-{fp}-{count+1}"
        d, plan = descriptor(self.area, row, name, attempt, self.spec)
        objs = objects_needed(self.area, d)
        if not self.inputs(objs, available):
            return False
        stage_objects(self.area, self.region, node["prefix"], objs, name, available)
        try:
            publish_claim(self.area, self.region, self.east, node["prefix"], self.state, row, d, plan)
        except ValueError as exc:
            if str(exc).startswith(("evaluation_token_limit", "preparation_token_limit")):
                self.state["blocked"][fp] = str(exc)
                return False
            raise
        self.state["retries"].pop(fp, None)
        return True

    def dispatch(self, nodes, rows):
        available = known_objects(self.area, (read(self.area.parent / "storage/hot-guiyang.json")
                                              if (self.area.parent / "storage/hot-guiyang.json").exists() else {"objects":{}})["objects"])
        if len(self.state["canaries"]) < len(CANARIES):
            common = {}
            for r in self.initial:
                if r.get("legacy_id") in CANARIES:
                    d, _ = descriptor(self.area, r, "prestage", "prestage", self.spec)
                    common.update({h:x for h,x in objects_needed(self.area,d).items() if h in available})
            for name,n in nodes.items():
                if not n["terminal"]:
                    stage_objects(self.area, self.region, n["prefix"], common, name, available)
        ready = {k:v for k,v in nodes.items() if v["ready"] and k not in self.state["drained"]}
        if not ready:
            return
        for a,c in list(self.state["attempts"].items()):
            if c["status"] == "claimed" and c["job"] in ready:
                self.send(c["row"], ready[c["job"]], available, a)
        counts, loads = Counter(), Counter()
        for c in self.state["attempts"].values():
            if c["status"] in {"claimed","dispatched"}:
                slot = c["job"], c["backend"]
                counts[slot] += 1
                loads[slot] += self.cost(c["row"])
        for row in rows:
            backend = row["backend"]
            usable = [k for k,n in ready.items() if counts[k,backend] < max(
                2, 2 * n["queue"]["workers"][0]["capacity"]["threads"] // row["threads"])]
            if not usable:
                continue
            if len(self.state["canaries"]) < len(CANARIES):
                started = [c["job"] for c in self.state["attempts"].values() if c["row"].get("legacy_id") in CANARIES]
                chosen = started[0] if started and started[0] in usable else min(usable, key=lambda k:nodes[k]["platform"].get("started") or 0)
            else:
                chosen = min(usable, key=lambda k:(loads[k,backend], counts[k,backend], k))
            if self.send(row, ready[chosen], available):
                counts[chosen,backend] += 1
                loads[chosen,backend] += self.cost(row)

    @staticmethod
    def cost(row):
        fraction = max(1/16, row.get("hbm_mib", 0)/(64*1024)) if row["backend"]=="npu" else row["threads"]
        return (row.get("estimated_seconds") or 1800) * fraction

    def tick(self):
        nodes = observe_allocations(self.area, self.spec, self.region)
        resources = stage_resource_report(self.area, self.spec, nodes)
        self.receipts(nodes)
        if self.archive_future is None or self.archive_future.done():
            if self.archive_future is not None:
                try:
                    self.archive_future.result()
                except Exception as exc:
                    self.log_error("archive-outputs", exc)
            self.archive_future = self.background.submit(archive_outputs, self.area, self.east)
        rows = self.candidates()
        deadline = resources["deadline_unix"] - time.time()
        exhausted = deadline <= 360 or resources["remaining_device_seconds"] <= 0
        if not exhausted and not self.dag["finished"]:
            self.dispatch(nodes, rows)
        counts = Counter(c["status"] for c in self.state["attempts"].values())
        terminal = not any(counts[k] for k in ("claimed", "dispatched"))
        reason = "completed" if self.dag["finished"] else "budget" if exhausted else None
        if terminal and not rows and (self.dag["blocked"] or self.state["blocked"]):
            reason = "awaiting_scientific_review"
        if terminal and len(self.state["canaries"]) < len(CANARIES) and any(
                c["status"] in {"failed","unconfirmed"} and c["row"].get("legacy_id") in CANARIES for c in self.state["attempts"].values()):
            reason = "canary_needs_review"
        for k,n in nodes.items():
            if n["terminal"] or k in self.state["drained"]:
                continue
            tail = self.dag["phase"] == "stage-b-confirmation" and not rows and not any(
                c["job"] == k and c["status"] in {"claimed", "dispatched"} for c in self.state["attempts"].values())
            if reason or tail:
                self.region.put_bytes(n["prefix"]+"/control/drain-tide.json", wire({"reason":reason or "confirmation_tail"}))
                if n["platform"]["phase"].lower() in {"pending", "waiting", "queuing"}:
                    with target_context("guiyang"):
                        stop_ref(n["id"])
                self.state["drained"][k]=reason or "confirmation_tail"
        snapshot = {"observed_unix":time.time(), "phase":self.dag["phase"], "finished": self.dag["finished"],
                    "selected_subjects":self.dag.get("selected_subjects"), "selected_controls":self.dag.get("selected_controls"),
                    "nodes": nodes, "physical_budget": resources, "counts":dict(counts), "canaries":self.state["canaries"],
                    "pending_candidates":len(rows), "gate_blockers": self.dag["blocked"], "budget_blockers":self.state["blocked"],
                    "errors":self.state.get("errors",{}), "drained":self.state["drained"]}
        write(self.area / "state.json", self.state)
        write(self.area / "status.json", snapshot)
        for name in ("state", "status"):
            self.east.put_file(PREFIX + "/control/" + name + ".json", self.area / (name + ".json"), overwrite=True)
        return bool(nodes and all(n["terminal"] for n in nodes.values()) and terminal and
                    all((self.area / "east-objects" / p.name).exists() for p in (self.area / "regional-objects").glob("*.json")))


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
