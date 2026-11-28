"""Publish the native campaign ledger and account for preparation on occupied physical boards."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from tide_cloud_allocation import PREFIX, REPO, wire
from tide_portable import read, write


def snapshot_campaign(area, east):
    source = area / "campaign/campaign"
    if not source.is_dir():
        return
    # Native task JSON and per-attempt runtime plans are already individually bound by each dispatch.
    paths = sorted(p for directory in (source, source / "decisions", source / "matches", source / "stage-b-capacity")
                   for p in directory.glob("*.json"))
    bodies = {p.relative_to(source).as_posix():p.read_bytes() for p in paths}
    hashes = {k:hashlib.sha256(v).hexdigest() for k,v in bodies.items()}
    revision = hashlib.sha256(wire(hashes)).hexdigest()
    pointer = area / "science-snapshot.json"
    if pointer.exists() and read(pointer)["revision"] == revision:
        return
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, content in bodies.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    raw = out.getvalue()
    h = hashlib.sha256(raw).hexdigest()
    key = PREFIX + "/science/snapshots/" + h + ".tgz"
    east.put_bytes(key, raw)
    rec = {"schema":"tide-cloud-science-snapshot/v1", "revision":revision, "files":hashes,
           "at":time.time(), "uri":east.uri(key), "size":len(raw), "sha256":h}
    east.put_bytes(PREFIX + "/science/current.json", wire(rec), overwrite=True)
    write(pointer, rec)


def preparation_budget(state, nodes, now):
    budget = read(REPO / ".local/pilot-r3/budget.json")
    old = sum(v.get("preparation_device_seconds", v.get("device_seconds", 0) if v.get("preparation") else 0)
              for v in budget["segments"].values())
    # Original runq charges the union of a preparation run's intervals on the same board.
    observed = state.setdefault("preparation_intervals", {})
    registrations = []
    for name, node in nodes.items():
        for worker in (node["queue"] or {}).get("tasks", []):
            claim = state["attempts"].get(worker["key"])
            if not claim or claim["row"]["phase"] != "preflight" or claim["backend"] != "npu":
                continue
            stage = worker.get("state")
            start = worker.get("started")
            finish = worker.get("finished")
            board = worker.get("physical_card")
            if start and board is not None:
                end = finish or min(now, node["queue"].get("at", now))
                observed[worker["key"]] = {"job":name, "board":board, "start":start, "end":end}
            if node.get("ready", True) and stage in {"queued", "reserved", "starting", "running"}:
                registrations.append({"job":name, "key":worker["key"], "state":stage, "board":board})
    intervals = defaultdict(list)
    for row in observed.values():
        intervals[row["job"], row["board"]].append((row["start"], row["end"]))
    total = 0.
    for spans in intervals.values():
        cursor = 0.
        for a,b in sorted(spans):
            total += max(0., b - max(cursor, a))
            cursor = max(cursor, b)
    return {"used_before_migration":old, "observed_physical_seconds":total,
            "remaining_seconds":budget["preparation_max_device_seconds"] - old - total,
            "registrations":registrations}
