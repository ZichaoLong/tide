"""Build content-addressed inputs for frozen R3 workers on single-node ModelArts jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3

from tide_portable import ident, read, require, sha, write


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def data_files(directory):
    manifest = read(directory / "manifest.json")
    require(manifest["schema_version"] == "tide-pilot-data-v3", "expected formal R3 data")
    corpus = Path(manifest["source_directory"]).resolve(strict=True)
    roots = {str(directory): "data", str(corpus): "corpus"}
    files = {}
    for root, prefix, table in ((directory, "data", manifest["files"]),
                                (corpus, "corpus", manifest["source_files"])):
        for name, item in {"manifest.json": {}, **table}.items():
            path = (root / name).resolve(strict=True)
            require(path.is_relative_to(root), "data member escaped its root")
            h = sha(path)
            require(item.get("sha256", h) == h and item.get("size_bytes", path.stat().st_size)
                    == path.stat().st_size, "data differs from manifest: " + name)
            files[str(path)] = (prefix + "/" + name, h, path.stat().st_size)
    require(files[str(corpus / "manifest.json")][1] == manifest["source_manifest_sha256"],
            "original corpus manifest differs")
    return roots, files


def make_base(payload, identity, data):
    key = digest(identity)
    base = payload / "bases" / key
    roots, data_items = data
    paths = {path: row[0] for path, row in data_items.items()}
    files = {name: {"sha256": h, "size": size} for name, h, size in data_items.values()}
    require(files["data/manifest.json"]["sha256"] == identity["data_manifest_sha256"],
            "source uses another dataset")
    roots = dict(roots)
    for path, expected in identity["source_files"].items():
        src = Path(path).resolve(strict=True)
        parent = next(p for p in src.parents if p.name in {"pilotlab", "tide"})
        prefix = "source/pilotlab" if parent.name == "pilotlab" else "foundation/src/tide"
        require(prefix not in roots.values() or roots.get(str(parent)) == prefix, "ambiguous source root")
        roots[str(parent)] = prefix
        member = prefix + "/" + src.relative_to(parent).as_posix()
        dest = base / member
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
        require(sha(dest) == expected, "frozen source digest differs: " + path)
        paths[path] = member
        files[member] = {"sha256": expected, "size": dest.stat().st_size}
    write(base / "base.json", {"schema": "tide-cloud-base/v1", "identity": key, "roots": roots,
                               "files": files, "paths": paths})
    return key


def queued_rows(launch):
    pause = read(Path(__file__).resolve().parents[1] /
                 "docs/evidence/pilot-r3-stage-a-checkpoint-pause-20261127.json")
    known = {}
    for group, receipt in pause["groups"].items():
        record = receipt["scheduler"]
        require(sha(record["path"]) == record["sha256"], "pause scheduler changed")
        scheduler = read(record["path"])
        owner = Path(receipt["scheduler_output"])
        for key, row in scheduler["tasks"].items():
            if row["status"] == "pending" or row["status"] == "stopped_budget" and row.get("reason") == "signal":
                path = Path(row.get("task_path") or owner / "tasks" / key / "task.json")
                known[row["runq_id"]] = (group, path, row)
    require(len(known) == 261, "pause record must cover 192 queued and 69 interrupted jobs")
    return known


def inventory(launch_file, pause_dir, output):
    launch = read(launch_file)
    records = queued_rows(launch)
    workers = read(pause_dir / "worker-receipts.json")["workers"]
    paused = read(pause_dir / "paused-checkpoints.json")["jobs"]
    evidence = read(Path(__file__).resolve().parents[1] /
                    "docs/evidence/pilot-r3-stage-a-checkpoint-pause-20261127.json")
    sizes = {group: read(Path(g["scheduler_output"]) / "model-resources.json")
             for group, g in evidence["groups"].items()}
    db = sqlite3.connect((Path(launch["runq_root"]) / "queue.sqlite3").as_uri()
                         + "?mode=ro&immutable=1", uri=True)
    data = data_files(Path(read(launch["plan_path"])["config"]["data"]).resolve())
    payload = output / "payload"
    bases, rows = set(), []
    for number, (group, file, row) in sorted(records.items()):
        task = read(file)
        require(task["id"] == row.get("task_id", file.parent.name), "frozen task location differs")
        src = task["source_identity"]
        identity = digest(src)
        if identity not in bases:
            make_base(payload, src, data)
            bases.add(identity)
        db_row = db.execute("SELECT spec FROM jobs WHERE id=?", (number,)).fetchone()
        require(db_row is not None, "old queue job missing")
        request = json.loads(db_row[0])
        backend = task["device"].split(":")[0]
        require(task["id"] == request["task_id"] and request["allowed"] == [backend],
                "old reservation belongs to different work")
        cp = paused.get(str(number))
        prior = []
        if cp:
            require(sha(file) == cp["original_task"]["sha256"], "paused task fingerprint changed")
            for attempt in row["attempts"]:
                r = Path(attempt["path"]) / "result.json"
                if r.exists() and read(r).get("stop_reason") == "signal":
                    prior.append(str(r))
            require(prior and prior[-1] == cp["result"]["path"] and sha(prior[-1]) == cp["result"]["sha256"],
                    "paused endpoint does not agree with attempt history")
        profile = sizes[group].get(task["model_id"], {})
        rate = profile.get("tokens_per_second_by_batch", {}).get(str(task["batch_size"]))
        if cp:
            remaining = workers[str(number)]["original_estimate_seconds"]
        elif rate and rate > 0:
            remaining = (task["target_tokens"] - task.get("evaluation_start_tokens", 0)
                         + 2 * task["eval_tokens"]) / rate + task.get("closing_reserve_seconds", 0)
        else:
            remaining = None
        rows.append({"id": task["id"], "legacy_id": number, "group": group, "path": str(file),
                     "fingerprint": sha(file), "source": identity, "phase": task["phase"],
                     "hbm_mib": math.ceil(request["hbm"]/2**20) if backend == "npu" else 0,
                     "ram_mib": math.ceil(request["ram"][backend]/2**20), "threads": task["cpu_threads"],
                     "backend": backend, "estimated_seconds": math.ceil(remaining) if remaining else None,
                     "prior_results": prior, "reserved_evaluation_tokens": row.get("evaluation_reserved_tokens", 0),
                     "resume": ({"sha256": cp["sha256"], "size": cp["size_bytes"], "steps": cp["steps"]}
                                if cp else None)})
    db.close()
    write(output / "initial-tasks.json", rows)
    write(output / "data-objects.json", [{"path": p, "sha256": x[1], "size": x[2]}
                                        for p, x in data[1].items()])
    return rows


def reference_file(record, original, h, size, *, inline=None):
    label = hashlib.sha256(original.encode()).hexdigest()[:20]
    name = "evidence/" + label + "/" + Path(original).name
    entry = {"name": name, "sha256": h, "size": size}
    if inline is not None:
        require(hashlib.sha256(inline).hexdigest() == h, "inline evidence changed")
        entry["bytes"] = base64.b64encode(inline).decode()
    record[original] = entry


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--launch", required=True, type=Path)
    cmd.add_argument("--pause", required=True, type=Path)
    cmd.add_argument("--output", required=True, type=Path)
    args = cmd.parse_args()
    tasks = inventory(args.launch, args.pause, args.output)
    print(json.dumps({"tasks": len(tasks), "paused": sum(bool(t["prior_results"]) for t in tasks)}))
