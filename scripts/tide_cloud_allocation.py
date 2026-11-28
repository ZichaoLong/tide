"""Freeze the R3 worker bundle and allocate the two independently schedulable Guiyang nodes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tide_portable import read, require, sha, write

REPO = Path(os.environ.get("TIDE_CLOUD_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
MAFLOW = Path(shutil.which("maflow") or Path.home() / ".local/bin/maflow").resolve(strict=True)
sys.path.insert(0, str(MAFLOW.parent))
from ma_locations import target_context
from ma_obs import Store
import job_registry
from ma_jobs import job_prefix

PREFIX = "tide-learning-lab/production/r3-20261128"
PROJECT = "tide-learning-lab-c1dd49b83311"
NAMES = ["tide-r3-20261128-1", "tide-r3-20261128-2"]
BOARDS = 8


def wire(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def freeze(area, runq_manifest):
    budget = read(REPO / ".local/pilot-r3/budget.json")
    release = read(runq_manifest)
    files = [REPO / "scripts" / name for name in ("tide_cloud_node.py", "tide_batch.py", "tide_portable.py")]
    files.append(REPO / "pilotlab/runtime_migration3.py")
    payload = area / "payload"
    require((payload / "bases").is_dir(), "prepare the original frozen model and data manifests first")
    meta = payload / "production.json"
    remaining = budget["max_device_seconds"] - budget["device_seconds"]
    if meta.exists():
        previous = read(meta)
        require(release["sha256"] == previous["runq_sha256"] and all(
            sha(payload / name) == h for name, h in previous["files"].items()), "already frozen payload changed")
        return previous
    for p in files:
        shutil.copyfile(p, payload / p.name)
    src = {f.relative_to(payload).as_posix(): sha(f) for f in sorted(payload.rglob("*")) if f.is_file()}
    # Both independent ceilings combined fit the entire unspent campaign budget.
    # Use whole platform hours, including its automatic coarse-grained stop.
    hours = int(remaining // (len(NAMES) * BOARDS * 3600))
    require(hours > 0, "campaign has no allocatable node hours")
    spec = {"schema": "tide-cloud-production/v1", "project": PROJECT, "jobs": NAMES,
            "boards_per_job": BOARDS, "region": "guiyang", "spec": "r3-node8",
            "runq_commit": release["commit"], "runq_sha256": release["sha256"], "files": src,
            "deadline_unix": budget["deadline_unix"], "remaining_device_seconds": remaining,
            "run_seconds": hours * 3600, "queue_seconds": max(1, int(budget["deadline_unix"] - time.time())),
            "initial_tasks_sha256": sha(area / "initial-tasks.json"), "data_objects_sha256": sha(area / "data-objects.json")}
    write(meta, spec)
    return spec


def allocation(area, spec, release, name):
    path = area / "allocations" / name
    path.mkdir(parents=True, exist_ok=True)
    with target_context("guiyang"):
        store, east = Store.named("guiyang"), Store.named("east")
        opts = [sys.executable, "-B", str(MAFLOW), "job", "submit", "--json", "--target", "guiyang", "--spec", spec["spec"],
                "--nodes", "1", "--name", name, "--for", str(spec["run_seconds"]), "--queue", str(spec["queue_seconds"]),
                "--runq-manifest", str(release), "--payload", str(area / "payload")]
        program = ["--", "python3", "-B", "-u", "tide_cloud_node.py", "serve"]
        def run(options):
            c = subprocess.run(opts + options + program, capture_output=True, text=True)
            if c.returncode:
                raise RuntimeError(name + " " + (c.stderr or c.stdout)[-2500:])
            return json.loads(c.stdout)
        if not (path / "preview.json").exists():
            write(path / "preview.json", run(["--dry-run"]))
        submitted = path / "submitted.json"
        if not submitted.exists():
            receipt = run([])
            require(receipt.get("id") or receipt.get("job_id") or receipt.get("submitted"), str(receipt))
            write(submitted, receipt)
        prepared = job_registry.prepared(name, store)
        require(prepared is not None, "platform receipt missing its frozen job manifest")
        m = prepared["manifest"]
        require(m["nodes"] == 1 and m["runq"]["sha256"] == spec["runq_sha256"] and m["name"] == name,
                "unexpected platform allocation")
        write(path / "runtime-allocation.json", m)
        store.put_file(job_prefix(name) + "/inputs/runtime-allocation.json", path / "runtime-allocation.json")
        for p in path.glob("*.json"):
            east.put_file(PREFIX + "/allocations/" + name + "/" + p.name, p)
        print(name, json.dumps(read(submitted), ensure_ascii=False), flush=True)


def provision(area, release):
    spec = freeze(area, release)
    east = Store.named("east")
    for p in (area / "initial-tasks.json", area / "data-objects.json", area / "payload/production.json"):
        east.put_file(PREFIX + "/control/" + p.name, p)
    # Keep a reconstructible standalone worker bundle in the authoritative region.
    from ma_batch_files import package
    archive = area / "worker-bundle.tgz"
    if not archive.exists():
        tmp = archive.with_suffix(".partial")
        tmp.unlink(missing_ok=True)
        package(area / "payload", tmp, pin_runq=True, runq_manifest=release)
        tmp.rename(archive)
    h = sha(archive)
    east.put_file(PREFIX + "/control/worker-bundle-" + h + ".tgz", archive, sha256=h)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(NAMES)) as pool:
        for task in concurrent.futures.as_completed([pool.submit(allocation, area, spec, release, name) for name in NAMES]):
            try:
                task.result()
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError("\n".join(errors))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", required=True, type=Path)
    cli.add_argument("--runq-manifest", required=True, type=Path)
    cli.add_argument("--freeze-only", action="store_true")
    args = cli.parse_args()
    if args.freeze_only:
        print(json.dumps(freeze(args.control.resolve(), args.runq_manifest.resolve()), ensure_ascii=False))
    else:
        provision(args.control.resolve(), args.runq_manifest.resolve())
