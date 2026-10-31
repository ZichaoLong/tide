#!/usr/bin/env python3
"""Bounded public-run follow-up: eight-card inference, small train, gated 1B train.

Run inside the site's cooperative device allocation, from frozen source with
an installed tidegraph. Large checkpoint data stays in admitted tmpfs; only
small records are copied to persistent output, including failed records.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tidegraph.experiment.container_job import prepare, supervise, inspect
from tidegraph.experiment.identity import capture
from tidegraph.experiment.matrix import resolve_matrix
from tidegraph.experiment.records import digest, now, write_json
from tidegraph.experiment.report import load, render, save


def quota_blocks():
    """Use the actual account quota, never the entire filesystem's df free."""
    result = subprocess.run(["quota", "-w", "-v"], text=True, capture_output=True, timeout=15)
    if result.returncode not in (0, 1):
        raise RuntimeError("quota inspection failed: " + result.stderr[-500:])
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0].startswith("/dev/") and len(fields) >= 4:
            rows.append(dict(filesystem=fields[0], used_kib=int(fields[1].rstrip("*")),
                             soft_kib=int(fields[2]), hard_kib=int(fields[3])))
    if not rows:
        raise RuntimeError("quota unavailable; inspect before writing persistent records")
    return rows


def collect_small(source, destination):
    """Preserve authoritative records without copying checkpoint payloads."""
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in (".json", ".jsonl", ".log"):
            continue
        if path.stat().st_size > 32 * 1024**2:
            raise RuntimeError("unexpectedly large persistent record: " + str(path))
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix")
    p.add_argument("--site", required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", required=True, help="fixed launcher/config Git commit")
    args = p.parse_args()
    matrix = resolve_matrix(Path(args.matrix).resolve(), site=Path(args.site).resolve())
    args.scratch, args.output = args.scratch.resolve(), args.output.resolve()
    if len(matrix["cases"]) != 3:
        p.error("expected inference, strict small training, exploratory 1B training")
    for case in matrix["cases"]:
        execution = case["config"]["execution"]["effective"]
        if len(execution.get("peer_devices", [])) != 8:
            p.error("every case must explicitly place the same model on eight logical cards")
    quota = quota_blocks()
    # More than 150 GiB below the stated hard limit, and protect soft grace too.
    if any(row["hard_kib"] and row["hard_kib"] - row["used_kib"] < 150 * 1024**2
           or row["soft_kib"] and row["used_kib"] >= row["soft_kib"] for row in quota):
        raise RuntimeError("persistent quota headroom below policy; do not launch")
    if args.scratch.exists() or args.output.exists():
        raise FileExistsError("choose new scratch and output directories")
    args.scratch.mkdir(parents=True)
    args.output.mkdir(parents=True)
    if subprocess.check_output(["stat", "-f", "-c", "%T", str(args.scratch)], text=True).strip() != "tmpfs":
        raise ValueError("--scratch must reside on tmpfs; no disk checkpoint fallback")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    # 53 GB training host estimate + retained 1B inference/training checkpoint
    # generations, libraries and margin; each public run also performs preflight.
    if available < 128 * 1024**3 or shutil.disk_usage(args.scratch).free < 48 * 1024**3:
        raise RuntimeError("joint host/tmpfs reserve insufficient")
    state = dict(schema="tide-bounded-followup-v1", state="running", source=args.source,
                 started=now(), quota_before=quota, memory_available=available,
                 scratch=str(args.scratch), visibility=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                 script_sha256=digest(__file__), rows=[],
                 interpretation="shared-resource cost observation; strict numeric, discrete and exact resume remain separate")
    write_json(args.output / "matrix.json", matrix)
    write_json(args.output / "progress.json", state)
    small_passed = False
    for index, case in enumerate(matrix["cases"]):
        name, config = case["name"], case["config"]
        if index == 2 and not small_passed:
            state["rows"].append(dict(name=name, state="not-run", reason="eight-card small training strict/exact gate failed"))
            break
        folder = args.scratch / name
        job, run = folder / "supervisor", folder / "run"
        identity = capture(config["site"]["paths"])
        prepare(job, command=[sys.executable, "-m", "tidegraph", "run", str(job / "resolved.json"),
                             "--output-dir", str(run)], cwd=args.scratch,
                timeout=config["measurement"]["timeout_seconds"],
                metadata=dict(source=args.source, config_id=config["config_id"], install_identity=identity))
        write_json(job / "resolved.json", config)
        state["current"] = name
        write_json(args.output / "progress.json", state)
        code = supervise(job)
        observed = inspect(job)
        if observed["observed_live_workload_pids"] or observed.get("remaining_live_pids"):
            raise RuntimeError("workload descendants remain; do not start the next case")
        collect_small(folder, args.output / name)
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
        state["rows"].append(dict(name=name, exit_code=code, state=result.get("state", "missing-result"),
                                   validation={k: result.get("validation", {}).get(k) for k in ("numeric", "discrete")},
                                   exact_resume=result.get("checkpoint", {}).get("exact_resume")))
        if index == 1:
            small_passed = code == 0 and result.get("strict_passed") is True and result.get("checkpoint", {}).get("exact_resume") is True
        write_json(args.output / "progress.json", state)
    paths = sorted(args.output.glob("*/run/result.json"))
    save(args.output / "results.json.gz", render(load(paths), format="archive"))
    state.update(state="completed", finished=now(), quota_after=quota_blocks(), review_required=True)
    write_json(args.output / "progress.json", state)
    return int(any(row.get("exit_code", 1) for row in state["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
