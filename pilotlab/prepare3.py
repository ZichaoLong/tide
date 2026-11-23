"""Freeze third-round preparation inputs and draft a durable launch.

This module never submits a service or allocates a device.  Budget creation is
an explicit separate command immediately before the first approved submission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

from .io import atomic_json, sha256
from .scheduler3 import freeze_json, read

MI = 2**20


def ensure_budget(path: str | Path, *, now: float | None = None) -> dict:
    """Create once; resubmission or repair never restarts the 30-day clock."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read(path)
        if existing is not None:
            if (existing.get("max_device_seconds") != 7200 * 3600
                    or existing.get("deadline_unix") != existing.get("started_unix", 0) + 30 * 86400
                    or not isinstance(existing.get("segments"), dict)):
                raise ValueError("existing budget does not match the approved immutable ceiling")
            return existing
        started = time.time() if now is None else now
        value = {"schema": "pilot-r3-budget-v1", "started_unix": started,
                 "deadline_unix": started + 30 * 86400, "max_device_seconds": 7200 * 3600,
                 "device_seconds": 0, "segments": {},
                 "preparation_max_device_seconds": 48 * 3600,
                 "preparation_max_processed_tokens": 64 * MI,
                 "preparation_token_reservations": {}}
        atomic_json(path, value)
        return value


def verify_data(data: Path) -> dict:
    """Full content verification at freeze time, outside worker timing."""
    manifest = read(data / "manifest.json")
    if manifest.get("schema_version") != "tide-pilot-data-v3":
        raise ValueError("third-round data must be prepared first")
    source = Path(manifest["source_directory"]).resolve()
    if sha256(source / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("original data manifest changed")
    checked = {}
    for base, records in ((source, manifest["source_files"]), (data, manifest["files"])):
        for name, expected in records.items():
            path = (base / name).resolve()
            if not path.is_relative_to(base.resolve()):
                raise ValueError("data asset escapes declared root")
            if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"data asset identity mismatch: {path}")
            checked[str(path)] = expected
    return {"manifest_path": str(data / "manifest.json"), "manifest_sha256": sha256(data / "manifest.json"),
            "assets": checked, "verified_unix": time.time()}


def freeze(repo: str | Path, foundation: str | Path, output: str | Path, *, python: str | Path,
           data: str | Path, budget: str | Path, max_workers: int = 4,
           storage_reserve_bytes: int = 0, max_rss_bytes: int = 512 * 1024**3) -> dict:
    repo, foundation, output = Path(repo).resolve(), Path(foundation).resolve(), Path(output).resolve()
    python, data, budget = Path(python).absolute(), Path(data).resolve(), Path(budget).resolve()
    if not python.is_file() or not (repo / "pilotlab").is_dir() or not (foundation / "src/tide").is_dir():
        raise ValueError("existing project, interpreter and foundation source are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to replace an existing preparation snapshot or launch")
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be positive")
    data_verification = verify_data(data)
    mappings = [(path, output / "source/pilotlab" / path.name)
                for path in sorted((repo / "pilotlab").glob("*.py"))]
    mappings += [(path, output / "foundation/src/tide" / path.relative_to(foundation / "src/tide"))
                 for path in sorted((foundation / "src/tide").rglob("*.py"))]
    originals = {str(source): sha256(source) for source, _ in mappings}
    required = {"__init__.py", "prepare3.py", "preflight3.py", "scheduler3.py", "worker3.py", "data3.py",
                "models3.py", "catalog3.py", "models.py", "worker.py", "io.py", "scheduler.py", "selection.py"}
    if required - {path.name for path, _ in mappings if path.parent == repo / "pilotlab"}:
        raise ValueError("preparation modules are not complete")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for source, target in mappings:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        digest = sha256(target)
        if digest != originals[str(source)]:
            raise RuntimeError("source changed while preparing frozen inputs")
        frozen[str(target)] = digest
    if any(sha256(source) != originals[str(source)] for source, _ in mappings):
        raise RuntimeError("working source changed during snapshot; use a new clean revision directory")
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "numpy")}
    identity = {"schema": "pilot-r3-source-v1", "source_files": frozen,
                "data_manifest_sha256": data_verification["manifest_sha256"],
                "runtime_packages": packages, "python_version": sys.version.split()[0]}
    prepared_path, launcher = output / "prepared.json", output / "launch.sh"
    prepared = {"schema": "pilot-r3-prepared-inputs-v1", "status": "frozen_not_submitted",
                "created_unix": time.time(), "python": str(python), "source": str(output / "source"),
                "foundation": str(output / "foundation"), "data": str(data), "output": str(output / "round"),
                "budget_ledger": str(budget), "run_id": "r3-" + output.name,
                "max_workers": max_workers, "storage_reserve_bytes": storage_reserve_bytes,
                "max_rss_bytes": max_rss_bytes, "source_identity": identity,
                "original_source_files": originals, "data_verification": data_verification,
                "environment": {"module": "ascend/dev-workspace-9.2.0", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                    "PYTHONPATH": os.pathsep.join((str(output / "source"), str(output / "foundation/src"))),
                    "PYTHONDONTWRITEBYTECODE": "1"},
                "limits": {"preparation_device_seconds": 48 * 3600, "preparation_processed_tokens": 64 * MI},
                "launcher": str(launcher)}
    freeze_json(prepared_path, prepared)
    command = [str(python), "-u", "-m", "pilotlab.preflight3", "--prepared", str(prepared_path)]
    script = "\n".join(("#!/usr/bin/env bash", "set -euo pipefail", "source /usr/share/Modules/init/bash",
        "module purge", "module load ascend/dev-workspace-9.2.0", "export TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "export PYTHONDONTWRITEBYTECODE=1", "export PYTHONHASHSEED=0", "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export PYTHONPATH=" + shlex.quote(prepared["environment"]["PYTHONPATH"]),
        "cd " + shlex.quote(prepared["source"]), "exec " + shlex.join(command), ""))
    launcher.write_text(script)
    launcher.chmod(0o755)
    with launcher.open("rb") as stream:
        os.fsync(stream.fileno())
    unit = "tide-pilot-r3-" + output.name + ".service"
    log = output / "preflight.log"
    launch_argv = ["systemd-run", "--user", f"--unit={unit}", "--service-type=exec", "--slice=background.slice",
        f"--working-directory={prepared['source']}", "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=172920", "--property=TimeoutStopSec=120",
        f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}", str(launcher)]
    draft = {"schema": "pilot-r3-launch-draft-v1", "status": "not_submitted", "unit": unit,
        "prepared_path": str(prepared_path), "prepared_sha256": sha256(prepared_path),
        "launcher": str(launcher), "launcher_sha256": sha256(launcher), "log": str(log),
        "budget_creation_command": shlex.join([str(python), "-m", "pilotlab.prepare3", "init-budget", "--budget", str(budget)]),
        "launch_argv": launch_argv, "launch_command": shlex.join(launch_argv),
        "inspection_command": f"systemctl --user show {unit} -p ActiveState -p MainPID -p ControlGroup -p ExecMainStatus",
        "stop_command": f"systemctl --user stop {unit}",
        "resume_policy": "inspect first; original budget and frozen inputs retained; no automatic code repair",
        "submission_note": "create/check the immutable budget immediately before actual submission; root agent submits and verifies"}
    freeze_json(output / "launch-draft.json", draft)
    return draft


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    budget = commands.add_parser("init-budget")
    budget.add_argument("--budget", type=Path, required=True)
    snapshot = commands.add_parser("freeze")
    snapshot.add_argument("--repo", type=Path, required=True)
    snapshot.add_argument("--foundation", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--python", type=Path, required=True)
    snapshot.add_argument("--data", type=Path, required=True)
    snapshot.add_argument("--budget", type=Path, required=True)
    snapshot.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "init-budget":
        result = ensure_budget(args.budget)
        result = dict(result, deadline_utc=datetime.fromtimestamp(result["deadline_unix"], timezone.utc).isoformat())
    else:
        result = freeze(args.repo, args.foundation, args.output, python=args.python, data=args.data,
                        budget=args.budget, max_workers=args.max_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
