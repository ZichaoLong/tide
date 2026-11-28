"""Package and run immutable round-three tasks across different directory layouts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ident(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,190}", value),
        "invalid identifier",
    )
    return value


def relative(base, name):
    path = Path(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts, "expected a relative member")
    target = (base / path).resolve()
    require(target.is_relative_to(base.resolve()), "member escaped root")
    return target


def write(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


@contextlib.contextmanager
def module_binding(module, name, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def pack(output, tasks, data, evidence=()):
    """Include original bytes and paths in a standalone, content-verified tree."""
    output = Path(output).absolute()
    require(not output.exists(), "output already exists")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tide-pack-", dir=output.parent) as temporary:
        root = Path(temporary) / "frozen"
        root.mkdir(mode=0o700)
        paths, files, requests, roots = {}, {}, {}, {}

        def add(source, name, expected=None):
            src = Path(source).resolve(strict=True)
            dst = relative(root, name)
            old = paths.get(str(src))
            if old:
                require(old == name or files[old]["sha256"] == sha(src), "source mapping changed")
                require(expected is None or files[old]["sha256"] == expected, "declared input digest differs")
                return old
            require(not dst.exists(), "two unrelated inputs have the same package path: " + name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            digest = sha(dst)
            require(expected is None or digest == expected, f"input hash differs: {src}")
            files[name] = {"sha256": digest, "size": dst.stat().st_size}
            paths[str(src)] = name
            return name

        def root_map(original, member):
            key = str(Path(original).resolve())
            require(
                roots.get(key, member) == member and member not in (v for k, v in roots.items() if k != key),
                "multiple source roots for one package; freeze a separate group per source identity",
            )
            roots[key] = member

        def reference(source, expected=None):
            src = Path(source).resolve(strict=True)
            name = (
                paths.get(str(src))
                or "evidence/" + hashlib.sha256(str(src).encode()).hexdigest()[:20] + "/" + src.name
            )
            return add(src, name, expected)

        data = Path(data).resolve(strict=True)
        manifest = read(data / "manifest.json")
        require(manifest["schema_version"] == "tide-pilot-data-v3", "expected R3 data")
        base = Path(manifest["source_directory"]).resolve(strict=True)
        root_map(data, "data")
        root_map(base, "corpus")
        add(data / "manifest.json", "data/manifest.json")
        add(base / "manifest.json", "corpus/manifest.json", manifest["source_manifest_sha256"])
        for directory, name, entries in (
            (data, "data", manifest["files"]),
            (base, "corpus", manifest["source_files"]),
        ):
            for k, r in entries.items():
                source = relative(directory, k)
                require(source.stat().st_size == r["size_bytes"], "data asset size differs")
                add(source, name + "/" + k, r["sha256"])
        for file in tasks:
            task = read(file)
            key = ident(task["id"])
            require(key not in requests, "duplicate task ID")
            entry = add(file, "tasks/" + key + ".json")
            identity = task["source_identity"]
            require(
                identity["schema"] == "pilot-r3-source-v1"
                and identity["data_manifest_sha256"] == files["data/manifest.json"]["sha256"],
                "task/data identity",
            )
            for old, expected in identity["source_files"].items():
                src = Path(old).resolve(strict=True)
                parent = next((p for p in src.parents if p.name in {"pilotlab", "tide"}), None)
                require(parent is not None, "file outside declared project or foundation module: " + old)
                target = "source/pilotlab" if parent.name == "pilotlab" else "foundation/src/tide"
                root_map(parent, target)
                add(src, target + "/" + src.relative_to(parent).as_posix(), expected)
            gate = task.get("gate_file") or task.get("stage_b_gate")
            if gate:
                reference(gate)
                record = read(gate)
                for k in ("manifest", "report"):
                    reference(record[k + "_path"], record[k + "_sha256"])
            auth = task.get("confirmation_authorization", {})
            if auth.get("manifest_path"):
                reference(auth["manifest_path"], auth["manifest_sha256"])
            if task.get("resume_checkpoint"):
                reference(task["resume_checkpoint"], task["resume_checkpoint_sha256"])
            requests[key] = {"path": entry, "fingerprint": files[entry]["sha256"]}
        for path in evidence:
            reference(path)
        write(
            root / "freeze.json",
            {"schema": "tide-portable/v1", "tasks": requests, "paths": paths, "roots": roots, "files": files},
        )
        root.rename(output)
    return read(output / "freeze.json")


class Frozen:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = read(self.directory / "freeze.json")
        require(self.manifest["schema"] == "tide-portable/v1", "invalid frozen tree")
        for name, record in self.manifest["files"].items():
            path = relative(self.directory, name)
            require(
                path.stat().st_size == record["size"] and sha(path) == record["sha256"],
                "changed input: " + name,
            )
        self.paths = {
            old: relative(self.directory, new)
            for old, new in {**self.manifest["roots"], **self.manifest["paths"]}.items()
        }

    def path(self, name):
        path = Path(name)
        return self.paths.get(str(path), path)

    def request(self, key, expected):
        spec = self.manifest["tasks"][key]
        file = relative(self.directory, spec["path"])
        require(expected == spec["fingerprint"] == sha(file), "frozen task fingerprint mismatch")
        task = read(file)
        require(task["id"] == key, "task ID mismatch")
        return task, file


def run_worker(args):
    """A dedicated subprocess reads the archived worker with its original identity checks."""
    frozen = Frozen(args.frozen)
    task, taskfile = frozen.request(args.task, args.fingerprint)
    sys.path[:0] = [str(frozen.directory / "source"), str(frozen.directory / "foundation/src")]
    from pilotlab import data3, worker3
    from pilotlab import gate3, scheduler3
    from pilotlab.io import sha256

    for mod in (worker3, data3, gate3, scheduler3):
        require(
            Path(mod.__file__).resolve().is_relative_to(frozen.directory / "source/pilotlab"),
            "worker or admission code was imported outside the frozen tree",
        )
    if task.get("device"):
        require(task["device"].split(":")[0] == args.device.split(":")[0], "frozen backend differs")
    require(bool(args.runtime_plan) == bool(args.runtime_plan_sha), "runtime plan and SHA-256 must be supplied together")
    require(bool(args.resume) == bool(args.resume_sha), "resume checkpoint and SHA-256 must be supplied together")
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = history = runtime = migration = None
    if args.runtime_plan:
        # The maintained adapter is supplied alongside the executable; all imports
        # under pilotlab/ used by training still resolve to the frozen worker tree.
        adjacent = Path(__file__).resolve().with_name("runtime_migration3.py")
        module_path = adjacent if adjacent.is_file() else Path(__file__).resolve().parents[1] / "pilotlab/runtime_migration3.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("tide_runtime_adapter", module_path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        plan, history = migration.read_plan(args.runtime_plan, args.runtime_plan_sha, task, args.fingerprint)
        name = os.environ.get("MAFLOW_RUN")
        manifest = read(Path(os.environ["MAFLOW_OBS"]) / "inputs/runtime-allocation.json")
        allocation = migration.valid_allocation(manifest.get("manifest", manifest), name)
        observed = observed_runtime()
        require(observed == plan["target_runtime"], f"observed runtime differs from the approved target: {observed}")
        runtime = {"schema": "tide-runtime-attempt/v1", "declared": plan["source_runtime"],
                   "observed": observed, "run": name, "allocation": allocation,
                   "prior_result_hashes": [row["sha256"] for row in plan["prior_results"]],
                   "runtime_plan_sha256": args.runtime_plan_sha, "adapter_sha256": sha(module_path)}
        require(bool(args.resume) == bool(history), "explicit resume requires an ordered same-task history")
        if history:
            require(args.resume_sha == history[-1]["checkpoint"]["sha256"], "resume is not the last recorded endpoint")
    require(not args.check_first_save or plan, "first-update migration receipt needs an approved runtime plan")
    if args.resume:
        require(args.resume_sha and sha(args.resume) == args.resume_sha, "resume artifact SHA256 mismatch")
        # Explicit runtime overrides must select the exact published generation.
        require(args.resume.is_file(), "resume requires a checkpoint payload")
    if args.stop_after_step is not None:
        require(args.stop_after_step > 0, "stop boundary must be a completed update")

    first_update = restored = None

    def save_and_interrupt(directory, state):
        nonlocal first_update
        record = original_save(directory, state)
        if args.check_first_save and first_update is None:
            import torch
            origin = history[-1]["checkpoint"]["steps"] if history else (
                task.get("evaluation_start_tokens", 0) // (task["batch_size"] * task["sequence_length"]))
            if state["steps"] == origin + 1:
                require(restored is not None or not (history or task.get("resume_checkpoint")),
                        "first update has no verified input generation")
                with (output / "steps.jsonl").open() as rows:
                    step = json.loads(rows.readline())
                require(step["step"] == state["steps"] and all(math.isfinite(step[k]) for k in ("loss", "grad_norm")),
                        "first restored update has invalid training observations")
                store = data3.DocumentStore3(frozen.directory / "data", allow_confirmation=bool(task.get("allow_confirmation")))
                check = worker3.verify_saved_checkpoint(record, task, store, torch.device(args.device))
                require(check["status"] == "exact" and check["state_sha256"] == record["state_sha256"],
                        "first update does not reload into fresh owners")
                first_update = {"schema": "tide-runtime-first-update/v1", "status": "passed", "task": task["id"],
                                "fingerprint": args.fingerprint, "runtime_plan_sha256": args.runtime_plan_sha,
                                "from_steps": origin, "steps": state["steps"], "tokens": state["tokens"],
                                "restored": restored,
                                "checkpoint_sha256": record["sha256"], "state_sha256": record["state_sha256"],
                                "checkpoint_load": check, "loss": step["loss"], "grad_norm": step["grad_norm"],
                                "runtime": runtime["observed"], "allocation": runtime["allocation"]}
                write(output / "runtime-first-update.json", first_update)
        if args.stop_after_step is not None and state["steps"] >= args.stop_after_step and not (output / "intentional-stop.json").exists():
            write(
                output / "intentional-stop.json",
                {
                    "pid": os.getpid(),
                    "steps": state["steps"],
                    "tokens": state["tokens"],
                    "sha256": record["sha256"],
                    "at": time.time(),
                },
            )
            os.kill(os.getpid(), signal.SIGTERM)
        return record

    # Old R3 snapshots use absolute input paths, including paths inside the gate
    # and data JSON. Map only the named frozen files/roots inside this subprocess.
    # The raw task, source identity, gate and training/checkpoint code stay intact.
    original_save = worker3.save_checkpoint
    with contextlib.ExitStack() as stack:
        for module in (worker3, data3):
            stack.enter_context(module_binding(module, "Path", frozen.path))
            stack.enter_context(module_binding(module, "sha256", lambda path: sha256(frozen.path(path))))
        if plan:
            original_identity_check = worker3.validate_source_identity

            def check_execution_identity(original, directory):
                require(original == task, "runtime migration received a different frozen task")
                identity = original["source_identity"]
                expected = plan["target_runtime"]
                declared = {**identity, "python_version": expected["python"],
                            "runtime_packages": {k: expected[k] for k in identity["runtime_packages"]}}
                original_identity_check({**original, "source_identity": declared}, directory)
                return identity

            stack.enter_context(module_binding(worker3, "validate_source_identity", check_execution_identity))
            owners = worker3.restore_owners

            def restore_generation(payload, model, optimizer, store, actual_task, device):
                nonlocal restored
                if history:
                    expected = history[-1]["checkpoint"]
                    require(payload["task_id"] == task["id"] and all(payload[k] == expected[k]
                            for k in ("steps", "tokens", "state_sha256")), "wrong input state or progress")
                value = owners(payload, model, optimizer, store, actual_task, device)
                restored = {k: payload[k] for k in ("task_id", "steps", "tokens", "state_sha256")}
                restored["checkpoint_sha256"] = args.resume_sha or task["resume_checkpoint_sha256"]
                return value

            stack.enter_context(module_binding(worker3, "restore_owners", restore_generation))
            if history and plan["evaluation"]["initial"] is not None:
                evaluate = worker3.evaluate
                inherited = False

                def evaluate_with_origin(model, store, actual_task, device, deadline, **kwargs):
                    nonlocal inherited
                    if not inherited and not kwargs:
                        inherited = True
                        return migration.inherited_evaluation(plan, actual_task)
                    return evaluate(model, store, actual_task, device, deadline, **kwargs)

                stack.enter_context(module_binding(worker3, "evaluate", evaluate_with_origin))
        if args.stop_after_step is not None or args.check_first_save:
            stack.enter_context(module_binding(worker3, "save_checkpoint", save_and_interrupt))
        signal.signal(signal.SIGTERM, worker3.stop_requested)
        signal.signal(signal.SIGINT, worker3.stop_requested)
        options = argparse.Namespace(
            output=output,
            data=frozen.directory / "data",
            task=taskfile,
            device=args.device,
            deadline_unix=args.deadline_unix,
            resume_checkpoint=args.resume,
        )
        result = worker3.execute(options)
        if plan:
            result["execution_runtime"] = runtime
            evaluation = result["evaluation"]
            prior = plan["evaluation"]["prior_observed_tokens"]
            evaluation.update(prior_attempt_tokens=prior, cumulative_task_observed_tokens=prior + evaluation["tokens"])
            write(output / "result.json", result)
        gate = worker3.validate_gate(task, args.fingerprint)
        declared = result.get("status")
        rc = 0 if declared == "completed" else 2 if declared == "budget_exhausted" else 1
        outcome, reason = scheduler3.evaluate_outcome(task, result, rc)
        checkpoint = result.get("checkpoint") or {}
        reasons = gate3.eligible(task, result) if outcome == "completed" else []
        status = "failed"
        if (
            outcome == "completed"
            and not reasons
            and result["recovery"]["checkpoint_load"]["status"] == "exact"
        ):
            status = "completed"
        if (
            outcome in {"stopped_budget", "interrupted"}
            and result.get("stop_reason") == "signal"
            and checkpoint
        ):
            bad = set(result["health"]["blocking_reasons"]) - {
                "declared_training_exposure_not_reached",
                "evaluation_incomplete",
                "checkpoint_load_not_verified",
                "causality_not_verified",
            }
            if not bad and 0 < checkpoint["tokens"] < task["target_tokens"]:
                payload = worker3._read_checkpoint(Path(checkpoint["path"]), checkpoint["sha256"])
                worker3.validate_checkpoint(payload, task, args.device)
                status = "interrupted"
        if args.resume and status in {"completed", "interrupted"}:
            resume = result.get("resume", {})
            require(
                resume.get("sha256") == args.resume_sha
                and not resume.get("fallback_used")
                and resume.get("checkpoint_load", {}).get("status") == "exact"
                and (result["training"]["steps"] > resume["from_steps"] or
                     resume["from_tokens"] == result["training"]["tokens"] == task["target_tokens"]),
                "exact checkpoint continuation not established",
            )
        require(result.get("task_sha256") == args.fingerprint, "result belongs to another task")
        if checkpoint:
            require(sha(checkpoint["path"]) == checkpoint["sha256"], "produced checkpoint differs")
        if plan and status in {"completed", "interrupted"}:
            migration.verify_result(task, result, plan, history, args.runtime_plan_sha)
            require(not args.check_first_save or first_update, "first-update canary ended without its reload proof")
        write(
            output / "project-verdict.json",
            {
                "status": status,
                "worker_status": declared,
                "reason": reason,
                "gate": gate,
                "eligibility_reasons": reasons,
                "fingerprint": args.fingerprint,
                "output": str(output),
                "device": args.device,
                "runtime": observed_runtime(),
                **({"execution_runtime": runtime} if runtime else {}),
                "worker_pid": os.getpid(),
                "ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                "task": task["id"],
                "steps": result["training"]["steps"],
                "tokens": result["training"]["tokens"],
                "resume": result.get("resume"),
                "checkpoint": checkpoint,
            },
        )
    return int(status == "failed")


def observed_runtime():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "torch-npu", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("pack")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--task", type=Path, action="append", required=True)
    freeze.add_argument("--data", type=Path, required=True)
    freeze.add_argument("--evidence", type=Path, action="append", default=[])
    run = sub.add_parser("worker")
    for flag, kind in (
        ("frozen", Path),
        ("task", str),
        ("fingerprint", str),
        ("output", Path),
        ("device", str),
        ("deadline-unix", float),
    ):
        run.add_argument("--" + flag, type=kind, required=True)
    run.add_argument("--resume", type=Path)
    run.add_argument("--resume-sha")
    run.add_argument("--stop-after-step", type=int)
    run.add_argument("--runtime-plan", type=Path)
    run.add_argument("--runtime-plan-sha")
    run.add_argument("--check-first-save", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "pack":
        manifest = pack(args.output, args.task, args.data, args.evidence)
        print(json.dumps({"tasks": manifest["tasks"], "sha256": sha(args.output / "freeze.json")}))
        return 0
    return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
