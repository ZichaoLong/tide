#!/usr/bin/env python3
"""Run the deterministic Plan corpus and write a portable local run record."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_INVALID_PLAN_IDENTITY_SCHEMA = "tide.invalid-plan-structural.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _structural_json_value(value: Any) -> Any:
    """Return stable JSON data for a possibly invalid Plan declaration.

    A Plan's semantic canonicalizer intentionally validates first, so it
    cannot identify validator mutants. This representation covers every
    public constructor field while excluding derived, private indexes. It is
    an artifact identity, not an alternative semantic Plan canonicalizer.
    """

    if value is None or type(value) in {bool, int, float, str}:
        return {
            "kind": "scalar",
            "type": "null" if value is None else type(value).__name__,
            "value": value.hex() if type(value) is float else value,
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _structural_json_value(getattr(value, field.name))]
                for field in dataclasses.fields(value)
                if field.init and not field.name.startswith("_")
            ],
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("structural Plan mappings must have string keys")
        return {
            "kind": "mapping",
            "items": [
                [key, _structural_json_value(value[key])] for key in sorted(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_structural_json_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_structural_json_value(item) for item in value],
        }
    raise TypeError(
        "unsupported value in structural Plan identity: "
        f"{type(value).__name__}"
    )


def _invalid_plan_identity(plan: Any) -> dict[str, str]:
    record = {
        "identity_schema": _INVALID_PLAN_IDENTITY_SCHEMA,
        "plan": _structural_json_value(plan),
    }
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "schema": _INVALID_PLAN_IDENTITY_SCHEMA,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _resolved_module_file(module: Any, module_name: str) -> Path:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"{module_name} does not have a concrete source file")
    return Path(source).resolve()


def _require_module_file(module: Any, module_name: str, expected: Path) -> None:
    actual = _resolved_module_file(module, module_name)
    if actual != expected.resolve():
        raise RuntimeError(
            f"{module_name} must be imported from {expected.resolve()}, got {actual}"
        )


def _prepare_repository_imports() -> None:
    """Make this checkout authoritative and reject foreign preloaded modules."""

    source_root = (_REPOSITORY_ROOT / "src").resolve()
    package_root = source_root / "tide"
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not (
            module_name == "tide" or module_name.startswith("tide.")
        ):
            continue
        actual = _resolved_module_file(module, module_name)
        try:
            actual.relative_to(package_root)
        except ValueError as exc:
            raise RuntimeError(
                f"preloaded {module_name} is outside this repository: {actual}"
            ) from exc

    retained = []
    for entry in sys.path:
        try:
            same_path = Path(entry or os.curdir).resolve() == source_root
        except (OSError, RuntimeError):
            same_path = False
        if not same_path:
            retained.append(entry)
    sys.path[:] = [str(source_root), *retained]

    package = importlib.import_module("tide")
    generators = importlib.import_module("tide.generators")
    plan = importlib.import_module("tide.plan")
    _require_module_file(package, "tide", package_root / "__init__.py")
    _require_module_file(
        generators, "tide.generators", package_root / "generators.py"
    )
    _require_module_file(plan, "tide.plan", package_root / "plan.py")


def _discover_repository_suite() -> unittest.TestSuite:
    """Discover the required test only from this checkout."""

    expected = (_REPOSITORY_ROOT / "tests" / "test_plan_corpus.py").resolve()
    preloaded = sys.modules.get("test_plan_corpus")
    if preloaded is not None:
        _require_module_file(preloaded, "test_plan_corpus", expected)
    suite = unittest.defaultTestLoader.discover(
        str(_REPOSITORY_ROOT / "tests"), pattern="test_plan_corpus.py"
    )
    imported = sys.modules.get("test_plan_corpus")
    if imported is None:
        raise RuntimeError("test discovery did not import test_plan_corpus")
    _require_module_file(imported, "test_plan_corpus", expected)
    return suite


def _verify_cpu_runtime(torch: Any) -> dict[str, str]:
    if not hasattr(torch, "get_default_device") or not hasattr(
        torch, "get_default_dtype"
    ):
        raise RuntimeError("Torch does not expose default device/dtype inspection")
    default_device = torch.get_default_device()
    probe = torch.empty(())
    default_dtype = torch.get_default_dtype()
    default_type = getattr(
        default_device, "type", str(default_device).split(":", 1)[0]
    )
    if (
        default_type != "cpu"
        or probe.device.type != "cpu"
        or default_dtype != torch.float32
        or probe.dtype != torch.float32
    ):
        raise RuntimeError(
            "development corpus requires actual Torch defaults CPU/float32"
        )
    return {
        "default_device": str(default_device),
        "default_device_probe": str(probe.device),
        "default_dtype": str(default_dtype),
        "default_dtype_probe": str(probe.dtype),
    }


def _establish_cpu_runtime(torch: Any) -> dict[str, str]:
    if not hasattr(torch, "set_default_device") or not hasattr(
        torch, "set_default_dtype"
    ):
        raise RuntimeError("Torch does not expose default device/dtype configuration")
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    return _verify_cpu_runtime(torch)


def _required_suite_outcome(result: Any) -> tuple[bool, int, int, int]:
    expected_failures = len(getattr(result, "expectedFailures", ()))
    unexpected_successes = len(getattr(result, "unexpectedSuccesses", ()))
    failed = (
        len(result.failures)
        + len(result.errors)
        + expected_failures
        + unexpected_successes
    )
    successful = bool(
        result.wasSuccessful()
        and result.testsRun > 0
        and not result.skipped
        and not expected_failures
        and not unexpected_successes
    )
    return successful, failed, expected_failures, unexpected_successes


def _atomic_json(path: Path, value: Any) -> None:
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_owned = True
    temporary_identity: tuple[int, int] | None = None
    try:
        initial = os.fstat(handle)
        temporary_identity = (initial.st_dev, initial.st_ino)
        stream = os.fdopen(handle, "wb")
        handle = -1
        try:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            try:
                stream.close()
            except BaseException:
                # Data and inode durability were established by the explicit
                # flush/fsync; close diagnostics must not replace the writer's
                # primary result.
                pass
        named = os.lstat(temporary)
        if (named.st_dev, named.st_ino) != temporary_identity:
            raise OSError("atomic JSON staging path changed before publication")
        os.replace(temporary, path)
        temporary_owned = False
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_handle = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_handle)
        finally:
            try:
                os.close(directory_handle)
            except BaseException:
                pass
    finally:
        if handle >= 0:
            try:
                os.close(handle)
            except BaseException:
                pass
        if temporary_owned and temporary_identity is not None:
            try:
                named = os.lstat(temporary)
                if (named.st_dev, named.st_ino) == temporary_identity:
                    os.unlink(temporary)
            except BaseException:
                # Never replace a write/replace/fsync failure with cleanup
                # noise or delete a concurrently recreated staging name.
                pass


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        check=True,
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ("git", *args),
        check=True,
        cwd=_REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _status_records(status: bytes) -> list[tuple[bytes, bytes]]:
    """Parse the path-bearing records from porcelain-v1 ``-z`` output."""

    fields = status.split(b"\0")
    records: list[tuple[bytes, bytes]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        if not field:
            index += 1
            continue
        if len(field) < 4 or field[2:3] != b" ":
            raise RuntimeError("git status returned malformed porcelain-v1 output")
        status_code = field[:2]
        records.append((status_code, field[3:]))
        index += 1
        if b"R" in status_code or b"C" in status_code:
            if index >= len(fields) or not fields[index]:
                raise RuntimeError("git status omitted a rename/copy source path")
            index += 1
    return records


def _inside_run_directory(path: bytes, excluded: bytes | None) -> bool:
    if excluded is None:
        return False
    normalized = path.rstrip(b"/")
    return normalized == excluded or normalized.startswith(excluded + b"/")


def _digest_part(digest: Any, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _untracked_identity(relative_path: bytes) -> bytes:
    absolute_path = os.path.join(os.fsencode(_REPOSITORY_ROOT), relative_path)
    before = os.lstat(absolute_path)
    digest = hashlib.sha256()
    _digest_part(digest, b"mode", str(before.st_mode).encode("ascii"))
    if stat.S_ISREG(before.st_mode):
        with open(absolute_path, "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    elif stat.S_ISLNK(before.st_mode):
        target = os.readlink(absolute_path)
        digest.update(os.fsencode(target) if isinstance(target, str) else target)
    else:
        _digest_part(digest, b"special-size", str(before.st_size).encode("ascii"))
    after = os.lstat(absolute_path)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError("an untracked source changed while its identity was sampled")
    return digest.digest()


def _run_dir_exclusion(run_dir: Path) -> bytes | None:
    try:
        relative = run_dir.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        return None
    if not relative.parts:
        raise ValueError("the run directory cannot be the repository root")
    return os.fsencode(str(relative))


def _source_snapshot(*, excluded_run_dir: bytes | None) -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    raw_status = _git_bytes(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    records = [
        (status_code, path)
        for status_code, path in _status_records(raw_status)
        if not _inside_run_directory(path, excluded_run_dir)
    ]
    tracked_diff = _git_bytes(
        "diff", "--binary", "--full-index", "--no-ext-diff", "HEAD", "--"
    )
    digest = hashlib.sha256()
    _digest_part(digest, b"schema", b"tide.worktree-fingerprint.v1")
    _digest_part(digest, b"commit", commit.encode("ascii"))
    _digest_part(digest, b"tracked-diff", tracked_diff)
    for status_code, path in records:
        _digest_part(digest, b"status", status_code)
        _digest_part(digest, b"path", path)
        if status_code == b"??":
            _digest_part(digest, b"untracked", _untracked_identity(path))
    return {
        "fingerprint_schema": "tide.worktree-fingerprint.v1",
        "commit": commit,
        "dirty": bool(records),
        "worktree_fingerprint": digest.hexdigest(),
        "observation_policy": {
            "tracked": "git diff --binary --full-index HEAD",
            "untracked": "git status --untracked-files=all plus type/mode/content",
            "ignored": "excluded",
            "run_directory": (
                "excluded:" + os.fsdecode(excluded_run_dir)
                if excluded_run_dir is not None
                else "outside-repository"
            ),
            "sampling": "before-and-after endpoints",
        },
        "status_counts": {
            "entries": len(records),
            "tracked": sum(status_code != b"??" for status_code, _ in records),
            "untracked": sum(status_code == b"??" for status_code, _ in records),
        },
    }


def _record_final_source(
    manifest: dict[str, Any], after: dict[str, Any]
) -> bool:
    source = manifest["source"]
    before = source["before"]
    changed = (
        before["commit"] != after["commit"]
        or before["worktree_fingerprint"] != after["worktree_fingerprint"]
    )
    source["after"] = after
    source["changed_during_run"] = changed
    source["exact_commit"] = bool(
        not before["dirty"]
        and not after["dirty"]
        and before["commit"] == after["commit"]
        and not changed
    )
    return changed


def _write_terminal_record(
    run_dir: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    _atomic_json(run_dir / "summary.json", summary)
    manifest["status"] = summary["status"]
    manifest["ended_at"] = summary["ended_at"]
    _atomic_json(run_dir / "run.json", manifest)


def _create_empty_file(path: Path) -> None:
    with path.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _append_json_line(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _note_secondary_failure(error: BaseException, message: str) -> None:
    """Attach diagnostic context without ever replacing ``error``.

    Exception subclasses may override attribute access, ``add_note()``, or
    ``__setattr__``.  Terminal-record recovery is already a secondary failure
    path, so every attachment operation is deliberately best-effort.
    """

    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(message)
            return
        # Python 3.9 compatibility for callers that inspect the exception.
        notes = list(getattr(error, "tide_secondary_failures", ()))
        notes.append(message)
        setattr(error, "tide_secondary_failures", tuple(notes))
    except BaseException:
        pass


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _corpus_record() -> tuple[dict[str, Any], str]:
    # Import project code only after ``main`` has sampled the initial source
    # identity.  A CLI run must not execute one cached revision while claiming
    # a later worktree snapshot.
    from tide.generators import (
        DEFAULT_PLAN_CORPUS_SEED,
        generate_invalid_plan_corpus,
        generate_plan_corpus,
    )
    from tide.plan import bind_dtypes

    legal = generate_plan_corpus()
    invalid = generate_invalid_plan_corpus()
    features = Counter(feature for case in legal for feature in case.features)
    record = {
        "schema_version": 1,
        "kind": "tide.development-plan-corpus.v1",
        "qualification": False,
        "seed": DEFAULT_PLAN_CORPUS_SEED,
        "legal_count": len(legal),
        "invalid_count": len(invalid),
        "vjp_plan_count": sum(case.vjp for case in legal),
        "dtypes": ["float32", "float64"],
        "executors": ["token-major-eager", "region-major-eager-reference"],
        "features": dict(sorted(features.items())),
        "legal_cases": [
            {
                "case_id": case.case_id,
                "motif": case.motif,
                "logical_plan_hash": case.plan.canonical_hash(),
                "typed_plan_hashes": {
                    dtype: bind_dtypes(
                        case.plan,
                        hidden=dtype,
                        parameter=dtype,
                        state=dtype,
                        readout=dtype,
                    ).typed_hash()
                    for dtype in ("float32", "float64")
                },
                "parameter_seed": case.parameter_seed,
                "input_seed": case.input_seed,
                "vjp": case.vjp,
                "features": sorted(case.features),
            }
            for case in legal
        ],
        "invalid_cases": [
            {
                "case_id": case.case_id,
                "mutation_kind": case.mutation_kind,
                "base_plan_hash": case.base_plan_hash,
                "mutated_plan_identity": _invalid_plan_identity(case.plan),
                "expected_codes": list(case.expected_codes),
            }
            for case in invalid
        ],
        "qualification_gaps": {
            "legal_plan_minimum": 256,
            "vjp_minimum": 64,
            "optimizer_step_minimum": 16,
            "invalid_mutant_minimum": 96,
            "packed_executor_present": False,
        },
    }
    content = _json_bytes(record)
    return record, hashlib.sha256(content).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run tests.test_plan_corpus and write run.json, metrics.jsonl, "
            "stdout.log, summary.json, and artifacts/corpus.json."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New run directory; an existing path is refused.",
    )
    parser.add_argument("--verbosity", type=int, choices=(1, 2), default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(parsed_argv)
    run_dir = args.run_dir.resolve()
    excluded_run_dir = _run_dir_exclusion(run_dir)
    source_before = _source_snapshot(excluded_run_dir=excluded_run_dir)
    _prepare_repository_imports()
    corpus, corpus_hash = _corpus_record()
    # Torch and the project test graph are intentionally imported only after
    # the initial repository snapshot.  The final snapshot below detects a
    # source edit that races this import or the workload.
    import torch

    cpu_runtime = _establish_cpu_runtime(torch)

    started_at = _utc_now()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-settlegraph-development-corpus-"
        + corpus_hash[:8]
        + "-"
        + uuid.uuid4().hex[:8]
    )
    resolved_argv = [
        sys.executable,
        "scripts/run_development_corpus.py",
        *parsed_argv,
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "project": "tide-settlegraph",
        "name": "settlegraph-development-plan-corpus",
        "parent_run_id": None,
        "status": "running",
        "created_at": started_at,
        "started_at": started_at,
        "ended_at": None,
        "source": {
            "repository": "fractal-latcarf",
            "commit": source_before["commit"],
            "dirty": source_before["dirty"],
            "worktree_fingerprint": source_before["worktree_fingerprint"],
            "before": source_before,
            "after": None,
            "changed_during_run": None,
            "exact_commit": False,
        },
        "command": {
            "argv": resolved_argv,
            "working_directory": ".",
        },
        "inputs": {
            "corpus": {
                "kind": corpus["kind"],
                "seed": corpus["seed"],
                "sha256": corpus_hash,
            }
        },
        "runtime": {
            "host_arch": platform.machine(),
            "accelerator_model": None,
            "resolved_device": "cpu",
            "resolution_reason": "explicit:cpu-development-corpus",
            **cpu_runtime,
            "dtype": "float32,float64",
            "seed": corpus["seed"],
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "backend_autoload": os.environ.get(
                "TORCH_DEVICE_BACKEND_AUTOLOAD", "<unset>"
            ),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "torch_num_threads": torch.get_num_threads(),
            "implementation_variants": [
                "token-major-eager",
                "region-major-eager-reference",
            ],
            "fallback_visibility": "not profiled in this CPU development run",
        },
        "experiment": {
            "hypothesis": (
                "token-major and independent region-major eager references "
                "agree on the deterministic expanded development corpus"
            ),
            "global_step_semantics": "one completed unittest corpus suite",
            "primary_metric": "validation/tests_failed",
            "stop_condition": "one bounded suite execution",
            "config": {
                "legal_plan_count": corpus["legal_count"],
                "invalid_plan_count": corpus["invalid_count"],
                "vjp_plan_count": corpus["vjp_plan_count"],
                "dtypes": corpus["dtypes"],
                "qualification": False,
            },
        },
        "tracking": {
            "mode": "off",
            "backend": None,
            "project": "tide-settlegraph",
            "run_name": "settlegraph-development-plan-corpus",
            "destination_kind": None,
            "local_data_root": None,
            "storage_mode": None,
            "status": "disabled",
        },
        "artifacts": {
            "metrics": "metrics.jsonl",
            "stdout": "stdout.log",
            "summary": "summary.json",
            "corpus": "artifacts/corpus.json",
        },
    }
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=False)
    stage = "setup"
    source_after: dict[str, Any] | None = None
    source_changed: bool | None = None
    try:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir()
        _atomic_json(artifacts_dir / "corpus.json", corpus)
        _create_empty_file(run_dir / "stdout.log")
        _create_empty_file(run_dir / "metrics.jsonl")
        _atomic_json(run_dir / "run.json", manifest)

        stage = "workload"
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            suite = _discover_repository_suite()
            result = unittest.TextTestRunner(
                stream=_Tee(sys.stdout, log),
                verbosity=args.verbosity,
            ).run(suite)
            _verify_cpu_runtime(torch)
        stage = "finalize"
        _fsync_file(run_dir / "stdout.log")
        elapsed = time.monotonic() - started
        source_after = _source_snapshot(excluded_run_dir=excluded_run_dir)
        source_changed = _record_final_source(manifest, source_after)
        (
            tests_successful,
            failed,
            expected_failures,
            unexpected_successes,
        ) = _required_suite_outcome(result)
        # This fixed development corpus has no optional or expected-failure
        # cases. Either outcome would silently reduce promised coverage, so it
        # is terminal even though unittest's wasSuccessful() permits both.
        status = (
            "completed" if tests_successful and not source_changed else "failed"
        )
        ended_at = _utc_now()
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "sequence": 0,
            "timestamp": ended_at,
            "step": 1,
            "elapsed_seconds": elapsed,
            "metrics": {
                "validation/tests_run": result.testsRun,
                "validation/tests_failed": failed,
                "validation/tests_skipped": len(result.skipped),
                "validation/tests_expected_failures": expected_failures,
                "validation/tests_unexpected_successes": unexpected_successes,
                "coverage/legal_plans": corpus["legal_count"],
                "coverage/invalid_plans": corpus["invalid_count"],
                "coverage/vjp_plans": corpus["vjp_plan_count"],
                "coverage/dtypes": len(corpus["dtypes"]),
            },
            "context": {
                "executor_pair": (
                    "token-major-eager/region-major-eager-reference"
                ),
                "qualification": False,
                "source_changed_during_run": source_changed,
            },
        }
        _append_json_line(run_dir / "metrics.jsonl", metrics)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "ended_at": ended_at,
            "exit_code": 0 if status == "completed" else 1,
            "primary_metrics": metrics["metrics"],
            "corpus_sha256": corpus_hash,
            "conclusion": (
                "expanded development corpus passed"
                if status == "completed"
                else (
                    "source changed during development corpus run"
                    if source_changed
                    else "expanded development corpus failed"
                )
            ),
            "qualification": False,
            "source_changed_during_run": source_changed,
        }
        _write_terminal_record(run_dir, manifest, summary)
    except BaseException as exc:
        ended_at = _utc_now()
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        exit_code = 130 if status == "interrupted" else 1
        if source_after is None:
            try:
                source_after = _source_snapshot(
                    excluded_run_dir=excluded_run_dir
                )
                source_changed = _record_final_source(manifest, source_after)
                if stage == "finalize":
                    # The required first post-workload sample failed.  A retry
                    # is useful diagnostics but cannot restore exact-source
                    # evidence for the interval that just ended.
                    manifest["source"]["exact_commit"] = False
            except BaseException as snapshot_error:
                manifest["source"]["exact_commit"] = False
                manifest["source"]["changed_during_run"] = None
                _note_secondary_failure(
                    exc,
                    "final source snapshot also failed with "
                    f"{type(snapshot_error).__name__}",
                )
        failure_summary = {
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "ended_at": ended_at,
            "exit_code": exit_code,
            "primary_metrics": {},
            "corpus_sha256": corpus_hash,
            "conclusion": (
                "development corpus interrupted"
                if status == "interrupted"
                else f"development corpus failed during {stage}"
            ),
            "qualification": False,
            "source_changed_during_run": source_changed,
            "error": {
                "stage": stage,
                "type": type(exc).__name__,
            },
        }
        try:
            _write_terminal_record(run_dir, manifest, failure_summary)
        except BaseException as record_error:
            _note_secondary_failure(
                exc,
                "terminal run-record update also failed with "
                f"{type(record_error).__name__}",
            )
        raise

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "run_id": run_id,
                "status": status,
                "corpus_sha256": corpus_hash,
                "tests_run": result.testsRun,
                "tests_failed": failed,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
