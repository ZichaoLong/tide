from __future__ import annotations

import dataclasses
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_development_corpus as runner


def _snapshot(*, fingerprint: str = "a" * 64, dirty: bool = False):
    return {
        "fingerprint_schema": "tide.worktree-fingerprint.v1",
        "commit": "1" * 40,
        "dirty": dirty,
        "worktree_fingerprint": fingerprint,
        "status_counts": {
            "entries": int(dirty),
            "tracked": int(dirty),
            "untracked": 0,
        },
    }


def _corpus_record():
    return (
        {
            "schema_version": 1,
            "kind": "tide.development-plan-corpus.v1",
            "qualification": False,
            "seed": 1729,
            "legal_count": 3,
            "invalid_count": 1,
            "vjp_plan_count": 0,
            "dtypes": ["float32", "float64"],
        },
        "b" * 64,
    )


def _successful_result():
    return SimpleNamespace(
        failures=[],
        errors=[],
        skipped=[],
        expectedFailures=[],
        unexpectedSuccesses=[],
        testsRun=4,
        wasSuccessful=lambda: True,
    )


class DevelopmentCorpusRunnerTests(unittest.TestCase):
    def test_invalid_plan_identity_binds_mutated_plan_content(self) -> None:
        runner._prepare_repository_imports()
        from tide.generators import generate_invalid_plan_corpus

        case = generate_invalid_plan_corpus()[0]
        replacement = dataclasses.replace(
            case.plan, terminal_node_ids=(case.plan.nodes[1].node_id,)
        )
        original_identity = runner._invalid_plan_identity(case.plan)
        replacement_identity = runner._invalid_plan_identity(replacement)
        self.assertEqual(
            original_identity["schema"], "tide.invalid-plan-structural.v1"
        )
        self.assertNotEqual(
            original_identity["sha256"], replacement_identity["sha256"]
        )
        corpus_record, _ = runner._corpus_record()
        self.assertEqual(
            corpus_record["invalid_cases"][0]["mutated_plan_identity"]["schema"],
            "tide.invalid-plan-structural.v1",
        )

    def test_structural_identity_keeps_container_type_domains_disjoint(self) -> None:
        @dataclasses.dataclass
        class Example:
            value: object

        values = (
            Example(1),
            {"dataclass": "Example", "fields": {"value": 1}},
            [1],
            (1,),
            1,
            1.0,
            True,
        )
        encoded = [runner._structural_json_value(value) for value in values]
        serialized = [runner._json_bytes(value) for value in encoded]
        self.assertEqual(len(set(serialized)), len(serialized))

    def test_foreign_preloaded_tide_module_is_rejected(self) -> None:
        foreign = types.ModuleType("tide.generators")
        foreign.__file__ = "/tmp/foreign/tide/generators.py"
        with mock.patch.dict(sys.modules, {"tide.generators": foreign}):
            with self.assertRaisesRegex(RuntimeError, "outside this repository"):
                runner._prepare_repository_imports()

    def test_repository_source_precedes_foreign_pythonpath_entry(self) -> None:
        source_root = str((runner._REPOSITORY_ROOT / "src").resolve())
        with mock.patch.object(
            sys, "path", ["/tmp/foreign", *sys.path]
        ):
            runner._prepare_repository_imports()
            self.assertEqual(sys.path[0], source_root)

    def test_foreign_preloaded_corpus_test_is_rejected(self) -> None:
        foreign = types.ModuleType("test_plan_corpus")
        foreign.__file__ = "/tmp/foreign/test_plan_corpus.py"
        with mock.patch.dict(sys.modules, {"test_plan_corpus": foreign}):
            with self.assertRaisesRegex(RuntimeError, "must be imported from"):
                runner._discover_repository_suite()

    def test_cpu_runtime_is_established_and_verified(self) -> None:
        class Device:
            def __init__(self, kind):
                self.type = kind

            def __str__(self):
                return self.type

        class FakeTorch:
            float32 = "float32"

            def __init__(self):
                self.default = "cuda"
                self.default_dtype = "float64"

            def set_default_device(self, value):
                self.default = value

            def get_default_device(self):
                return Device(self.default)

            def set_default_dtype(self, value):
                self.default_dtype = value

            def get_default_dtype(self):
                return self.default_dtype

            def empty(self, shape):
                return types.SimpleNamespace(
                    device=Device(self.default), dtype=self.default_dtype
                )

        fake = FakeTorch()
        record = runner._establish_cpu_runtime(fake)
        self.assertEqual(fake.default, "cpu")
        self.assertEqual(fake.default_dtype, fake.float32)
        self.assertEqual(
            record,
            {
                "default_device": "cpu",
                "default_device_probe": "cpu",
                "default_dtype": "float32",
                "default_dtype_probe": "float32",
            },
        )
        fake.default = "npu"
        with self.assertRaisesRegex(RuntimeError, "defaults CPU/float32"):
            runner._verify_cpu_runtime(fake)
        fake.default = "cpu"
        fake.default_dtype = "float64"
        with self.assertRaisesRegex(RuntimeError, "defaults CPU/float32"):
            runner._verify_cpu_runtime(fake)

    def test_expected_and_unexpected_outcomes_fail_required_suite(self) -> None:
        for attribute in ("expectedFailures", "unexpectedSuccesses"):
            result = _successful_result()
            setattr(result, attribute, [("case", "detail")])
            successful, failed, expected, unexpected = (
                runner._required_suite_outcome(result)
            )
            with self.subTest(attribute=attribute):
                self.assertFalse(successful)
                self.assertEqual(failed, 1)
                self.assertEqual(expected + unexpected, 1)

    def test_expected_and_unexpected_outcomes_make_run_terminally_failed(
        self,
    ) -> None:
        for attribute, metric in (
            ("expectedFailures", "validation/tests_expected_failures"),
            ("unexpectedSuccesses", "validation/tests_unexpected_successes"),
        ):
            result = _successful_result()
            setattr(result, attribute, [("case", "detail")])
            with (
                self.subTest(attribute=attribute),
                tempfile.TemporaryDirectory() as directory,
            ):
                run_dir = Path(directory) / "run"
                with (
                    mock.patch.object(
                        runner, "_corpus_record", side_effect=_corpus_record
                    ),
                    mock.patch.object(
                        runner,
                        "_source_snapshot",
                        side_effect=[_snapshot(), _snapshot()],
                    ),
                    mock.patch.object(
                        runner, "_discover_repository_suite", return_value=object()
                    ),
                    mock.patch.object(
                        runner.unittest.TextTestRunner,
                        "run",
                        return_value=result,
                    ),
                ):
                    self.assertEqual(runner.main(["--run-dir", str(run_dir)]), 1)
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["primary_metrics"][metric], 1)
                self.assertEqual(
                    summary["primary_metrics"]["validation/tests_failed"], 1
                )

    def test_secondary_failure_note_cannot_replace_a_hostile_primary(self) -> None:
        class LockedError(RuntimeError):
            def add_note(self, note):
                raise RuntimeError("notes locked")

            def __setattr__(self, name, value):
                raise AttributeError("attributes locked")

        primary = LockedError("primary")
        runner._note_secondary_failure(primary, "secondary")
        self.assertEqual(str(primary), "primary")

    def test_atomic_json_fsyncs_file_and_parent_directory(self) -> None:
        calls = []
        real_fsync = os.fsync

        def observed_fsync(descriptor):
            calls.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
            return real_fsync(descriptor)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner.os, "fsync", side_effect=observed_fsync
        ):
            path = Path(directory) / "record.json"
            runner._atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
        self.assertEqual(calls, [False, True])

    def test_atomic_json_cleanup_cannot_replace_primary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            primary = OSError("synthetic replace failure")
            with (
                mock.patch.object(runner.os, "replace", side_effect=primary),
                mock.patch.object(
                    runner.os,
                    "unlink",
                    side_effect=OSError("synthetic cleanup failure"),
                ),
                self.assertRaises(OSError) as raised,
            ):
                runner._atomic_json(path, {"ok": True})
            self.assertIs(raised.exception, primary)

    def test_atomic_json_does_not_delete_a_recreated_staging_name(self) -> None:
        recreated = None
        real_replace = os.replace

        def replace_and_recreate(source, target):
            nonlocal recreated
            real_replace(source, target)
            recreated = Path(source)
            recreated.write_bytes(b"foreign")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            with mock.patch.object(
                runner.os, "replace", side_effect=replace_and_recreate
            ):
                runner._atomic_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertIsNotNone(recreated)
            assert recreated is not None
            self.assertEqual(recreated.read_bytes(), b"foreign")
            recreated.unlink()

    def test_unchanged_dirty_source_is_not_an_exact_commit(self) -> None:
        before = _snapshot(dirty=True)
        manifest = {"source": {"before": before, "exact_commit": False}}
        self.assertFalse(runner._record_final_source(manifest, _snapshot(dirty=True)))
        self.assertFalse(manifest["source"]["changed_during_run"])
        self.assertFalse(manifest["source"]["exact_commit"])

    def test_source_fingerprint_covers_contents_and_excludes_own_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)

            def git(*arguments: str) -> None:
                subprocess.run(
                    ("git", *arguments),
                    cwd=repository,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            git("init", "-q")
            git("config", "user.name", "runner-test")
            git("config", "user.email", "runner-test@example.invalid")
            tracked = repository / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            git("add", "tracked.py")
            git("commit", "-qm", "initial")

            with mock.patch.object(runner, "_REPOSITORY_ROOT", repository):
                clean = runner._source_snapshot(excluded_run_dir=b"runs/example")
                self.assertFalse(clean["dirty"])

                own_run = repository / "runs" / "example"
                own_run.mkdir(parents=True)
                (own_run / "run.json").write_text("{}\n", encoding="utf-8")
                excluded = runner._source_snapshot(excluded_run_dir=b"runs/example")
                self.assertEqual(excluded, clean)

                untracked = repository / "new.py"
                untracked.write_text("VALUE = 2\n", encoding="utf-8")
                first = runner._source_snapshot(excluded_run_dir=b"runs/example")
                untracked.write_text("VALUE = 3\n", encoding="utf-8")
                second = runner._source_snapshot(excluded_run_dir=b"runs/example")
                self.assertTrue(first["dirty"])
                self.assertEqual(first["status_counts"]["untracked"], 1)
                self.assertNotEqual(
                    first["worktree_fingerprint"], second["worktree_fingerprint"]
                )

                untracked.chmod(0o755)
                executable = runner._source_snapshot(
                    excluded_run_dir=b"runs/example"
                )
                self.assertNotEqual(
                    second["worktree_fingerprint"],
                    executable["worktree_fingerprint"],
                )

                tracked.write_text("VALUE = 4\n", encoding="utf-8")
                third = runner._source_snapshot(excluded_run_dir=b"runs/example")
                self.assertEqual(third["status_counts"]["tracked"], 1)
                self.assertNotEqual(
                    second["worktree_fingerprint"], third["worktree_fingerprint"]
                )

    def test_explicit_main_argv_and_clean_source_identity_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            arguments = ["--run-dir", str(run_dir), "--verbosity", "1"]
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=_successful_result(),
                ),
            ):
                self.assertEqual(runner.main(arguments), 0)

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(
                manifest["command"]["argv"],
                [sys.executable, "scripts/run_development_corpus.py", *arguments],
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(manifest["source"]["changed_during_run"])
            self.assertTrue(manifest["source"]["exact_commit"])
            self.assertEqual(manifest["source"]["before"], _snapshot())
            self.assertEqual(manifest["source"]["after"], _snapshot())

    def test_workload_exception_is_terminal_and_is_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner,
                    "_discover_repository_suite",
                    side_effect=RuntimeError("synthetic discovery failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic discovery failure"):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIsNotNone(manifest["ended_at"])
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["exit_code"], 1)
            self.assertEqual(
                summary["error"], {"stage": "workload", "type": "RuntimeError"}
            )

    def test_keyboard_interrupt_is_terminal_and_is_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner,
                    "_discover_repository_suite",
                    side_effect=KeyboardInterrupt,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["exit_code"], 130)

    def test_source_change_turns_a_passing_suite_into_a_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner,
                    "_source_snapshot",
                    side_effect=[
                        _snapshot(fingerprint="a" * 64),
                        _snapshot(fingerprint="c" * 64),
                    ],
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=_successful_result(),
                ),
            ):
                self.assertEqual(runner.main(["--run-dir", str(run_dir)]), 1)

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["source"]["changed_during_run"])
            self.assertFalse(manifest["source"]["exact_commit"])
            self.assertEqual(summary["status"], "failed")
            self.assertIn("source changed", summary["conclusion"])

    def test_a_skipped_corpus_case_is_a_failed_run(self) -> None:
        skipped_result = SimpleNamespace(
            failures=[],
            errors=[],
            skipped=[("case", "synthetic skip")],
            expectedFailures=[],
            unexpectedSuccesses=[],
            testsRun=4,
            wasSuccessful=lambda: True,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=skipped_result,
                ),
            ):
                self.assertEqual(runner.main(["--run-dir", str(run_dir)]), 1)

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(
                summary["primary_metrics"]["validation/tests_skipped"], 1
            )

    def test_empty_discovery_is_a_failed_run(self) -> None:
        empty_result = SimpleNamespace(
            failures=[],
            errors=[],
            skipped=[],
            expectedFailures=[],
            unexpectedSuccesses=[],
            testsRun=0,
            wasSuccessful=lambda: True,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=empty_result,
                ),
            ):
                self.assertEqual(runner.main(["--run-dir", str(run_dir)]), 1)

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(
                summary["primary_metrics"]["validation/tests_run"], 0
            )

    def test_setup_failure_is_recorded_as_terminal_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            real_atomic_json = runner._atomic_json
            failed = False

            def fail_corpus_once(path, value):
                nonlocal failed
                if not failed and path.name == "corpus.json":
                    failed = True
                    raise OSError("synthetic corpus write failure")
                return real_atomic_json(path, value)

            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_atomic_json", side_effect=fail_corpus_once
                ),
            ):
                with self.assertRaisesRegex(OSError, "synthetic corpus write"):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error"]["stage"], "setup")

    def test_final_source_snapshot_failure_is_terminal_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner,
                    "_source_snapshot",
                    side_effect=[
                        _snapshot(),
                        RuntimeError("synthetic final snapshot failure"),
                        _snapshot(),
                    ],
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=_successful_result(),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic final snapshot"):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["error"]["stage"], "finalize")
            self.assertFalse(manifest["source"]["exact_commit"])

    def test_metrics_failure_is_recorded_as_terminal_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=_successful_result(),
                ),
                mock.patch.object(
                    runner,
                    "_append_json_line",
                    side_effect=OSError("synthetic metrics write failure"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "synthetic metrics write"):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["error"]["stage"], "finalize")
            self.assertEqual((run_dir / "metrics.jsonl").read_text(), "")

    def test_terminal_write_failure_is_retried_as_failed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            real_atomic_json = runner._atomic_json
            failed = False

            def fail_first_summary(path, value):
                nonlocal failed
                if not failed and path.name == "summary.json":
                    failed = True
                    raise OSError("synthetic terminal write failure")
                return real_atomic_json(path, value)

            with (
                mock.patch.object(runner, "_corpus_record", side_effect=_corpus_record),
                mock.patch.object(
                    runner, "_source_snapshot", side_effect=[_snapshot(), _snapshot()]
                ),
                mock.patch.object(
                    runner, "_discover_repository_suite", return_value=object()
                ),
                mock.patch.object(
                    runner.unittest.TextTestRunner,
                    "run",
                    return_value=_successful_result(),
                ),
                mock.patch.object(
                    runner, "_atomic_json", side_effect=fail_first_summary
                ),
            ):
                with self.assertRaisesRegex(OSError, "synthetic terminal write"):
                    runner.main(["--run-dir", str(run_dir)])

            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error"]["stage"], "finalize")


if __name__ == "__main__":
    unittest.main()
