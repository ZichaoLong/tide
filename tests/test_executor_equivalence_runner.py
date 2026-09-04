from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_executor_equivalence as runner


def _snapshot(*, fingerprint: str = "a" * 64, dirty: bool = False):
    return {
        "fingerprint_schema": "tide.worktree-fingerprint.v1",
        "commit": "1" * 40,
        "dirty": dirty,
        "worktree_fingerprint": fingerprint,
        "observation_policy": {"sampling": "test-double"},
        "status_counts": {
            "entries": int(dirty),
            "tracked": int(dirty),
            "untracked": 0,
        },
    }


def _qualification_gaps():
    return [
        {"id": "fixtures.self-contained-bundles", "detail": "not frozen"},
        {"id": "optimizer.minimum-16", "detail": "not in this suite"},
    ]


def _executor_record(accepted_ids, vjp_ids):
    return {
        "accepted_count": len(accepted_ids),
        "accepted_case_ids": list(accepted_ids),
        "rejected_count": 256 - len(accepted_ids),
        "vjp_test_case_count": len(vjp_ids),
        "vjp_test_case_ids": list(vjp_ids),
    }


def _corpus_record():
    case_ids = [f"synthetic.case.{ordinal:03d}" for ordinal in range(256)]
    single_layer_ids = [case_ids[23], *case_ids[:7]]
    hb_ids = [case_ids[144], *case_ids[24:39]]
    return (
        {
            "schema_version": 1,
            "kind": "tide.core-v1-executor-equivalence-candidate.v1",
            "qualification": False,
            "qualification_gaps": _qualification_gaps(),
            "seed": 20260903,
            "legal_count": 256,
            "vjp_plan_count": 64,
            "candidate_identity_sha256": "c" * 64,
            "dtypes": ["float64", "float32"],
            "comparison_contract": {
                "float64": {"name": "T64", "atol": 1e-10, "rtol": 1e-8},
                "float32": {"name": "T32", "atol": 1e-6, "rtol": 1e-5},
                "finite_required": True,
                "discrete_values": "exact",
                "live_autograd_metadata": {
                    "requires_grad": "exact-by-public-tensor-occurrence",
                    "is_leaf": "not-compared",
                    "grad_fn": "not-compared",
                },
            },
            "equivalence_matrix": {
                "packed": {
                    "forward_case_count": 256,
                    "vjp_case_count": 64,
                    "dtypes": ["float64", "float32"],
                    "forward_execution_modes": [
                        "full-prefill",
                        "two-chunk-prefill",
                        "token-by-token-decode",
                    ],
                    "vjp_execution_modes": ["full-prefill"],
                    "compares_to": ["token-major-eager-reference"],
                },
                "single_layer": {
                    "forward_case_count": 8,
                    "vjp_case_count": 8,
                    "dtypes": ["float64", "float32"],
                    "forward_execution_modes": [
                        "full-prefill",
                        "two-chunk-prefill",
                        "token-by-token-decode",
                    ],
                    "vjp_execution_modes": ["full-prefill"],
                    "compares_to": [
                        "token-major-eager-reference",
                        "tide.generic-packed.torch.v1",
                    ],
                },
                "hb": {
                    "forward_case_count": 16,
                    "vjp_case_count": 16,
                    "dtypes": ["float64", "float32"],
                    "forward_execution_modes": [
                        "full-prefill",
                        "two-chunk-prefill",
                        "token-by-token-decode",
                    ],
                    "vjp_execution_modes": ["full-prefill"],
                    "compares_to": [
                        "token-major-eager-reference",
                        "tide.generic-packed.torch.v1",
                    ],
                },
            },
            "implementation_variants": list(runner._IMPLEMENTATION_VARIANTS),
            "support": {
                "partition_sha256": "p" * 64,
                "executors": {
                    "packed": _executor_record(case_ids, case_ids[:64]),
                    "single_layer": _executor_record(
                        single_layer_ids, single_layer_ids
                    ),
                    "hb": _executor_record(hb_ids, hb_ids),
                },
            },
            "cases": [
                {"ordinal": ordinal, "case_id": case_id}
                for ordinal, case_id in enumerate(case_ids)
            ],
        },
        "d" * 64,
    )


def _discovery():
    required = sorted(runner._REQUIRED_SEMANTIC_TEST_IDS)
    total = sum(runner._EXPECTED_MODULE_TEST_COUNTS.values())
    test_ids = tuple(
        [*required, *(f"synthetic.Example.test_{index:03d}" for index in range(total - len(required)))]
    )
    return runner._DiscoveredSuite(
        suite=unittest.TestSuite(),
        test_ids=test_ids,
        module_test_counts=dict(runner._EXPECTED_MODULE_TEST_COUNTS),
    )


def _result(
    *,
    tests_run: int = 60,
    skipped=(),
    expected_failures=(),
    unexpected_successes=(),
    failures=(),
    errors=(),
):
    successful = not (
        failures or errors or unexpected_successes
    )
    return SimpleNamespace(
        failures=list(failures),
        errors=list(errors),
        skipped=list(skipped),
        expectedFailures=list(expected_failures),
        unexpectedSuccesses=list(unexpected_successes),
        testsRun=tests_run,
        wasSuccessful=lambda: successful,
    )


def _complete_receipts():
    corpus, _ = _corpus_record()
    support = corpus["support"]["executors"]
    receipts = []

    def add(**record):
        receipts.append(
            {"sequence": len(receipts), "outcome": "passed", **record}
        )

    modes = (
        ("full-prefill", 1, "full-call"),
        ("two-chunk-prefill", 4, "each-call-and-canonical-merge"),
        ("token-by-token-decode", 3, "each-call-and-canonical-merge"),
    )
    observables = [
        "output",
        "final-state",
        "balance-sufficient-statistics",
        "canonical-trace",
        "exact-route",
    ]
    executor_rows = (
        (runner._PACKED_IMPLEMENTATION_ID, support["packed"]),
        ("single-layer.v1", support["single_layer"]),
        ("hb-line.v1", support["hb"]),
    )
    for executor, executor_support in executor_rows:
        for case_id in executor_support["accepted_case_ids"]:
            for dtype in runner._DTYPES:
                for mode, calls, trace_scope in modes:
                    call_counts = {"eager": calls, "packed": calls}
                    if executor != runner._PACKED_IMPLEMENTATION_ID:
                        call_counts["specialized"] = calls
                    record = dict(
                        kind="forward-cell",
                        executor=executor,
                        case_id=case_id,
                        dtype=dtype,
                        mode=mode,
                        call_counts=call_counts,
                        observables=observables,
                        trace_scope=trace_scope,
                    )
                    if mode == "two-chunk-prefill":
                        record["splits"] = [1, 2]
                    add(**record)

    packed_objectives = (
        ("output", "output"),
        ("balance", "balance-loss"),
        ("final-state:x:y:tensor", "final-state-component"),
        ("balance-region:r", "balance-region-soft-sum"),
        ("trace.region-logits:s:0:r:0", "trace-region-event-logits"),
        (
            "trace.region-probabilities:s:0:r:0",
            "trace-region-event-probabilities",
        ),
        ("combined", "combined-output-balance-state"),
        ("output.repeat", "output"),
    )
    for executor, executor_support in executor_rows:
        for case_id in executor_support["vjp_test_case_ids"]:
            for dtype in runner._DTYPES:
                objectives = (
                    packed_objectives
                    if executor == runner._PACKED_IMPLEMENTATION_ID
                    else ((
                        "combined-output-balance-state",
                        "combined-output-balance-state",
                    ),)
                )
                objective_ids = []
                for objective_id, family in objectives:
                    objective_ids.append(objective_id)
                    add(
                        kind="vjp-objective",
                        executor=executor,
                        case_id=case_id,
                        dtype=dtype,
                        mode="full-prefill",
                        objective_id=objective_id,
                        objective_family=family,
                    )
                add(
                    kind="vjp-case-complete",
                    executor=executor,
                    case_id=case_id,
                    dtype=dtype,
                    mode="full-prefill",
                    objective_ids=objective_ids,
                )

    cases = {case["ordinal"]: case["case_id"] for case in corpus["cases"]}
    for executor, ordinal in (
        (runner._PACKED_IMPLEMENTATION_ID, 176),
        (runner._PACKED_IMPLEMENTATION_ID, 19),
        ("single-layer.v1", 23),
        ("hb-line.v1", 144),
    ):
        add(
            kind="lifecycle-scenario",
            executor=executor,
            case_id=cases[ordinal],
            dtype="float64",
        )
    return tuple(receipts)


class ExecutorEquivalenceRunnerTests(unittest.TestCase):
    def _run_with_result(
        self,
        run_dir: Path,
        result,
        *,
        snapshots=None,
        arguments=None,
        receipts=None,
    ) -> int:
        if snapshots is None:
            snapshots = [_snapshot(), _snapshot()]
        if arguments is None:
            arguments = ["--run-dir", str(run_dir)]
        if receipts is None:
            receipts = _complete_receipts()
        with (
            mock.patch.object(
                runner, "_corpus_record", side_effect=_corpus_record
            ),
            mock.patch.object(
                runner, "_source_snapshot", side_effect=snapshots
            ),
            mock.patch.object(
                runner, "_discover_repository_suite", return_value=_discovery()
            ),
            mock.patch.object(
                runner.unittest.TextTestRunner, "run", return_value=result
            ),
            mock.patch.object(runner, "_reset_repository_receipts"),
            mock.patch.object(
                runner,
                "_collect_repository_receipts",
                return_value=receipts,
            ),
            mock.patch("builtins.print"),
        ):
            return runner.main(arguments)

    def test_real_corpus_record_freezes_identity_support_and_vjp_sets(self) -> None:
        runner._prepare_repository_imports()
        corpus, artifact_sha256 = runner._corpus_record()
        self.assertEqual(corpus["legal_count"], 256)
        self.assertEqual(corpus["vjp_plan_count"], 64)
        self.assertEqual(
            corpus["candidate_identity_sha256"],
            runner._CANDIDATE_IDENTITY_SHA256,
        )
        self.assertEqual(
            corpus["support"]["partition_sha256"],
            runner._SUPPORT_PARTITION_SHA256,
        )
        self.assertEqual(len(artifact_sha256), 64)
        self.assertEqual(
            [case["ordinal"] for case in corpus["cases"]], list(range(256))
        )
        self.assertEqual(
            set(corpus["cases"][0]["typed_plan_hashes"]),
            {"float64", "float32"},
        )

        support = corpus["support"]["executors"]
        self.assertEqual(
            (
                support["packed"]["accepted_count"],
                support["packed"]["rejected_count"],
                support["packed"]["vjp_test_case_count"],
                support["packed"]["marked_vjp_intersection_count"],
            ),
            (256, 0, 64, 64),
        )
        self.assertEqual(
            (
                support["single_layer"]["accepted_count"],
                support["single_layer"]["rejected_count"],
                support["single_layer"]["vjp_test_case_count"],
                support["single_layer"]["marked_vjp_intersection_count"],
            ),
            (8, 248, 8, 0),
        )
        self.assertEqual(
            (
                support["hb"]["accepted_count"],
                support["hb"]["rejected_count"],
                support["hb"]["vjp_test_case_count"],
                support["hb"]["marked_vjp_intersection_count"],
            ),
            (16, 240, 16, 4),
        )
        self.assertEqual(
            support["packed"]["accepted_case_ids"],
            [case["case_id"] for case in corpus["cases"]],
        )
        self.assertEqual(
            support["packed"]["vjp_test_case_ids"],
            [case["case_id"] for case in corpus["cases"] if case["vjp"]],
        )
        self.assertEqual(
            support["single_layer"]["vjp_test_case_ids"],
            support["single_layer"]["accepted_case_ids"],
        )
        self.assertEqual(
            support["hb"]["vjp_test_case_ids"],
            support["hb"]["accepted_case_ids"],
        )
        self.assertEqual(len(corpus["support"]["partition_rows"]), 256)

    def test_record_is_explicitly_nonqualifying_and_names_all_variants(self) -> None:
        runner._prepare_repository_imports()
        corpus, _ = runner._corpus_record()
        self.assertFalse(corpus["qualification"])
        self.assertEqual(
            corpus["implementation_variants"],
            [
                "token-major-eager-reference",
                "tide.generic-packed.torch.v1",
                "single-layer.v1",
                "hb-line.v1",
            ],
        )
        self.assertEqual(corpus["dtypes"], ["float64", "float32"])
        self.assertEqual(
            corpus["comparison_contract"],
            {
                "float64": {"name": "T64", "atol": 1e-10, "rtol": 1e-8},
                "float32": {"name": "T32", "atol": 1e-6, "rtol": 1e-5},
                "finite_required": True,
                "discrete_values": "exact",
                "live_autograd_metadata": {
                    "requires_grad": "exact-by-public-tensor-occurrence",
                    "is_leaf": "not-compared",
                    "grad_fn": "not-compared",
                },
            },
        )
        self.assertEqual(
            corpus["equivalence_matrix"]["packed"]["forward_execution_modes"],
            [
                "full-prefill",
                "two-chunk-prefill",
                "token-by-token-decode",
            ],
        )
        self.assertEqual(
            corpus["equivalence_matrix"]["packed"]["trace_execution_modes"],
            [
                "full-prefill",
                "each-two-chunk-prefill-call",
                "each-token-by-token-decode-call",
            ],
        )
        self.assertEqual(
            corpus["equivalence_matrix"]["packed"]["vjp_objective_policy"],
            "isolated-public-roots-and-combined",
        )
        self.assertIn(
            "each-final-state-tensor-component-with-frozen-cotangent",
            corpus["equivalence_matrix"]["packed"]["vjp_objectives"],
        )
        self.assertIn(
            "each-trace-region-event-logits-with-frozen-cotangent",
            corpus["equivalence_matrix"]["packed"]["vjp_objectives"],
        )
        self.assertIn(
            "none-vs-tensor-connectivity",
            corpus["equivalence_matrix"]["packed"]["vjp_observables"],
        )
        for name in ("single_layer", "hb"):
            self.assertEqual(
                corpus["equivalence_matrix"][name]["forward_execution_modes"],
                [
                    "full-prefill",
                    "two-chunk-prefill",
                    "token-by-token-decode",
                ],
            )
        for name in ("packed", "single_layer", "hb"):
            self.assertEqual(
                corpus["equivalence_matrix"][name]["vjp_execution_modes"],
                ["full-prefill"],
            )
        self.assertEqual(
            (
                corpus["equivalence_matrix"]["packed"]["forward_case_count"],
                corpus["equivalence_matrix"]["packed"]["vjp_case_count"],
                corpus["equivalence_matrix"]["single_layer"][
                    "forward_case_count"
                ],
                corpus["equivalence_matrix"]["single_layer"]["vjp_case_count"],
                corpus["equivalence_matrix"]["hb"]["forward_case_count"],
                corpus["equivalence_matrix"]["hb"]["vjp_case_count"],
            ),
            (256, 64, 8, 8, 16, 16),
        )
        gap_ids = {gap["id"] for gap in corpus["qualification_gaps"]}
        self.assertTrue(
            {
                "fixtures.self-contained-bundles",
                "gates.i00-c01",
                "vjp.finite-difference",
                "optimizer.minimum-16",
                "negative-and-scenarios",
                "capability.fresh-process-exact-source",
                "platform.coverage",
                "integration-and-performance",
            }.issubset(gap_ids)
        )
        scenario_gap = next(
            gap
            for gap in corpus["qualification_gaps"]
            if gap["id"] == "negative-and-scenarios"
        )["detail"]
        self.assertIn("all 256 packed candidates", scenario_gap)
        self.assertIn("all 24 specialization-supported cases", scenario_gap)
        self.assertIn("cover full prefill", scenario_gap)
        self.assertIn("both nonempty T=3 two-chunk splits", scenario_gap)
        self.assertIn("token-by-token decode", scenario_gap)
        self.assertIn("complete short/long chunk-partition", scenario_gap)
        self.assertIn("concurrency, rollback", scenario_gap)
        self.assertIn("checkpoint/resume scenarios", scenario_gap)

    def test_discovery_contains_every_required_repository_module(self) -> None:
        runner._prepare_repository_imports()
        discovery = runner._discover_repository_suite()
        self.assertEqual(
            set(discovery.module_test_counts), set(runner._REQUIRED_TEST_MODULES)
        )
        self.assertTrue(
            all(count > 0 for count in discovery.module_test_counts.values())
        )
        self.assertEqual(
            discovery.module_test_counts,
            runner._EXPECTED_MODULE_TEST_COUNTS,
        )
        self.assertTrue(
            runner._REQUIRED_SEMANTIC_TEST_IDS.issubset(discovery.test_ids)
        )
        self.assertEqual(
            len(discovery.test_ids), sum(discovery.module_test_counts.values())
        )

    def test_foreign_preloaded_required_test_module_is_rejected(self) -> None:
        module_name = Path(runner._REQUIRED_TEST_MODULES[0]).stem
        foreign = types.ModuleType(module_name)
        foreign.__file__ = "/tmp/foreign/test_core_v1_candidate_corpus.py"
        with mock.patch.dict(sys.modules, {module_name: foreign}):
            with self.assertRaisesRegex(RuntimeError, "must be imported from"):
                runner._discover_repository_suite()

    def test_missing_required_test_module_is_rejected(self) -> None:
        module_name = Path(runner._REQUIRED_TEST_MODULES[0]).stem
        with (
            mock.patch.dict(sys.modules, {module_name: None}),
            mock.patch.object(
                runner.unittest.defaultTestLoader,
                "discover",
                return_value=unittest.TestSuite(),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not import"):
                runner._discover_repository_suite()

    def test_success_writes_complete_record_and_exact_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            arguments = ["--run-dir", str(run_dir), "--verbosity", "1"]
            self.assertEqual(
                self._run_with_result(
                    run_dir, _result(), arguments=arguments
                ),
                0,
            )

            expected = {
                "run.json",
                "metrics.jsonl",
                "stdout.log",
                "summary.json",
                "artifacts/corpus.json",
                "artifacts/execution-receipt.json",
            }
            self.assertEqual(
                {
                    str(path.relative_to(run_dir))
                    for path in run_dir.rglob("*")
                    if path.is_file()
                },
                expected,
            )
            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            corpus = json.loads(
                (run_dir / "artifacts" / "corpus.json").read_text()
            )
            metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(len(metrics_lines), 1)
            metrics = json.loads(metrics_lines[0])
            receipt = json.loads(
                (
                    run_dir / "artifacts" / "execution-receipt.json"
                ).read_text()
            )

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(
                manifest["command"]["argv"],
                [
                    sys.executable,
                    "scripts/run_executor_equivalence.py",
                    *arguments,
                ],
            )
            self.assertEqual(manifest["source"]["before"], _snapshot())
            self.assertEqual(manifest["source"]["after"], _snapshot())
            self.assertTrue(manifest["source"]["exact_commit"])
            self.assertFalse(manifest["source"]["changed_during_run"])
            self.assertFalse(manifest["qualification"])
            self.assertFalse(summary["qualification"])
            self.assertFalse(corpus["qualification"])
            self.assertEqual(
                manifest["runtime"]["implementation_variants"],
                list(runner._IMPLEMENTATION_VARIANTS),
            )
            self.assertEqual(
                manifest["experiment"]["suite"]["required_modules"],
                list(runner._REQUIRED_TEST_MODULES),
            )
            self.assertEqual(metrics["metrics"]["validation/tests_run"], 60)
            self.assertEqual(
                metrics["metrics"]["validation/tests_discovered"], 60
            )
            self.assertEqual(
                metrics["metrics"]["validation/suite_complete"], 1
            )
            self.assertEqual(
                metrics["metrics"]["validation/execution_receipt_complete"],
                1,
            )
            self.assertEqual(metrics["metrics"]["validation/tests_failed"], 0)
            self.assertEqual(metrics["metrics"]["support/legal_plans"], 256)
            self.assertEqual(metrics["metrics"]["support/vjp_plans"], 64)
            self.assertEqual(metrics["metrics"]["support/packed_accepted"], 256)
            self.assertEqual(
                metrics["metrics"]["support/single_layer_vjp_cases"], 8
            )
            self.assertEqual(metrics["metrics"]["support/hb_vjp_cases"], 16)
            self.assertEqual(
                metrics["metrics"]["coverage/packed_forward_cells"], 1536
            )
            self.assertEqual(
                metrics["metrics"]["coverage/single_layer_forward_cells"], 48
            )
            self.assertEqual(
                metrics["metrics"]["coverage/hb_forward_cells"], 96
            )
            self.assertEqual(
                metrics["metrics"]["coverage/packed_vjp_case_dtype_groups"],
                128,
            )
            self.assertEqual(receipt["validation"]["status"], "passed")
            self.assertEqual(receipt["derived_coverage"]["forward_cells"], 1680)
            self.assertEqual(
                manifest["evidence"]["execution_receipt"]["sha256"],
                summary["execution_receipt_sha256"],
            )
            self.assertEqual(metrics["metrics"]["qualification/passed"], 0)

    def test_existing_run_directory_is_refused_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            sentinel = run_dir / "owned.txt"
            sentinel.write_bytes(b"preserve-me")
            before = tuple(
                (str(path.relative_to(run_dir)), path.read_bytes())
                for path in run_dir.rglob("*")
                if path.is_file()
            )
            with (
                mock.patch.object(
                    runner, "_corpus_record", side_effect=_corpus_record
                ),
                mock.patch.object(
                    runner, "_source_snapshot", return_value=_snapshot()
                ),
            ):
                with self.assertRaises(FileExistsError):
                    runner.main(["--run-dir", str(run_dir)])
            after = tuple(
                (str(path.relative_to(run_dir)), path.read_bytes())
                for path in run_dir.rglob("*")
                if path.is_file()
            )
            self.assertEqual(after, before)

    def test_skip_expected_failure_and_incomplete_suite_are_terminal_failures(
        self,
    ) -> None:
        variants = (
            (_result(skipped=(("case", "skip"),)), "validation/tests_skipped"),
            (
                _result(expected_failures=(("case", "expected"),)),
                "validation/tests_expected_failures",
            ),
            (_result(tests_run=0), "validation/tests_run"),
            (_result(tests_run=59), "validation/suite_complete"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (result, metric) in enumerate(variants):
                with self.subTest(metric=metric):
                    run_dir = Path(directory) / f"run-{index}"
                    self.assertEqual(self._run_with_result(run_dir, result), 1)
                    summary = json.loads((run_dir / "summary.json").read_text())
                    self.assertEqual(summary["status"], "failed")
                    expected = (
                        0
                        if metric
                        in {"validation/tests_run", "validation/suite_complete"}
                        else 1
                    )
                    self.assertEqual(summary["primary_metrics"][metric], expected)

    def test_missing_execution_receipt_cell_is_a_terminal_failure(self) -> None:
        receipts = list(_complete_receipts())
        removed = next(
            index
            for index, receipt in enumerate(receipts)
            if receipt["kind"] == "forward-cell"
        )
        del receipts[removed]
        receipts = tuple(
            {**receipt, "sequence": index}
            for index, receipt in enumerate(receipts)
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            self.assertEqual(
                self._run_with_result(
                    run_dir,
                    _result(),
                    receipts=receipts,
                ),
                1,
            )
            summary = json.loads((run_dir / "summary.json").read_text())
            receipt = json.loads(
                (
                    run_dir / "artifacts" / "execution-receipt.json"
                ).read_text()
            )
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(receipt["validation"]["status"], "failed")
            self.assertEqual(
                summary["primary_metrics"][
                    "validation/execution_receipt_complete"
                ],
                0,
            )
            self.assertTrue(
                any(
                    "missing 1 forward cells" in error
                    for error in receipt["validation"]["errors"]
                )
            )

    def test_source_change_fails_passing_suite_and_clears_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            self.assertEqual(
                self._run_with_result(
                    run_dir,
                    _result(),
                    snapshots=[
                        _snapshot(fingerprint="a" * 64),
                        _snapshot(fingerprint="b" * 64),
                    ],
                ),
                1,
            )
            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertTrue(manifest["source"]["changed_during_run"])
            self.assertFalse(manifest["source"]["exact_commit"])
            self.assertEqual(summary["status"], "failed")
            self.assertIn("source changed", summary["conclusion"])

    def test_workload_exception_is_recorded_and_reraised_unchanged(self) -> None:
        primary = RuntimeError("synthetic discovery failure")
        with tempfile.TemporaryDirectory() as directory:
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
                    runner,
                    "_discover_repository_suite",
                    side_effect=primary,
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    runner.main(["--run-dir", str(run_dir)])
            self.assertIs(raised.exception, primary)
            manifest = json.loads((run_dir / "run.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(
                summary["error"], {"stage": "workload", "type": "RuntimeError"}
            )
            self.assertEqual((run_dir / "metrics.jsonl").read_text(), "")

    def test_keyboard_interrupt_is_recorded_and_reraised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
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
                    runner,
                    "_discover_repository_suite",
                    side_effect=KeyboardInterrupt,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.main(["--run-dir", str(run_dir)])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["exit_code"], 130)

    def test_setup_failure_writes_terminal_record_when_possible(self) -> None:
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
                mock.patch.object(
                    runner, "_corpus_record", side_effect=_corpus_record
                ),
                mock.patch.object(
                    runner,
                    "_source_snapshot",
                    side_effect=[_snapshot(), _snapshot()],
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


if __name__ == "__main__":
    unittest.main()
