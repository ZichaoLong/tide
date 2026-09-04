#!/usr/bin/env python3
"""Run the bounded CPU executor-equivalence suite with durable records.

This is a controlled development/equivalence runner.  It intentionally does
not claim a core-v1 qualification result: the candidate Plans are generated
records rather than frozen self-contained fixture bundles, and several gates
from the qualification plan remain outside this suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import unittest
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# ``python scripts/run_executor_equivalence.py`` puts ``scripts/`` rather than
# the checkout root first on sys.path.  Put this checkout first for the
# project-owned recording helpers; authoritative ``tide`` imports are
# established later, after the initial source snapshot.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_root_text = str(_REPOSITORY_ROOT)
_retained_sys_path = []
for _entry in sys.path:
    try:
        _same_root = Path(_entry or os.curdir).resolve() == _REPOSITORY_ROOT
    except (OSError, RuntimeError, TypeError):
        _same_root = False
    if not _same_root:
        _retained_sys_path.append(_entry)
sys.path[:] = [_root_text, *_retained_sys_path]

from scripts import run_development_corpus as _record_helpers  # noqa: E402

_helper_source = getattr(_record_helpers, "__file__", None)
if not isinstance(_helper_source, str) or Path(_helper_source).resolve() != (
    _REPOSITORY_ROOT / "scripts" / "run_development_corpus.py"
).resolve():
    raise RuntimeError(
        "executor-equivalence recording helpers must come from this checkout"
    )

from scripts.run_development_corpus import (  # noqa: E402
    _Tee,
    _append_json_line,
    _atomic_json,
    _create_empty_file,
    _establish_cpu_runtime,
    _fsync_file,
    _json_bytes,
    _note_secondary_failure,
    _prepare_repository_imports,
    _record_final_source,
    _require_module_file,
    _required_suite_outcome,
    _run_dir_exclusion,
    _source_snapshot,
    _utc_now,
    _verify_cpu_runtime,
    _write_terminal_record,
)


_REQUIRED_TEST_MODULES = (
    "test_core_v1_candidate_corpus.py",
    "test_core_v1_executor_equivalence.py",
    "test_packed.py",
    "test_specialized.py",
)
_EXPECTED_MODULE_TEST_COUNTS = {
    "test_core_v1_candidate_corpus.py": 4,
    "test_core_v1_executor_equivalence.py": 14,
    "test_packed.py": 25,
    "test_specialized.py": 17,
}
_REQUIRED_SEMANTIC_TEST_IDS = frozenset(
    {
        "test_core_v1_executor_equivalence.ExecutorContractParityTests."
        "test_all_executor_bindings_require_exact_boolean_detach_at_end",
        "test_core_v1_executor_equivalence.CandidateForwardEquivalenceTests."
        "test_all_packed_cases_match_full_chunked_and_decode_execution",
        "test_core_v1_executor_equivalence.CandidateForwardEquivalenceTests."
        "test_all_specialized_supported_cases_match_both_cpu_dtypes",
        "test_core_v1_executor_equivalence.CandidateForwardEquivalenceTests."
        "test_all_specialized_cases_match_live_public_result_metadata",
        "test_core_v1_executor_equivalence.CandidateVJPEquivalenceTests."
        "test_all_64_marked_packed_vjp_cases_match_values_and_none",
        "test_core_v1_executor_equivalence.CandidateVJPEquivalenceTests."
        "test_every_statically_accepted_specialization_matches_vjp_and_none",
        "test_core_v1_executor_equivalence.CandidateVJPEquivalenceTests."
        "test_hb_external_differentiable_initial_state_matches_three_way",
        "test_core_v1_executor_equivalence.CandidateVJPEquivalenceTests."
        "test_prefill_and_decode_default_detach_match_no_detach_three_way",
        "test_core_v1_executor_equivalence.CandidateLifecycleEquivalenceTests."
        "test_representative_packed_and_specialized_lifecycle_scenarios",
        "test_packed.PackedForwardTraceTests."
        "test_trace_linked_occurrences_keep_event_local_autograd_provenance",
        "test_packed.PackedForwardTraceTests."
        "test_independent_chain_aggregate_occurrences_and_output_provenance",
        "test_packed.PackedForwardTraceTests."
        "test_frozen_active_formula_lane_does_not_inherit_trainable_lane_graph",
        "test_packed.PackedForwardTraceTests."
        "test_same_node_occurrences_keep_sequence_local_state_provenance",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_empty_initial_state_trace_metadata_matches_before_first_observe",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_unobserved_detached_input_state_is_carried_without_graph_pollution",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_cross_chunk_trace_state_roots_retain_event_local_vjps",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_differentiable_initial_state_and_cross_chunk_objectives_match",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_supported_vjp_cases_match_values_and_none_connectivity",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_prefill_default_detaches_cross_chunk_graph_and_false_retains_it",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_chunk_detach_cuts_only_the_cross_chunk_state_gradient",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_decode_default_detaches_and_explicit_false_preserves_state_graph",
        "test_packed.PackedChunkDecodeAndGradientTests."
        "test_each_node_event_logit_vjp_matches_none_connectivity",
        "test_packed.PackedForwardTraceTests."
        "test_sd_pre_state_replay_preserves_natural_route_at_fp32_boundaries",
        "test_packed.PackedForwardTraceTests."
        "test_upstream_node_compute_preserves_downstream_fp32_tie_route",
        "test_specialized.SpecializedEquivalenceTests."
        "test_single_layer_mlp_fp32_prefill_preserves_eager_boundary_route",
        "test_specialized.SpecializedEquivalenceTests."
        "test_decode_default_detaches_and_explicit_false_keeps_cross_call_vjp",
        "test_specialized.SpecializedEquivalenceTests."
        "test_prefill_preflight_failures_match_stable_envelopes",
        "test_specialized.SpecializedEquivalenceTests."
        "test_invalid_state_shape_and_owner_alias_match_stable_envelopes",
        "test_specialized.SpecializedEquivalenceTests."
        "test_late_empty_terminal_failure_preserves_entry_state",
    }
)
_CANDIDATE_IDENTITY_SHA256 = (
    "8497fccea52a958373ae5963c433a0f8420874005c88639ebd9e35d51fec6111"
)
_SUPPORT_PARTITION_SHA256 = (
    "49fb9c797f40546f29534bbaf1fac5c4b04669b4990239d4af22d3154fa4c703"
)
_DTYPES = ("float64", "float32")
_PACKED_IMPLEMENTATION_ID = "tide.generic-packed.torch.v1"
_IMPLEMENTATION_VARIANTS = (
    "token-major-eager-reference",
    _PACKED_IMPLEMENTATION_ID,
    "single-layer.v1",
    "hb-line.v1",
)


def _qualification_gaps() -> list[dict[str, str]]:
    """Return the explicit reasons this run cannot be called qualification."""

    return [
        {
            "id": "fixtures.self-contained-bundles",
            "detail": (
                "the 256 candidates are deterministic generated Plans, not "
                "frozen self-contained fixture/result bundles"
            ),
        },
        {
            "id": "gates.i00-c01",
            "detail": (
                "the 35-probe evidence-infrastructure gate and the complete "
                "independent C01 formula/exact-trace goldens are not run"
            ),
        },
        {
            "id": "vjp.finite-difference",
            "detail": (
                "the 64 marked VJP cases run, but the separately frozen 32 "
                "FP64 finite-difference checks are absent"
            ),
        },
        {
            "id": "optimizer.minimum-16",
            "detail": "the required 16 optimizer-step fixtures are absent",
        },
        {
            "id": "negative-and-scenarios",
            "detail": (
                "all 256 packed candidates and all 24 specialization-supported "
                "cases cover full prefill, both nonempty T=3 two-chunk splits, "
                "and token-by-token decode; absent are the frozen 104 negative "
                "containers and the qualification-required complete short/long "
                "chunk-partition, concurrency, rollback, and checkpoint/resume "
                "scenarios"
            ),
        },
        {
            "id": "capability.fresh-process-exact-source",
            "detail": (
                "tests execute in one runner process and do not publish all "
                "fresh-process capability artifacts; the runner does require "
                "its own run to start and end at one clean exact commit"
            ),
        },
        {
            "id": "platform.coverage",
            "detail": (
                "this CPU-only run does not establish separate x86_64, CUDA, "
                "or NPU capability cells"
            ),
        },
        {
            "id": "integration-and-performance",
            "detail": (
                "Qwen integration, short training, and formal packed "
                "performance qualification are outside this suite"
            ),
        },
    ]


@dataclasses.dataclass(frozen=True)
class _DiscoveredSuite:
    suite: unittest.TestSuite
    test_ids: tuple[str, ...]
    module_test_counts: Mapping[str, int]


def _iter_tests(suite: unittest.TestSuite) -> Iterator[unittest.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _check_preloaded_test_module(module_name: str, expected: Path) -> None:
    for imported_name in (module_name, f"tests.{module_name}"):
        module = sys.modules.get(imported_name)
        if module is not None:
            _require_module_file(module, imported_name, expected)


def _discover_repository_suite() -> _DiscoveredSuite:
    """Discover every required module, only from this checkout, with no gaps."""

    tests_root = (_REPOSITORY_ROOT / "tests").resolve()
    combined = unittest.TestSuite()
    all_ids: list[str] = []
    counts: dict[str, int] = {}
    for filename in _REQUIRED_TEST_MODULES:
        module_name = Path(filename).stem
        expected = tests_root / filename
        _check_preloaded_test_module(module_name, expected)
        discovered = unittest.defaultTestLoader.discover(
            str(tests_root), pattern=filename
        )
        imported = sys.modules.get(module_name)
        if imported is None:
            raise RuntimeError(f"test discovery did not import {module_name}")
        _require_module_file(imported, module_name, expected)

        tests = tuple(_iter_tests(discovered))
        if not tests:
            raise RuntimeError(f"required test module {filename} has zero tests")
        foreign = [
            test.id()
            for test in tests
            if test.__class__.__module__ != module_name
        ]
        if foreign:
            raise RuntimeError(
                f"required test module {filename} discovered foreign tests: "
                + ", ".join(foreign)
            )
        ids = tuple(test.id() for test in tests)
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"required test module {filename} has duplicate IDs")
        counts[filename] = len(ids)
        expected_count = _EXPECTED_MODULE_TEST_COUNTS[filename]
        if len(ids) != expected_count:
            raise RuntimeError(
                f"required test module {filename} changed test count: "
                f"expected {expected_count}, got {len(ids)}"
            )
        all_ids.extend(ids)
        combined.addTests(discovered)

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("required executor-equivalence suite has duplicate test IDs")
    missing_semantic_tests = sorted(
        _REQUIRED_SEMANTIC_TEST_IDS.difference(all_ids)
    )
    if missing_semantic_tests:
        raise RuntimeError(
            "required executor-equivalence semantic tests are missing: "
            + ", ".join(missing_semantic_tests)
        )
    return _DiscoveredSuite(combined, tuple(all_ids), counts)


def _receipt_test_module() -> Any:
    expected = (
        _REPOSITORY_ROOT / "tests" / "test_core_v1_executor_equivalence.py"
    ).resolve()
    modules = {
        id(module): module
        for name in (
            "test_core_v1_executor_equivalence",
            "tests.test_core_v1_executor_equivalence",
        )
        if (module := sys.modules.get(name)) is not None
    }
    if len(modules) != 1:
        raise RuntimeError(
            "executor-equivalence receipt module must be imported exactly once"
        )
    module = next(iter(modules.values()))
    _require_module_file(module, module.__name__, expected)
    return module


def _reset_repository_receipts() -> None:
    reset = getattr(_receipt_test_module(), "reset_execution_receipts", None)
    if not callable(reset):
        raise RuntimeError("executor-equivalence tests do not expose receipt reset")
    reset()


def _collect_repository_receipts() -> tuple[Mapping[str, Any], ...]:
    collect = getattr(_receipt_test_module(), "execution_receipts", None)
    if not callable(collect):
        raise RuntimeError("executor-equivalence tests do not expose receipts")
    receipts = collect()
    if not isinstance(receipts, tuple) or not all(
        isinstance(receipt, Mapping) for receipt in receipts
    ):
        raise RuntimeError("executor-equivalence receipt collection is malformed")
    return receipts


def _receipt_key_text(key: Sequence[str]) -> str:
    return json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))


def _validate_execution_receipts(
    corpus: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Close declared executor coverage against actual successful events."""

    errors: list[str] = []
    for index, receipt in enumerate(receipts):
        if receipt.get("sequence") != index:
            errors.append(f"receipt sequence mismatch at index {index}")
        if receipt.get("outcome") != "passed":
            errors.append(f"receipt {index} is not a passed event")

    support = corpus["support"]["executors"]
    executor_records = {
        _PACKED_IMPLEMENTATION_ID: support["packed"],
        "single-layer.v1": support["single_layer"],
        "hb-line.v1": support["hb"],
    }
    expected_forward: set[tuple[str, str, str, str]] = set()
    modes = (
        "full-prefill",
        "two-chunk-prefill",
        "token-by-token-decode",
    )
    for executor, record in executor_records.items():
        for case_id in record["accepted_case_ids"]:
            for dtype in _DTYPES:
                for mode in modes:
                    expected_forward.add((executor, case_id, dtype, mode))

    forward_rows = [
        receipt for receipt in receipts if receipt.get("kind") == "forward-cell"
    ]
    actual_forward: set[tuple[str, str, str, str]] = set()
    required_observables = {
        "output",
        "final-state",
        "balance-sufficient-statistics",
        "canonical-trace",
        "exact-route",
    }
    expected_mode_calls = {
        "full-prefill": 1,
        # Both nonempty boundaries of the fixed T=3 fixture are executed.
        "two-chunk-prefill": 4,
        "token-by-token-decode": 3,
    }
    for receipt in forward_rows:
        key = tuple(
            str(receipt.get(field))
            for field in ("executor", "case_id", "dtype", "mode")
        )
        if key in actual_forward:
            errors.append("duplicate forward receipt " + _receipt_key_text(key))
        actual_forward.add(key)
        executor, _, _, mode = key
        calls = expected_mode_calls.get(mode)
        expected_call_counts = (
            {"eager": calls, "packed": calls}
            if executor == _PACKED_IMPLEMENTATION_ID
            else {"eager": calls, "packed": calls, "specialized": calls}
        )
        if calls is None or receipt.get("call_counts") != expected_call_counts:
            errors.append("invalid call counts for " + _receipt_key_text(key))
        if set(receipt.get("observables", ())) != required_observables:
            errors.append("invalid observables for " + _receipt_key_text(key))
        expected_trace_scope = (
            "full-call"
            if mode == "full-prefill"
            else "each-call-and-canonical-merge"
        )
        if receipt.get("trace_scope") != expected_trace_scope:
            errors.append("invalid trace scope for " + _receipt_key_text(key))
        if mode == "two-chunk-prefill":
            if receipt.get("splits") != [1, 2]:
                errors.append(
                    "invalid short chunk splits for " + _receipt_key_text(key)
                )
        elif "splits" in receipt:
            errors.append("unexpected chunk splits for " + _receipt_key_text(key))

    missing_forward = sorted(expected_forward - actual_forward)
    extra_forward = sorted(actual_forward - expected_forward)
    if missing_forward:
        errors.append(
            f"missing {len(missing_forward)} forward cells: "
            + ", ".join(_receipt_key_text(key) for key in missing_forward)
        )
    if extra_forward:
        errors.append(
            f"unexpected {len(extra_forward)} forward cells: "
            + ", ".join(_receipt_key_text(key) for key in extra_forward)
        )

    expected_vjp_groups: set[tuple[str, str, str]] = set()
    for executor, record in executor_records.items():
        for case_id in record["vjp_test_case_ids"]:
            for dtype in _DTYPES:
                expected_vjp_groups.add((executor, case_id, dtype))

    objective_rows = [
        receipt for receipt in receipts if receipt.get("kind") == "vjp-objective"
    ]
    complete_rows = [
        receipt
        for receipt in receipts
        if receipt.get("kind") == "vjp-case-complete"
    ]
    objectives_by_group: dict[tuple[str, str, str], list[str]] = {}
    objective_families: Counter[str] = Counter()
    allowed_objective_families = {
        "output",
        "balance-loss",
        "final-state-component",
        "balance-region-soft-sum",
        "trace-region-event-logits",
        "trace-region-event-probabilities",
        "combined-output-balance-state",
    }
    for receipt in objective_rows:
        group = tuple(
            str(receipt.get(field))
            for field in ("executor", "case_id", "dtype")
        )
        objective_id = str(receipt.get("objective_id"))
        objectives_by_group.setdefault(group, []).append(objective_id)
        family = str(receipt.get("objective_family"))
        objective_families[family] += 1
        if family not in allowed_objective_families:
            errors.append(
                f"unknown VJP objective family {family!r} for "
                + _receipt_key_text(group)
            )
        if receipt.get("mode") != "full-prefill":
            errors.append("non-prefill VJP receipt " + _receipt_key_text(group))

    actual_vjp_groups: set[tuple[str, str, str]] = set()
    for receipt in complete_rows:
        group = tuple(
            str(receipt.get(field))
            for field in ("executor", "case_id", "dtype")
        )
        if group in actual_vjp_groups:
            errors.append("duplicate VJP completion " + _receipt_key_text(group))
        actual_vjp_groups.add(group)
        declared = receipt.get("objective_ids")
        if not isinstance(declared, list) or not all(
            isinstance(objective, str) for objective in declared
        ):
            errors.append("malformed objective list " + _receipt_key_text(group))
            continue
        observed = objectives_by_group.get(group, [])
        if declared != observed:
            errors.append("VJP objective list mismatch " + _receipt_key_text(group))
        if len(declared) != len(set(declared)):
            errors.append("duplicate VJP objective " + _receipt_key_text(group))
        if group[0] == _PACKED_IMPLEMENTATION_ID:
            required = {"output", "combined", "output.repeat"}
            if not required.issubset(declared) or not declared or (
                declared[-1] != "output.repeat"
            ):
                errors.append(
                    "packed VJP root policy mismatch " + _receipt_key_text(group)
                )
        elif declared != ["combined-output-balance-state"]:
            errors.append(
                "specialized VJP root policy mismatch "
                + _receipt_key_text(group)
            )

    missing_vjp = sorted(expected_vjp_groups - actual_vjp_groups)
    extra_vjp = sorted(actual_vjp_groups - expected_vjp_groups)
    if missing_vjp:
        errors.append(
            f"missing {len(missing_vjp)} VJP groups: "
            + ", ".join(_receipt_key_text(key) for key in missing_vjp)
        )
    if extra_vjp:
        errors.append(
            f"unexpected {len(extra_vjp)} VJP groups: "
            + ", ".join(_receipt_key_text(key) for key in extra_vjp)
        )
    orphan_objective_groups = sorted(
        set(objectives_by_group).difference(actual_vjp_groups)
    )
    if orphan_objective_groups:
        errors.append(
            "VJP objectives lack completion records: "
            + ", ".join(
                _receipt_key_text(key) for key in orphan_objective_groups
            )
        )

    required_packed_families = allowed_objective_families
    missing_families = sorted(
        required_packed_families.difference(objective_families)
    )
    if missing_families:
        errors.append("missing VJP objective families: " + ", ".join(missing_families))

    case_by_ordinal = {
        int(case["ordinal"]): str(case["case_id"])
        for case in corpus["cases"]
    }
    expected_lifecycle = {
        (_PACKED_IMPLEMENTATION_ID, case_by_ordinal[176], "float64"),
        (_PACKED_IMPLEMENTATION_ID, case_by_ordinal[19], "float64"),
        ("single-layer.v1", case_by_ordinal[23], "float64"),
        ("hb-line.v1", case_by_ordinal[144], "float64"),
    }
    lifecycle_rows = [
        receipt
        for receipt in receipts
        if receipt.get("kind") == "lifecycle-scenario"
    ]
    actual_lifecycle = {
        tuple(
            str(receipt.get(field))
            for field in ("executor", "case_id", "dtype")
        )
        for receipt in lifecycle_rows
    }
    if len(actual_lifecycle) != len(lifecycle_rows):
        errors.append("duplicate lifecycle receipt")
    if actual_lifecycle != expected_lifecycle:
        errors.append("lifecycle receipt set does not match the frozen scenarios")

    known_kinds = {
        "forward-cell",
        "vjp-objective",
        "vjp-case-complete",
        "lifecycle-scenario",
    }
    unknown_kinds = sorted(
        {
            str(receipt.get("kind"))
            for receipt in receipts
            if receipt.get("kind") not in known_kinds
        }
    )
    if unknown_kinds:
        errors.append("unknown receipt kinds: " + ", ".join(unknown_kinds))

    forward_counts = Counter(key[0] for key in actual_forward)
    vjp_counts = Counter(key[0] for key in actual_vjp_groups)
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "derived_coverage": {
            "forward_cells": len(actual_forward),
            "packed_forward_cells": forward_counts[_PACKED_IMPLEMENTATION_ID],
            "single_layer_forward_cells": forward_counts["single-layer.v1"],
            "hb_forward_cells": forward_counts["hb-line.v1"],
            "vjp_case_dtype_groups": len(actual_vjp_groups),
            "packed_vjp_case_dtype_groups": vjp_counts[
                _PACKED_IMPLEMENTATION_ID
            ],
            "single_layer_vjp_case_dtype_groups": vjp_counts[
                "single-layer.v1"
            ],
            "hb_vjp_case_dtype_groups": vjp_counts["hb-line.v1"],
            "vjp_objective_queries": len(objective_rows),
            "vjp_objective_families": dict(sorted(objective_families.items())),
            "lifecycle_scenarios": len(actual_lifecycle),
        },
        "expected": {
            "forward_cells": len(expected_forward),
            "vjp_case_dtype_groups": len(expected_vjp_groups),
            "lifecycle_scenarios": len(expected_lifecycle),
        },
    }


def _single_layer_reason_code(reason: str) -> str:
    if reason == "requires exactly one region":
        return "single-layer.topology.region-count"
    if reason == "requires a graph with no receiver edges":
        return "single-layer.topology.edges"
    if "sole region" in reason or "every receiver" in reason:
        return "single-layer.topology.flat-boundary"
    if "profile N" in reason:
        return "single-layer.profile"
    if "fixed K" in reason:
        return "single-layer.k"
    if "selector context" in reason or "context_dim" in reason:
        return "single-layer.context"
    if "selector history" in reason:
        return "single-layer.history"
    if "must be stateless" in reason:
        return "single-layer.stateful"
    if "not batchable" in reason:
        return "single-layer.selector-read"
    raise RuntimeError(f"unclassified single-layer support reason: {reason}")


def _hb_reason_code(reason: str) -> str:
    if "topology_kind" in reason:
        return "hb.topology.kind"
    if "Line" in reason or "deeper Line" in reason:
        return "hb.topology.line"
    if "non-fixed K" in reason:
        return "hb.k"
    if "selector context" in reason or "context_dim" in reason:
        return "hb.context"
    if "selector history" in reason:
        return "hb.history"
    raise RuntimeError(f"unclassified HB support reason: {reason}")


def _candidate_identity(cases: tuple[Any, ...]) -> str:
    identity = "\n".join(
        f"{case.case_id}\t{case.plan.canonical_hash()}\t"
        f"{case.parameter_seed}\t{case.input_seed}\t{int(case.vjp)}"
        for case in cases
    ).encode("ascii")
    return hashlib.sha256(identity).hexdigest()


def _support_record(
    name: str,
    implementation_variant: str,
    cases: tuple[Any, ...],
    reason_codes: tuple[tuple[str, ...], ...],
    *,
    support_id: str,
    vjp_policy: str,
) -> dict[str, Any]:
    accepted = [
        case.case_id
        for case, reasons in zip(cases, reason_codes)
        if not reasons
    ]
    rejected = [
        {"case_id": case.case_id, "reason_codes": list(reasons)}
        for case, reasons in zip(cases, reason_codes)
        if reasons
    ]
    accepted_set = set(accepted)
    marked_vjp = [
        case.case_id
        for case in cases
        if case.vjp and case.case_id in accepted_set
    ]
    if vjp_policy == "marked-corpus-subset":
        vjp_cases = marked_vjp
    elif vjp_policy == "all-statically-accepted":
        vjp_cases = accepted
    else:
        raise ValueError(f"unknown VJP policy {vjp_policy!r}")
    return {
        "name": name,
        "implementation_variant": implementation_variant,
        "support_id": support_id,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "vjp_policy": vjp_policy,
        "vjp_test_case_count": len(vjp_cases),
        "vjp_test_case_ids": vjp_cases,
        "marked_vjp_intersection_count": len(marked_vjp),
        "marked_vjp_intersection_case_ids": marked_vjp,
        "accepted_case_ids": accepted,
        "rejected_cases": rejected,
    }


def _corpus_record() -> tuple[dict[str, Any], str]:
    """Build and preflight the static candidate/support evidence artifact."""

    from tide import generators as generators_module
    from tide import equivalence as equivalence_module
    from tide import packed as packed_module
    from tide import plan as plan_module
    from tide import specialized as specialized_module
    from tide.generators import (
        CORE_V1_CANDIDATE_CORPUS_SEED,
        CORE_V1_CANDIDATE_CORPUS_SIZE,
        CORE_V1_CANDIDATE_VJP_SIZE,
        generate_core_v1_candidate_corpus,
    )
    from tide.equivalence import (
        CPU_FLOAT64_TOLERANCE,
        SAME_BACKEND_FLOAT32_TOLERANCE,
    )
    from tide.packed import PACKED_EXECUTOR_ID, inspect_packed_support
    from tide.plan import bind_dtypes
    from tide.specialized import (
        HB_LINE_V1,
        SINGLE_LAYER_V1,
        hb_line_v1_support,
        single_layer_v1_support,
    )

    if PACKED_EXECUTOR_ID != _PACKED_IMPLEMENTATION_ID:
        raise RuntimeError(
            "runner packed implementation identity disagrees with the executor: "
            f"{PACKED_EXECUTOR_ID!r}"
        )

    package_root = (_REPOSITORY_ROOT / "src" / "tide").resolve()
    _require_module_file(
        generators_module,
        "tide.generators",
        package_root / "generators.py",
    )
    _require_module_file(
        equivalence_module,
        "tide.equivalence",
        package_root / "equivalence.py",
    )
    _require_module_file(
        plan_module,
        "tide.plan",
        package_root / "plan.py",
    )
    _require_module_file(
        packed_module,
        "tide.packed",
        package_root / "packed.py",
    )
    _require_module_file(
        specialized_module,
        "tide.specialized",
        package_root / "specialized.py",
    )

    cases = generate_core_v1_candidate_corpus()
    if len(cases) != CORE_V1_CANDIDATE_CORPUS_SIZE or len(cases) != 256:
        raise RuntimeError("core-v1 candidate corpus must contain exactly 256 Plans")
    if sum(case.vjp for case in cases) != CORE_V1_CANDIDATE_VJP_SIZE or sum(
        case.vjp for case in cases
    ) != 64:
        raise RuntimeError("core-v1 candidate corpus must mark exactly 64 VJP Plans")
    if tuple(case.ordinal for case in cases) != tuple(range(256)):
        raise RuntimeError("core-v1 candidate ordinals must be the fixed range 0..255")

    candidate_identity_sha256 = _candidate_identity(cases)
    if candidate_identity_sha256 != _CANDIDATE_IDENTITY_SHA256:
        raise RuntimeError(
            "core-v1 candidate identity does not match the frozen development "
            f"digest: {candidate_identity_sha256}"
        )

    packed_codes: list[tuple[str, ...]] = []
    single_layer_codes: list[tuple[str, ...]] = []
    hb_codes: list[tuple[str, ...]] = []
    partition_rows: list[dict[str, Any]] = []
    partition_lines: list[str] = []
    for case in cases:
        packed = inspect_packed_support(case.plan)
        single_layer = single_layer_v1_support(case.plan)
        hb = hb_line_v1_support(case.plan)
        p_codes = tuple(sorted({issue.code for issue in packed.issues}))
        s_codes = tuple(
            sorted(
                {
                    _single_layer_reason_code(reason)
                    for reason in single_layer.reasons
                }
            )
        )
        h_codes = tuple(sorted({_hb_reason_code(reason) for reason in hb.reasons}))
        if packed.accepted == bool(p_codes):
            raise RuntimeError("packed support decision disagrees with its issues")
        if single_layer.supported == bool(s_codes):
            raise RuntimeError(
                "single-layer support decision disagrees with its reasons"
            )
        if hb.supported == bool(h_codes):
            raise RuntimeError("HB support decision disagrees with its reasons")
        packed_codes.append(p_codes)
        single_layer_codes.append(s_codes)
        hb_codes.append(h_codes)
        partition_rows.append(
            {
                "case_id": case.case_id,
                "packed_reason_codes": list(p_codes),
                "single_layer_reason_codes": list(s_codes),
                "hb_reason_codes": list(h_codes),
            }
        )
        partition_lines.append(
            "\t".join(
                (
                    case.case_id,
                    "P:" + ",".join(p_codes),
                    "S:" + ",".join(s_codes),
                    "H:" + ",".join(h_codes),
                )
            )
        )

    partition_sha256 = hashlib.sha256(
        "\n".join(partition_lines).encode("utf-8")
    ).hexdigest()
    if partition_sha256 != _SUPPORT_PARTITION_SHA256:
        raise RuntimeError(
            "executor support partition does not match the frozen development "
            f"digest: {partition_sha256}"
        )

    support = {
        "partition_schema": "tide.core-v1-executor-support-partition.v1",
        "partition_sha256": partition_sha256,
        "partition_digest_encoding": (
            "UTF-8 rows in ordinal order, joined by LF without a trailing LF; "
            "each row is case_id<TAB>P:codes<TAB>S:codes<TAB>H:codes"
        ),
        "partition_rows": partition_rows,
        "executors": {
            "packed": _support_record(
                "generic packed",
                _PACKED_IMPLEMENTATION_ID,
                cases,
                tuple(packed_codes),
                support_id=PACKED_EXECUTOR_ID,
                vjp_policy="marked-corpus-subset",
            ),
            "single_layer": _support_record(
                "single-layer topology specialization",
                SINGLE_LAYER_V1,
                cases,
                tuple(single_layer_codes),
                support_id=SINGLE_LAYER_V1,
                vjp_policy="all-statically-accepted",
            ),
            "hb": _support_record(
                "HB Line topology specialization",
                HB_LINE_V1,
                cases,
                tuple(hb_codes),
                support_id=HB_LINE_V1,
                vjp_policy="all-statically-accepted",
            ),
        },
    }
    expected_counts = {
        "packed": (256, 0, 64, 64),
        "single_layer": (8, 248, 8, 0),
        "hb": (16, 240, 16, 4),
    }
    for name, expected in expected_counts.items():
        executor = support["executors"][name]
        actual = (
            executor["accepted_count"],
            executor["rejected_count"],
            executor["vjp_test_case_count"],
            executor["marked_vjp_intersection_count"],
        )
        if actual != expected:
            raise RuntimeError(
                f"{name} support counts changed: expected {expected}, got {actual}"
            )

    features = Counter(feature for case in cases for feature in case.features)
    motifs = Counter(case.motif for case in cases)
    record = {
        "schema_version": 1,
        "kind": "tide.core-v1-executor-equivalence-candidate.v1",
        "qualification": False,
        "qualification_gaps": _qualification_gaps(),
        "seed": CORE_V1_CANDIDATE_CORPUS_SEED,
        "legal_count": len(cases),
        "vjp_plan_count": sum(case.vjp for case in cases),
        "candidate_identity_schema": (
            "case-id/logical-plan-hash/parameter-seed/input-seed/vjp.v1"
        ),
        "candidate_identity_sha256": candidate_identity_sha256,
        "dtypes": list(_DTYPES),
        "comparison_contract": {
            "float64": {
                "name": "T64",
                "atol": CPU_FLOAT64_TOLERANCE.atol,
                "rtol": CPU_FLOAT64_TOLERANCE.rtol,
            },
            "float32": {
                "name": "T32",
                "atol": SAME_BACKEND_FLOAT32_TOLERANCE.atol,
                "rtol": SAME_BACKEND_FLOAT32_TOLERANCE.rtol,
            },
            "finite_required": True,
            "discrete_values": "exact",
            "live_autograd_metadata": {
                "requires_grad": "exact-by-public-tensor-occurrence",
                "is_leaf": "not-compared",
                "grad_fn": "not-compared",
            },
        },
        "implementation_variants": list(_IMPLEMENTATION_VARIANTS),
        "equivalence_matrix": {
            "packed": {
                "forward_case_count": 256,
                "vjp_case_count": 64,
                "dtypes": list(_DTYPES),
                "forward_execution_modes": [
                    "full-prefill",
                    "two-chunk-prefill",
                    "token-by-token-decode",
                ],
                "forward_observables": [
                    "output",
                    "final-state",
                    "balance-sufficient-statistics",
                    "canonical-trace",
                    "exact-route",
                ],
                "trace_execution_modes": [
                    "full-prefill",
                    "each-two-chunk-prefill-call",
                    "each-token-by-token-decode-call",
                ],
                "vjp_execution_modes": ["full-prefill"],
                "vjp_objective_policy": "isolated-public-roots-and-combined",
                "vjp_objectives": [
                    "output-with-frozen-cotangent",
                    "balance-loss-when-differentiable",
                    "each-final-state-tensor-component-with-frozen-cotangent",
                    "each-region-soft-sum-with-frozen-cotangent",
                    "each-trace-region-event-logits-with-frozen-cotangent",
                    "each-trace-region-event-probabilities-with-frozen-cotangent",
                    "combined-output-balance-state",
                    "output-repeat-after-root-switches",
                ],
                "vjp_observables": [
                    "hidden",
                    "every-named-parameter",
                    "none-vs-tensor-connectivity",
                    "finite-values",
                ],
                "compares_to": ["token-major-eager-reference"],
            },
            "single_layer": {
                "forward_case_count": 8,
                "vjp_case_count": 8,
                "dtypes": list(_DTYPES),
                "forward_execution_modes": [
                    "full-prefill",
                    "two-chunk-prefill",
                    "token-by-token-decode",
                ],
                "vjp_execution_modes": ["full-prefill"],
                "vjp_objective_policy": "combined-output-balance-state",
                "vjp_observables": [
                    "hidden",
                    "every-named-parameter",
                    "none-vs-tensor-connectivity",
                ],
                "compares_to": [
                    "token-major-eager-reference",
                    _PACKED_IMPLEMENTATION_ID,
                ],
            },
            "hb": {
                "forward_case_count": 16,
                "vjp_case_count": 16,
                "dtypes": list(_DTYPES),
                "forward_execution_modes": [
                    "full-prefill",
                    "two-chunk-prefill",
                    "token-by-token-decode",
                ],
                "vjp_execution_modes": ["full-prefill"],
                "vjp_objective_policy": "combined-output-balance-state",
                "vjp_observables": [
                    "hidden",
                    "every-named-parameter",
                    "none-vs-tensor-connectivity",
                ],
                "compares_to": [
                    "token-major-eager-reference",
                    _PACKED_IMPLEMENTATION_ID,
                ],
            },
        },
        "required_test_modules": list(_REQUIRED_TEST_MODULES),
        "motifs": dict(sorted(motifs.items())),
        "features": dict(sorted(features.items())),
        "support": support,
        "cases": [
            {
                "case_id": case.case_id,
                "ordinal": case.ordinal,
                "motif": case.motif,
                "generation_seed": case.generation_seed,
                "parameter_seed": case.parameter_seed,
                "input_seed": case.input_seed,
                "vjp": case.vjp,
                "features": sorted(case.features),
                "logical_plan_hash": case.plan.canonical_hash(),
                "typed_plan_hashes": {
                    dtype: bind_dtypes(
                        case.plan,
                        hidden=dtype,
                        parameter=dtype,
                        state=dtype,
                        readout=dtype,
                    ).typed_hash()
                    for dtype in _DTYPES
                },
            }
            for case in cases
        ],
    }
    content = _json_bytes(record)
    return record, hashlib.sha256(content).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the required CPU executor-equivalence tests and write "
            "run.json, metrics.jsonl, stdout.log, summary.json, and "
            "the corpus and execution-receipt artifacts."
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

    # Torch and the test graph are imported only after the initial source
    # observation.  The endpoint snapshot detects edits racing the workload.
    import torch

    cpu_runtime = _establish_cpu_runtime(torch)
    started_at = _utc_now()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-executor-equivalence-"
        + corpus["candidate_identity_sha256"][:8]
        + "-"
        + uuid.uuid4().hex[:8]
    )
    resolved_argv = [
        sys.executable,
        "scripts/run_executor_equivalence.py",
        *parsed_argv,
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "project": "tide-settlegraph",
        "name": "core-v1-cpu-executor-equivalence-development",
        "parent_run_id": None,
        "status": "running",
        "created_at": started_at,
        "started_at": started_at,
        "ended_at": None,
        "qualification": False,
        "qualification_gaps": corpus["qualification_gaps"],
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
        "command": {"argv": resolved_argv, "working_directory": "."},
        "inputs": {
            "corpus": {
                "kind": corpus["kind"],
                "seed": corpus["seed"],
                "artifact_sha256": corpus_hash,
                "candidate_identity_sha256": corpus[
                    "candidate_identity_sha256"
                ],
                "support_partition_sha256": corpus["support"][
                    "partition_sha256"
                ],
            }
        },
        "runtime": {
            "host_arch": platform.machine(),
            "accelerator_model": None,
            "resolved_device": "cpu",
            "resolution_reason": "explicit:cpu-executor-equivalence",
            **cpu_runtime,
            "dtype": ",".join(_DTYPES),
            "dtypes": list(_DTYPES),
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
            "implementation_variants": list(_IMPLEMENTATION_VARIANTS),
        },
        "experiment": {
            "class": "controlled-development-comparison",
            "hypothesis": (
                "generic packed prefill/decode and every statically applicable "
                "topology specialization agree with the token-major eager "
                "reference on the fixed 256-candidate CPU corpus"
            ),
            "global_step_semantics": "one completed required unittest suite",
            "primary_metric": "validation/tests_failed",
            "stop_condition": "one bounded suite execution",
            "config": {
                "legal_plan_count": corpus["legal_count"],
                "vjp_plan_count": corpus["vjp_plan_count"],
                "dtypes": corpus["dtypes"],
                "comparison_contract": corpus["comparison_contract"],
                "equivalence_matrix": corpus["equivalence_matrix"],
                "implementation_variants": corpus[
                    "implementation_variants"
                ],
                "required_test_modules": list(_REQUIRED_TEST_MODULES),
                "qualification": False,
            },
            "suite": None,
        },
        "tracking": {
            "mode": "off",
            "backend": None,
            "project": "tide-settlegraph",
            "run_name": "core-v1-cpu-executor-equivalence-development",
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
            "execution_receipt": "artifacts/execution-receipt.json",
        },
    }

    # Exclusive creation is intentionally outside the recovery block: an
    # existing target is owned by another run and must remain byte-for-byte
    # untouched.
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage = "setup"
    source_after: dict[str, Any] | None = None
    source_changed: bool | None = None
    receipts: tuple[Mapping[str, Any], ...] = ()
    receipt_validation: dict[str, Any] | None = None
    receipt_hash: str | None = None
    try:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir()
        _atomic_json(artifacts_dir / "corpus.json", corpus)
        _create_empty_file(run_dir / "stdout.log")
        _create_empty_file(run_dir / "metrics.jsonl")
        _atomic_json(run_dir / "run.json", manifest)

        stage = "workload"
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            discovery = _discover_repository_suite()
            manifest["experiment"]["suite"] = {
                "required_modules": list(_REQUIRED_TEST_MODULES),
                "module_test_counts": dict(discovery.module_test_counts),
                "discovered_test_count": len(discovery.test_ids),
                "test_ids": list(discovery.test_ids),
            }
            _reset_repository_receipts()
            _atomic_json(run_dir / "run.json", manifest)
            result = unittest.TextTestRunner(
                stream=_Tee(sys.stdout, log), verbosity=args.verbosity
            ).run(discovery.suite)
            _verify_cpu_runtime(torch)
            receipts = _collect_repository_receipts()

        stage = "finalize"
        _fsync_file(run_dir / "stdout.log")
        receipt_validation = _validate_execution_receipts(corpus, receipts)
        receipt_record = {
            "schema_version": 1,
            "kind": "tide.executor-equivalence-execution-receipt.v1",
            "run_id": run_id,
            "candidate_identity_sha256": corpus[
                "candidate_identity_sha256"
            ],
            "support_partition_sha256": corpus["support"][
                "partition_sha256"
            ],
            "events": [dict(receipt) for receipt in receipts],
            "derived_coverage": receipt_validation["derived_coverage"],
            "validation": {
                "status": receipt_validation["status"],
                "errors": receipt_validation["errors"],
                "expected": receipt_validation["expected"],
            },
        }
        receipt_hash = hashlib.sha256(_json_bytes(receipt_record)).hexdigest()
        _atomic_json(artifacts_dir / "execution-receipt.json", receipt_record)
        manifest["evidence"] = {
            "execution_receipt": {
                "path": "artifacts/execution-receipt.json",
                "sha256": receipt_hash,
                "validation_status": receipt_validation["status"],
            }
        }
        elapsed = time.monotonic() - started
        source_after = _source_snapshot(excluded_run_dir=excluded_run_dir)
        source_changed = _record_final_source(manifest, source_after)
        exact_commit = bool(manifest["source"]["exact_commit"])
        (
            tests_successful,
            failed,
            expected_failures,
            unexpected_successes,
        ) = _required_suite_outcome(result)
        discovered_count = len(discovery.test_ids)
        suite_complete = result.testsRun == discovered_count
        receipt_complete = receipt_validation["status"] == "passed"
        status = (
            "completed"
            if (
                tests_successful
                and suite_complete
                and receipt_complete
                and not source_changed
                and exact_commit
            )
            else "failed"
        )
        ended_at = _utc_now()
        support = corpus["support"]["executors"]
        derived = receipt_validation["derived_coverage"]
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "sequence": 0,
            "timestamp": ended_at,
            "step": 1,
            "elapsed_seconds": elapsed,
            "metrics": {
                "validation/tests_run": result.testsRun,
                "validation/tests_discovered": discovered_count,
                "validation/suite_complete": int(suite_complete),
                "validation/tests_failed": failed,
                "validation/tests_skipped": len(result.skipped),
                "validation/tests_expected_failures": expected_failures,
                "validation/tests_unexpected_successes": unexpected_successes,
                "validation/execution_receipt_complete": int(
                    receipt_complete
                ),
                "support/legal_plans": corpus["legal_count"],
                "support/vjp_plans": corpus["vjp_plan_count"],
                "support/dtypes": len(corpus["dtypes"]),
                "support/packed_accepted": support["packed"][
                    "accepted_count"
                ],
                "support/single_layer_accepted": support["single_layer"][
                    "accepted_count"
                ],
                "support/hb_accepted": support["hb"]["accepted_count"],
                "support/packed_vjp_cases": support["packed"][
                    "vjp_test_case_count"
                ],
                "support/single_layer_vjp_cases": support["single_layer"][
                    "vjp_test_case_count"
                ],
                "support/hb_vjp_cases": support["hb"][
                    "vjp_test_case_count"
                ],
                "coverage/forward_cells": derived["forward_cells"],
                "coverage/packed_forward_cells": derived[
                    "packed_forward_cells"
                ],
                "coverage/single_layer_forward_cells": derived[
                    "single_layer_forward_cells"
                ],
                "coverage/hb_forward_cells": derived["hb_forward_cells"],
                "coverage/vjp_case_dtype_groups": derived[
                    "vjp_case_dtype_groups"
                ],
                "coverage/packed_vjp_case_dtype_groups": derived[
                    "packed_vjp_case_dtype_groups"
                ],
                "coverage/single_layer_vjp_case_dtype_groups": derived[
                    "single_layer_vjp_case_dtype_groups"
                ],
                "coverage/hb_vjp_case_dtype_groups": derived[
                    "hb_vjp_case_dtype_groups"
                ],
                "coverage/vjp_objective_queries": derived[
                    "vjp_objective_queries"
                ],
                "coverage/lifecycle_scenarios": derived[
                    "lifecycle_scenarios"
                ],
                "qualification/passed": 0,
            },
            "context": {
                "candidate_identity_sha256": corpus[
                    "candidate_identity_sha256"
                ],
                "support_partition_sha256": corpus["support"][
                    "partition_sha256"
                ],
                "qualification": False,
                "source_changed_during_run": source_changed,
                "exact_commit": exact_commit,
                "execution_receipt_sha256": receipt_hash,
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
            "candidate_identity_sha256": corpus[
                "candidate_identity_sha256"
            ],
            "support_partition_sha256": corpus["support"][
                "partition_sha256"
            ],
            "execution_receipt_sha256": receipt_hash,
            "execution_receipt_validation": receipt_validation,
            "conclusion": (
                "controlled CPU executor-equivalence suite passed; "
                "qualification remains false"
                if status == "completed"
                else (
                    "source changed during the controlled CPU "
                    "executor-equivalence run"
                    if source_changed
                    else (
                        "controlled run did not start and end at a clean exact commit"
                        if not exact_commit
                        else (
                            "required unittest discovery/execution did not close"
                            if not suite_complete
                            else (
                                "execution receipt did not close declared coverage"
                                if not receipt_complete
                                else "controlled CPU executor-equivalence suite failed"
                            )
                        )
                    )
                )
            ),
            "qualification": False,
            "qualification_gaps": corpus["qualification_gaps"],
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
            "candidate_identity_sha256": corpus[
                "candidate_identity_sha256"
            ],
            "support_partition_sha256": corpus["support"][
                "partition_sha256"
            ],
            "conclusion": (
                "controlled CPU executor-equivalence run interrupted"
                if status == "interrupted"
                else f"controlled CPU executor-equivalence run failed during {stage}"
            ),
            "qualification": False,
            "qualification_gaps": corpus["qualification_gaps"],
            "source_changed_during_run": source_changed,
            "error": {"stage": stage, "type": type(exc).__name__},
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
                "candidate_identity_sha256": corpus[
                    "candidate_identity_sha256"
                ],
                "support_partition_sha256": corpus["support"][
                    "partition_sha256"
                ],
                "tests_run": result.testsRun,
                "tests_failed": failed,
                "qualification": False,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
