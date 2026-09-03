"""Persistent executor checks over the 256-slot core-v1 candidate corpus."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import unittest
from collections import Counter
from typing import Any, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor

from tide.engine import (
    BalanceStats,
    ExecutionContractError,
    ExecutionResult,
    ExecutionTrace,
    SettleGraph,
    StateStore,
    _build_trace,
)
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    Tolerance,
    compare_nested,
    validate_trace_invariants,
)
from tide.generators import (
    CORE_V1_CANDIDATE_CORPUS_SIZE,
    PlanCorpusCase,
    generate_core_v1_candidate_corpus,
)
from tide.ops import AttentionState
from tide.packed import (
    PACKED_EXECUTOR_ID,
    PackedSettleGraph,
    PackedSupportReport,
    inspect_packed_support,
)
from tide.plan import bind_dtypes
from tide.specialized import (
    HB_LINE_V1,
    SINGLE_LAYER_V1,
    SpecializationSupport,
    SpecializedExecutor,
    hb_line_v1_support,
    single_layer_v1_support,
)


_SEQUENCE_IDS = ("candidate.sequence.a", "candidate.sequence.b")
_EXECUTION_MASK = torch.tensor(
    [[True, True, True], [True, True, False]], dtype=torch.bool
)
_TOKEN_POSITIONS = torch.tensor([[0, 1, 2], [0, 1, 99]], dtype=torch.int64)
_LM_TARGET_MASK = torch.tensor(
    [[False, True, True], [False, True, False]], dtype=torch.bool
)
_ROUTING_STATS_MASK = torch.tensor(
    [[True, False, True], [True, True, False]], dtype=torch.bool
)
_EXECUTED_TOKENS = (
    ("candidate.sequence.a", 0),
    ("candidate.sequence.a", 1),
    ("candidate.sequence.a", 2),
    ("candidate.sequence.b", 0),
    ("candidate.sequence.b", 1),
)


# The controlled runner snapshots these records after the suite.  Ordinary
# unittest invocations may ignore them; keeping collection in memory avoids
# coupling correctness tests to a filesystem or runner process layout.
_EXECUTION_RECEIPTS: list[dict[str, Any]] = []


def reset_execution_receipts() -> None:
    _EXECUTION_RECEIPTS.clear()


def execution_receipts() -> Tuple[Mapping[str, Any], ...]:
    return tuple(dict(receipt) for receipt in _EXECUTION_RECEIPTS)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float32:
        return "float32"
    raise AssertionError(f"unrecordable executor-equivalence dtype {dtype}")


def _record_execution_receipt(**receipt: Any) -> None:
    source_test_id = receipt.get("source_test_id")
    if isinstance(source_test_id, str) and not source_test_id.startswith(
        "test_core_v1_executor_equivalence."
    ):
        receipt["source_test_id"] = (
            "test_core_v1_executor_equivalence." + source_test_id
        )
    _EXECUTION_RECEIPTS.append(
        {
            "sequence": len(_EXECUTION_RECEIPTS),
            "outcome": "passed",
            **receipt,
        }
    )


def _objective_family(label: str) -> str:
    if label in {"output", "output.repeat"}:
        return "output"
    if label == "balance":
        return "balance-loss"
    if label.startswith("final-state:"):
        return "final-state-component"
    if label.startswith("balance-region:"):
        return "balance-region-soft-sum"
    if label.startswith("trace.region-logits:"):
        return "trace-region-event-logits"
    if label.startswith("trace.region-probabilities:"):
        return "trace-region-event-probabilities"
    if label == "combined":
        return "combined-output-balance-state"
    raise AssertionError(f"unclassified executor-equivalence objective {label}")


@dataclasses.dataclass(frozen=True)
class _CaseSupport:
    case: PlanCorpusCase
    packed: PackedSupportReport
    single_layer: SpecializationSupport
    hb: SpecializationSupport
    packed_reason_codes: Tuple[str, ...]
    single_layer_reason_codes: Tuple[str, ...]
    hb_reason_codes: Tuple[str, ...]


@functools.lru_cache(maxsize=1)
def _corpus() -> Tuple[PlanCorpusCase, ...]:
    return generate_core_v1_candidate_corpus()


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
    raise AssertionError(f"unclassified single-layer support reason: {reason}")


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
    raise AssertionError(f"unclassified HB support reason: {reason}")


@functools.lru_cache(maxsize=1)
def _support_partition() -> Tuple[_CaseSupport, ...]:
    records = []
    for case in _corpus():
        packed = inspect_packed_support(case.plan)
        single_layer = single_layer_v1_support(case.plan)
        hb = hb_line_v1_support(case.plan)
        records.append(
            _CaseSupport(
                case=case,
                packed=packed,
                single_layer=single_layer,
                hb=hb,
                packed_reason_codes=tuple(
                    sorted({issue.code for issue in packed.issues})
                ),
                single_layer_reason_codes=tuple(
                    sorted(
                        {
                            _single_layer_reason_code(reason)
                            for reason in single_layer.reasons
                        }
                    )
                ),
                hb_reason_codes=tuple(
                    sorted({_hb_reason_code(reason) for reason in hb.reasons})
                ),
            )
        )
    return tuple(records)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float32:
        return "float32"
    raise AssertionError(dtype)


def _tolerance(dtype: torch.dtype) -> Tolerance:
    return (
        CPU_FLOAT64_TOLERANCE
        if dtype == torch.float64
        else SAME_BACKEND_FLOAT32_TOLERANCE
    )


def _model(case: PlanCorpusCase, dtype: torch.dtype) -> SettleGraph:
    name = _dtype_name(dtype)
    typed = bind_dtypes(
        case.plan,
        hidden=name,
        parameter=name,
        state=name,
        readout=name,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.parameter_seed)
        model = SettleGraph(typed).to(device="cpu", dtype=dtype)
    model.eval()
    return model


def _hidden(case: PlanCorpusCase, dtype: torch.dtype, length: int = 3) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed)
    return torch.randn(
        (2, length, case.plan.d_model),
        generator=generator,
        dtype=torch.float64,
    ).to(dtype=dtype)


def _call_arguments(*, record_trace: bool = False) -> Mapping[str, Any]:
    return {
        "execution_mask": _EXECUTION_MASK,
        "sequence_ids": _SEQUENCE_IDS,
        "token_positions": _TOKEN_POSITIONS,
        "lm_target_mask": _LM_TARGET_MASK,
        "routing_stats_mask": _ROUTING_STATS_MASK,
        "detach_at_end": False,
        "record_trace": record_trace,
    }


def _merge_traces(
    executor: Any, results: Iterable[ExecutionResult]
) -> ExecutionTrace:
    traces = []
    for result in results:
        assert result.trace is not None
        traces.append(result.trace)
    model = executor if isinstance(executor, SettleGraph) else executor.model
    return _build_trace(
        model.plan,
        tuple(event for trace in traces for event in trace.node_events),
        tuple(event for trace in traces for event in trace.edge_events),
        tuple(event for trace in traces for event in trace.boundary_events),
        tuple(event for trace in traces for event in trace.region_events),
        tuple(event for trace in traces for event in trace.state_writes),
        tuple(event for trace in traces for event in trace.output_events),
    )


def _run_two_chunk_prefill(
    executor: Any,
    hidden: Tensor,
    split: int,
) -> Tuple[ExecutionResult, ExecutionResult, ExecutionResult]:
    first = executor.prefill(
        hidden[:, :split],
        _EXECUTION_MASK[:, :split],
        _SEQUENCE_IDS,
        _TOKEN_POSITIONS[:, :split],
        lm_target_mask=_LM_TARGET_MASK[:, :split],
        routing_stats_mask=_ROUTING_STATS_MASK[:, :split],
        record_trace=True,
    )
    second = executor.prefill(
        hidden[:, split:],
        _EXECUTION_MASK[:, split:],
        _SEQUENCE_IDS,
        _TOKEN_POSITIONS[:, split:],
        state=first.state,
        lm_target_mask=_LM_TARGET_MASK[:, split:],
        routing_stats_mask=_ROUTING_STATS_MASK[:, split:],
        record_trace=True,
    )
    combined = ExecutionResult(
        torch.cat((first.output, second.output), dim=1),
        second.state,
        first.balance_stats.merge(second.balance_stats),
        _merge_traces(executor, (first, second)),
    )
    return first, second, combined


def _run_token_decode(
    executor: Any,
    hidden: Tensor,
) -> Tuple[Tuple[ExecutionResult, ...], ExecutionResult]:
    state: Optional[StateStore] = None
    steps = []
    outputs = []
    stats: Optional[BalanceStats] = None
    for token in range(hidden.shape[1]):
        method = (
            executor.interpret_token
            if isinstance(executor, SettleGraph)
            else executor.decode
        )
        step = method(
            hidden[:, token],
            _EXECUTION_MASK[:, token],
            _SEQUENCE_IDS,
            _TOKEN_POSITIONS[:, token],
            state=state,
            lm_target_mask=_LM_TARGET_MASK[:, token],
            routing_stats_mask=_ROUTING_STATS_MASK[:, token],
            record_trace=True,
        )
        steps.append(step)
        outputs.append(step.output)
        state = step.state
        stats = step.balance_stats if stats is None else stats.merge(step.balance_stats)
    assert state is not None and stats is not None
    return tuple(steps), ExecutionResult(
        torch.stack(outputs, dim=1),
        state,
        stats,
        _merge_traces(executor, steps),
    )


def _compare(
    reference: object,
    candidate: object,
    *,
    dtype: torch.dtype,
    path: str = "$",
) -> None:
    compare_nested(
        reference,
        candidate,
        tolerance=_tolerance(dtype),
        path=path,
    ).require_pass()


def _state_objective(store: StateStore, reference: Tensor) -> Tensor:
    result = reference.sum() * 0.0
    for key in sorted(store.values):
        value = store.values[key]
        if isinstance(value, Tensor):
            result = result + value.sum()
        elif isinstance(value, AttentionState):
            result = result + value.keys.sum() + value.values.sum()
    return result


def _gradient_record(model: SettleGraph, hidden: Tensor) -> Mapping[str, object]:
    return {
        "hidden": hidden.grad,
        "parameters": {
            name: parameter.grad for name, parameter in model.named_parameters()
        },
    }


def _backward_objective(
    case: PlanCorpusCase, result: ExecutionResult, hidden: Tensor
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed ^ 0x5A17)
    cotangent = torch.randn(
        result.output.shape,
        generator=generator,
        dtype=torch.float64,
    ).to(dtype=result.output.dtype)
    return (
        (result.output * cotangent).sum()
        + 0.07 * result.balance_loss
        + 0.03 * _state_objective(result.state, hidden)
    )


def _cotangent(case: PlanCorpusCase, label: str, index: int, value: Tensor) -> Tensor:
    digest = hashlib.sha256(
        f"{case.case_id}\0{label}\0{index}".encode("utf-8")
    ).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    return torch.randn(
        value.shape, generator=generator, dtype=torch.float64
    ).to(dtype=value.dtype)


def _cotangent_objective(
    case: PlanCorpusCase,
    label: str,
    values: Iterable[Tensor],
) -> Optional[Tensor]:
    objective: Optional[Tensor] = None
    for index, value in enumerate(values):
        if not value.requires_grad:
            continue
        term = (value * _cotangent(case, label, index, value)).sum()
        objective = term if objective is None else objective + term
    return objective


def _isolated_objectives(
    case: PlanCorpusCase, result: ExecutionResult
) -> Mapping[str, Tensor]:
    """Construct independent public-result roots with frozen cotangents."""

    objectives: dict[str, Tensor] = {}
    output = _cotangent_objective(case, "output", (result.output,))
    assert output is not None
    objectives["output"] = output

    if result.balance_loss.requires_grad:
        objectives["balance"] = result.balance_loss

    for key in sorted(result.state.values):
        value = result.state.values[key]
        if isinstance(value, Tensor):
            components = (("tensor", value),)
        elif isinstance(value, AttentionState):
            components = (("keys", value.keys), ("values", value.values))
        else:
            continue
        for component, tensor in components:
            label = f"final-state:{key[0]}:{key[1]}:{component}"
            objective = _cotangent_objective(case, label, (tensor,))
            if objective is not None:
                objectives[label] = objective

    for region_id in sorted(result.balance_stats.regions):
        stats = result.balance_stats.regions[region_id]
        label = f"balance-region:{region_id}"
        objective = _cotangent_objective(case, label, (stats.soft_sum,))
        if objective is not None:
            objectives[label] = objective

    if result.trace is not None:
        # Keep every selector event as an independent root.  A single sum over
        # the whole trace could hide an event-local liveness error through
        # cancellation, especially in the required None-vs-connected-zero
        # comparison.
        for event_index, event in enumerate(result.trace.region_events):
            identity = (
                f"{event.sequence_id}:{event.token_position}:"
                f"{event.region_id}:{event_index}"
            )
            for field, value in (
                ("logits", event.logits),
                ("probabilities", event.probabilities),
            ):
                if value is None:
                    continue
                label = f"trace.region-{field}:{identity}"
                objective = _cotangent_objective(case, label, (value,))
                if objective is not None:
                    objectives[label] = objective

    objectives["combined"] = _backward_objective(case, result, result.output)
    # Revisit the first root after all other retained-graph queries.  Packed
    # structural liveness is per backward call and must replace, not union,
    # the roots selected by the preceding objective.
    objectives["output.repeat"] = output
    return objectives


def _autograd_record(
    objective: Tensor,
    model: SettleGraph,
    hidden: Tensor,
    *,
    retain_graph: bool,
) -> Mapping[str, object]:
    named = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        objective,
        (hidden, *(parameter for _, parameter in named)),
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return {
        "hidden": gradients[0],
        "parameters": {
            name: gradient
            for (name, _), gradient in zip(named, gradients[1:])
        },
    }


class CandidateSupportPartitionTests(unittest.TestCase):
    def test_static_acceptance_and_reason_partition_is_frozen(self) -> None:
        records = _support_partition()
        self.assertEqual(len(records), CORE_V1_CANDIDATE_CORPUS_SIZE)
        self.assertEqual(
            Counter(
                "accepted" if record.packed.accepted else "rejected"
                for record in records
            ),
            Counter({"accepted": 256}),
        )
        self.assertEqual(
            Counter(
                "accepted" if record.single_layer.supported else "rejected"
                for record in records
            ),
            Counter({"accepted": 8, "rejected": 248}),
        )
        self.assertEqual(
            Counter(
                "accepted" if record.hb.supported else "rejected"
                for record in records
            ),
            Counter({"accepted": 16, "rejected": 240}),
        )

        packed_reasons = Counter(
            reason
            for record in records
            if not record.packed.accepted
            for reason in record.packed_reason_codes
        )
        single_layer_reasons = Counter(
            reason
            for record in records
            if not record.single_layer.supported
            for reason in record.single_layer_reason_codes
        )
        hb_reasons = Counter(
            reason
            for record in records
            if not record.hb.supported
            for reason in record.hb_reason_codes
        )
        self.assertEqual(packed_reasons, Counter())
        self.assertEqual(
            single_layer_reasons,
            Counter(
                {
                    "single-layer.stateful": 238,
                    "single-layer.topology.region-count": 208,
                    "single-layer.topology.edges": 208,
                    "single-layer.profile": 40,
                    "single-layer.selector-read": 22,
                }
            ),
        )
        self.assertEqual(
            hb_reasons,
            Counter({"hb.topology.kind": 240, "hb.topology.line": 240}),
        )

        partition_text = "\n".join(
            "\t".join(
                (
                    record.case.case_id,
                    "P:" + ",".join(record.packed_reason_codes),
                    "S:" + ",".join(record.single_layer_reason_codes),
                    "H:" + ",".join(record.hb_reason_codes),
                )
            )
            for record in records
        )
        self.assertEqual(
            hashlib.sha256(partition_text.encode("utf-8")).hexdigest(),
            "49fb9c797f40546f29534bbaf1fac5c4b04669b4990239d4af22d3154fa4c703",
        )

    def test_vjp_marked_subset_has_a_stable_executor_intersection(self) -> None:
        records = _support_partition()
        self.assertEqual(sum(record.case.vjp for record in records), 64)
        self.assertEqual(
            sum(record.case.vjp and record.packed.accepted for record in records),
            64,
        )
        self.assertEqual(
            sum(
                record.case.vjp and record.single_layer.supported
                for record in records
            ),
            0,
        )
        self.assertEqual(
            sum(record.case.vjp and record.hb.supported for record in records),
            4,
        )


class ExecutorContractParityTests(unittest.TestCase):
    """Shared call-domain boundaries must not vary by executor binding."""

    @staticmethod
    def _bindings(model: SettleGraph) -> Tuple[object, ...]:
        return (
            model,
            PackedSettleGraph(model),
            SpecializedExecutor(model, SINGLE_LAYER_V1),
        )

    def test_all_executor_bindings_reject_unclosed_low_precision_roles(self) -> None:
        case = next(
            record.case
            for record in _support_partition()
            if record.single_layer.supported
        )
        for dtype in (torch.float16, torch.bfloat16):
            model = SettleGraph(case.plan).to(device="cpu", dtype=dtype)
            hidden = torch.zeros((1, 1, case.plan.d_model), dtype=dtype)
            for binding in self._bindings(model):
                with self.subTest(dtype=str(dtype), binding=type(binding).__name__):
                    with self.assertRaisesRegex(
                        ExecutionContractError, "only float32 and float64"
                    ):
                        binding.prefill(
                            hidden,
                            torch.ones((1, 1), dtype=torch.bool),
                            ("candidate.low-precision",),
                            torch.zeros((1, 1), dtype=torch.int64),
                        )

    def test_all_executor_bindings_reject_empty_prefill_and_decode_batch(self) -> None:
        case = next(
            record.case
            for record in _support_partition()
            if record.single_layer.supported
        )
        model = SettleGraph(case.plan).to(device="cpu", dtype=torch.float64)
        prefill_hidden = torch.empty((0, 1, case.plan.d_model), dtype=torch.float64)
        decode_hidden = torch.empty((0, case.plan.d_model), dtype=torch.float64)
        for binding in self._bindings(model):
            name = type(binding).__name__
            with self.subTest(binding=name, call="prefill"):
                with self.assertRaisesRegex(
                    ExecutionContractError, "requires a non-empty batch"
                ):
                    binding.prefill(
                        prefill_hidden,
                        torch.empty((0, 1), dtype=torch.bool),
                        (),
                        torch.empty((0, 1), dtype=torch.int64),
                    )
            method = (
                binding.interpret_token
                if isinstance(binding, SettleGraph)
                else binding.decode
            )
            with self.subTest(binding=name, call="decode"):
                with self.assertRaisesRegex(
                    ExecutionContractError, "requires a non-empty batch"
                ):
                    method(
                        decode_hidden,
                        torch.empty((0,), dtype=torch.bool),
                        (),
                        torch.empty((0,), dtype=torch.int64),
                    )


class CandidateForwardEquivalenceTests(unittest.TestCase):
    def test_all_packed_supported_cases_match_both_cpu_dtypes(self) -> None:
        accepted = [
            record for record in _support_partition() if record.packed.accepted
        ]
        for dtype in (torch.float64, torch.float32):
            for record in accepted:
                case = record.case
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    model = _model(case, dtype)
                    packed = PackedSettleGraph(model)
                    hidden = _hidden(case, dtype)
                    reference = model.prefill(
                        hidden.clone(), **_call_arguments(record_trace=True)
                    )
                    actual = packed.prefill(
                        hidden.clone(), **_call_arguments(record_trace=True)
                    )
                    _compare(reference, actual, dtype=dtype)
                    assert actual.trace is not None
                    validate_trace_invariants(
                        case.plan,
                        actual.trace,
                        _EXECUTED_TOKENS,
                        tolerance=_tolerance(dtype),
                    )
                    assert packed.last_profile is not None
                    self.assertEqual(packed.last_profile.python_token_hot_loops, 0)
                    self.assertEqual(
                        packed.last_profile.python_batch_row_hot_loops, 0
                    )
                    self.assertEqual(
                        packed.last_profile.python_node_event_hot_loops, 0
                    )
                    self.assertEqual(packed.last_profile.eager_scheduler_calls, 0)

    def test_all_packed_cases_match_full_chunked_and_decode_execution(self) -> None:
        records = _support_partition()
        self.assertEqual(len(records), 256)
        self.assertTrue(all(record.packed.accepted for record in records))

        for dtype in (torch.float64, torch.float32):
            for record in records:
                case = record.case
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    model = _model(case, dtype)
                    packed = PackedSettleGraph(model)
                    hidden = _hidden(case, dtype)
                    common = dict(_call_arguments(record_trace=True))
                    common["detach_at_end"] = True
                    with torch.no_grad():
                        reference_full = model.prefill(hidden.clone(), **common)
                        packed_full = packed.prefill(hidden.clone(), **common)
                        _compare(reference_full, packed_full, dtype=dtype)
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=PACKED_EXECUTOR_ID,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="full-prefill",
                            call_counts={"eager": 1, "packed": 1},
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="full-call",
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_packed_cases_match_full_chunked_and_decode_execution"
                            ),
                        )

                        # Alternate the nonempty boundary across the fixed
                        # three-token corpus shape so every case exercises
                        # state publication and continuation without adding a
                        # result-dependent case selection rule.
                        split = 1 + case.ordinal % 2
                        reference_chunks = _run_two_chunk_prefill(
                            model, hidden, split
                        )
                        packed_chunks = _run_two_chunk_prefill(
                            packed, hidden, split
                        )
                        for index, label in ((0, "first"), (1, "second")):
                            _compare(
                                reference_chunks[index],
                                packed_chunks[index],
                                dtype=dtype,
                                path=f"chunk.{label}.eager-packed",
                            )
                        _compare(
                            reference_full,
                            reference_chunks[2],
                            dtype=dtype,
                            path="chunk.reference-vs-full",
                        )
                        _compare(
                            packed_full,
                            packed_chunks[2],
                            dtype=dtype,
                            path="chunk.packed-vs-full",
                        )
                        _compare(
                            reference_chunks[2],
                            packed_chunks[2],
                            dtype=dtype,
                            path="chunk.eager-packed-combined",
                        )
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=PACKED_EXECUTOR_ID,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="two-chunk-prefill",
                            call_counts={"eager": 2, "packed": 2},
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="each-call-and-canonical-merge",
                            split=split,
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_packed_cases_match_full_chunked_and_decode_execution"
                            ),
                        )

                        reference_steps, reference_decode = _run_token_decode(
                            model, hidden
                        )
                        packed_steps, packed_decode = _run_token_decode(
                            packed, hidden
                        )
                        for token, (reference_step, packed_step) in enumerate(
                            zip(reference_steps, packed_steps)
                        ):
                            _compare(
                                reference_step,
                                packed_step,
                                dtype=dtype,
                                path=f"decode[{token}].eager-packed",
                            )
                        _compare(
                            reference_full,
                            reference_decode,
                            dtype=dtype,
                            path="decode.reference-vs-full",
                        )
                        _compare(
                            packed_full,
                            packed_decode,
                            dtype=dtype,
                            path="decode.packed-vs-full",
                        )
                        _compare(
                            reference_decode,
                            packed_decode,
                            dtype=dtype,
                            path="decode.eager-packed-combined",
                        )
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=PACKED_EXECUTOR_ID,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="token-by-token-decode",
                            call_counts={"eager": 3, "packed": 3},
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="each-call-and-canonical-merge",
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_packed_cases_match_full_chunked_and_decode_execution"
                            ),
                        )

    def test_all_specialized_supported_cases_match_both_cpu_dtypes(self) -> None:
        items = []
        for record in _support_partition():
            if record.single_layer.supported:
                items.append((record, SINGLE_LAYER_V1))
            if record.hb.supported:
                items.append((record, HB_LINE_V1))
        self.assertEqual(len(items), 24)

        for dtype in (torch.float64, torch.float32):
            for record, specialization in items:
                case = record.case
                with self.subTest(
                    dtype=str(dtype),
                    case=case.case_id,
                    specialization=specialization,
                ):
                    model = _model(case, dtype)
                    specialized = SpecializedExecutor(model, specialization)
                    packed = PackedSettleGraph(model)
                    hidden = _hidden(case, dtype)
                    with torch.no_grad():
                        reference = model.prefill(
                            hidden.clone(), **_call_arguments(record_trace=True)
                        )
                        actual = specialized.prefill(
                            hidden.clone(), **_call_arguments(record_trace=True)
                        )
                        _compare(reference, actual, dtype=dtype)
                        assert actual.trace is not None
                        validate_trace_invariants(
                            case.plan,
                            actual.trace,
                            _EXECUTED_TOKENS,
                            tolerance=_tolerance(dtype),
                        )

                        # The generic packed predicate covers every core-v1
                        # case.  Exercise true three-way comparisons rather
                        # than inferring packed equivalence transitively.
                        self.assertTrue(record.packed.accepted)
                        packed_result = packed.prefill(
                            hidden.clone(),
                            **_call_arguments(record_trace=True),
                        )
                        _compare(reference, packed_result, dtype=dtype)
                        _compare(packed_result, actual, dtype=dtype)
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=specialization,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="full-prefill",
                            call_counts={
                                "eager": 1,
                                "packed": 1,
                                "specialized": 1,
                            },
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="full-call",
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_specialized_supported_cases_match_both_cpu_dtypes"
                            ),
                        )

                        split = 1 + case.ordinal % 2
                        reference_chunks = _run_two_chunk_prefill(
                            model, hidden, split
                        )
                        packed_chunks = _run_two_chunk_prefill(
                            packed, hidden, split
                        )
                        specialized_chunks = _run_two_chunk_prefill(
                            specialized, hidden, split
                        )
                        for index, label in ((0, "first"), (1, "second")):
                            reference_chunk = reference_chunks[index]
                            packed_chunk = packed_chunks[index]
                            specialized_chunk = specialized_chunks[index]
                            _compare(
                                reference_chunk,
                                packed_chunk,
                                dtype=dtype,
                                path=f"two-chunk.{label}.packed",
                            )
                            _compare(
                                reference_chunk,
                                specialized_chunk,
                                dtype=dtype,
                                path=f"two-chunk.{label}.{specialization}",
                            )
                            _compare(
                                packed_chunk,
                                specialized_chunk,
                                dtype=dtype,
                                path=f"two-chunk.{label}.three-way",
                            )
                        for name, combined in (
                            ("reference", reference_chunks[2]),
                            ("packed", packed_chunks[2]),
                            (specialization, specialized_chunks[2]),
                        ):
                            _compare(
                                reference,
                                combined,
                                dtype=dtype,
                                path=f"two-chunk.{name}.vs-full",
                            )
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=specialization,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="two-chunk-prefill",
                            call_counts={
                                "eager": 2,
                                "packed": 2,
                                "specialized": 2,
                            },
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="each-call-and-canonical-merge",
                            split=split,
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_specialized_supported_cases_match_both_cpu_dtypes"
                            ),
                        )

                        reference_steps, reference_decode = _run_token_decode(
                            model, hidden
                        )
                        packed_steps, packed_decode = _run_token_decode(
                            packed, hidden
                        )
                        specialized_steps, specialized_decode = _run_token_decode(
                            specialized, hidden
                        )
                        for token, step_items in enumerate(
                            zip(
                                reference_steps,
                                packed_steps,
                                specialized_steps,
                            )
                        ):
                            reference_step, packed_step, specialized_step = step_items
                            _compare(
                                reference_step,
                                packed_step,
                                dtype=dtype,
                                path=f"decode[{token}].packed",
                            )
                            _compare(
                                reference_step,
                                specialized_step,
                                dtype=dtype,
                                path=f"decode[{token}].{specialization}",
                            )
                            _compare(
                                packed_step,
                                specialized_step,
                                dtype=dtype,
                                path=f"decode[{token}].three-way",
                            )
                        for name, combined in (
                            ("reference", reference_decode),
                            ("packed", packed_decode),
                            (specialization, specialized_decode),
                        ):
                            _compare(
                                reference,
                                combined,
                                dtype=dtype,
                                path=f"decode.{name}.vs-full",
                            )
                        _record_execution_receipt(
                            kind="forward-cell",
                            executor=specialization,
                            case_id=case.case_id,
                            dtype=_dtype_name(dtype),
                            mode="token-by-token-decode",
                            call_counts={
                                "eager": 3,
                                "packed": 3,
                                "specialized": 3,
                            },
                            observables=[
                                "output",
                                "final-state",
                                "balance-sufficient-statistics",
                                "canonical-trace",
                                "exact-route",
                            ],
                            trace_scope="each-call-and-canonical-merge",
                            source_test_id=(
                                "CandidateForwardEquivalenceTests."
                                "test_all_specialized_supported_cases_match_both_cpu_dtypes"
                            ),
                        )


class CandidateVJPEquivalenceTests(unittest.TestCase):
    def _compare_isolated_packed_vjps(
        self,
        case: PlanCorpusCase,
        dtype: torch.dtype,
    ) -> None:
        reference_model = _model(case, dtype)
        packed_model = _model(case, dtype)
        packed_model.load_state_dict(reference_model.state_dict(), strict=True)
        source = _hidden(case, dtype)
        reference_hidden = source.clone().requires_grad_(True)
        packed_hidden = source.clone().requires_grad_(True)
        reference = reference_model.prefill(
            reference_hidden, **_call_arguments(record_trace=True)
        )
        packed = PackedSettleGraph(packed_model).prefill(
            packed_hidden, **_call_arguments(record_trace=True)
        )
        reference_objectives = _isolated_objectives(case, reference)
        packed_objectives = _isolated_objectives(case, packed)
        self.assertEqual(
            tuple(reference_objectives), tuple(packed_objectives)
        )
        labels = tuple(reference_objectives)
        self.assertIn("output", labels)
        self.assertIn("combined", labels)
        for index, label in enumerate(labels):
            retain_graph = index + 1 < len(labels)
            with self.subTest(objective=label):
                _compare(
                    reference_objectives[label],
                    packed_objectives[label],
                    dtype=dtype,
                    path=f"packed.objective.{label}",
                )
                reference_gradients = _autograd_record(
                    reference_objectives[label],
                    reference_model,
                    reference_hidden,
                    retain_graph=retain_graph,
                )
                packed_gradients = _autograd_record(
                    packed_objectives[label],
                    packed_model,
                    packed_hidden,
                    retain_graph=retain_graph,
                )
                _compare(
                    reference_gradients,
                    packed_gradients,
                    dtype=dtype,
                    path=f"packed.isolated-vjp.{label}",
                )
                for gradient in (
                    packed_gradients["hidden"],
                    *packed_gradients["parameters"].values(),
                ):
                    if gradient is not None:
                        self.assertTrue(
                            bool(torch.isfinite(gradient).all().item())
                        )
                _record_execution_receipt(
                    kind="vjp-objective",
                    executor=PACKED_EXECUTOR_ID,
                    case_id=case.case_id,
                    dtype=_dtype_name(dtype),
                    mode="full-prefill",
                    objective_id=label,
                    objective_family=_objective_family(label),
                    checks=[
                        "objective-value",
                        "hidden-vjp",
                        "all-named-parameter-vjp",
                        "none-vs-tensor-connectivity",
                        "finite",
                    ],
                    source_test_id=(
                        "CandidateVJPEquivalenceTests."
                        "test_all_64_marked_packed_vjp_cases_match_values_and_none"
                    ),
                )
        _record_execution_receipt(
            kind="vjp-case-complete",
            executor=PACKED_EXECUTOR_ID,
            case_id=case.case_id,
            dtype=_dtype_name(dtype),
            mode="full-prefill",
            objective_ids=list(labels),
            source_test_id=(
                "CandidateVJPEquivalenceTests."
                "test_all_64_marked_packed_vjp_cases_match_values_and_none"
            ),
        )

    def _compare_vjp(
        self,
        case: PlanCorpusCase,
        dtype: torch.dtype,
        candidate_kind: str,
    ) -> None:
        reference_model = _model(case, dtype)
        candidate_model = _model(case, dtype)
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        packed_model: Optional[SettleGraph] = None
        source = _hidden(case, dtype)
        reference_hidden = source.clone().requires_grad_(True)
        candidate_hidden = source.clone().requires_grad_(True)
        reference = reference_model.prefill(
            reference_hidden, **_call_arguments()
        )
        if candidate_kind == "packed":
            candidate = PackedSettleGraph(candidate_model).prefill(
                candidate_hidden, **_call_arguments()
            )
        else:
            candidate = SpecializedExecutor(
                candidate_model, candidate_kind
            ).prefill(candidate_hidden, **_call_arguments())
            packed_model = _model(case, dtype)
            packed_model.load_state_dict(
                reference_model.state_dict(), strict=True
            )
            packed_hidden = source.clone().requires_grad_(True)
            packed = PackedSettleGraph(packed_model).prefill(
                packed_hidden, **_call_arguments()
            )
        _backward_objective(case, reference, reference_hidden).backward()
        _backward_objective(case, candidate, candidate_hidden).backward()
        reference_gradients = _gradient_record(reference_model, reference_hidden)
        candidate_gradients = _gradient_record(candidate_model, candidate_hidden)
        _compare(
            reference_gradients,
            candidate_gradients,
            dtype=dtype,
            path=f"{candidate_kind}.vjp",
        )
        if packed_model is not None:
            _backward_objective(case, packed, packed_hidden).backward()
            packed_gradients = _gradient_record(packed_model, packed_hidden)
            _compare(
                reference_gradients,
                packed_gradients,
                dtype=dtype,
                path=f"{candidate_kind}.packed-reference-vjp",
            )
            _compare(
                packed_gradients,
                candidate_gradients,
                dtype=dtype,
                path=f"{candidate_kind}.packed-specialized-vjp",
            )
        _record_execution_receipt(
            kind="vjp-objective",
            executor=candidate_kind,
            case_id=case.case_id,
            dtype=_dtype_name(dtype),
            mode="full-prefill",
            objective_id="combined-output-balance-state",
            objective_family="combined-output-balance-state",
            checks=[
                "hidden-vjp",
                "all-named-parameter-vjp",
                "none-vs-tensor-connectivity",
                "three-way-eager-packed-specialized",
            ],
            source_test_id=(
                "CandidateVJPEquivalenceTests."
                "test_every_statically_accepted_specialization_matches_vjp_and_none"
            ),
        )
        _record_execution_receipt(
            kind="vjp-case-complete",
            executor=candidate_kind,
            case_id=case.case_id,
            dtype=_dtype_name(dtype),
            mode="full-prefill",
            objective_ids=["combined-output-balance-state"],
            source_test_id=(
                "CandidateVJPEquivalenceTests."
                "test_every_statically_accepted_specialization_matches_vjp_and_none"
            ),
        )

    def test_all_64_marked_packed_vjp_cases_match_values_and_none(self) -> None:
        records = _support_partition()
        selected = [record for record in records if record.case.vjp]
        self.assertEqual(len(selected), 64)
        self.assertTrue(all(record.packed.accepted for record in selected))
        for dtype in (torch.float64, torch.float32):
            for record in selected:
                case = record.case
                with self.subTest(
                    dtype=str(dtype), case=case.case_id, executor="packed"
                ):
                    self._compare_isolated_packed_vjps(case, dtype)

    def test_every_statically_accepted_specialization_matches_vjp_and_none(
        self,
    ) -> None:
        single_layer = [
            record
            for record in _support_partition()
            if record.single_layer.supported
        ]
        hb = [record for record in _support_partition() if record.hb.supported]
        self.assertEqual(len(single_layer), 8)
        self.assertEqual(len(hb), 16)
        items = (
            *((record, SINGLE_LAYER_V1) for record in single_layer),
            *((record, HB_LINE_V1) for record in hb),
        )
        for dtype in (torch.float64, torch.float32):
            for record, specialization in items:
                with self.subTest(
                    dtype=str(dtype),
                    case=record.case.case_id,
                    executor=specialization,
                ):
                    self._compare_vjp(record.case, dtype, specialization)


class CandidateLifecycleEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _binding(
        model: SettleGraph, candidate_kind: str
    ) -> PackedSettleGraph | SpecializedExecutor:
        if candidate_kind == "packed":
            return PackedSettleGraph(model)
        return SpecializedExecutor(model, candidate_kind)

    def _exercise_lifecycle(
        self, case: PlanCorpusCase, candidate_kind: str
    ) -> None:
        dtype = torch.float64
        reference_model = _model(case, dtype)
        candidate_model = _model(case, dtype)
        candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
        candidate = self._binding(candidate_model, candidate_kind)

        hidden = _hidden(case, dtype, length=4)
        execution = torch.tensor(
            [[True, False, True, True], [True, True, False, True]],
            dtype=torch.bool,
        )
        positions = torch.tensor(
            [[0, 99, 1, 2], [0, 1, 99, 2]], dtype=torch.int64
        )
        lm_mask = torch.tensor(
            [[False, False, True, True], [False, True, False, True]],
            dtype=torch.bool,
        )
        route_mask = torch.tensor(
            [[True, False, False, True], [False, True, False, True]],
            dtype=torch.bool,
        )
        common = {
            "execution_mask": execution,
            "sequence_ids": _SEQUENCE_IDS,
            "token_positions": positions,
            "lm_target_mask": lm_mask,
            "routing_stats_mask": route_mask,
            "detach_at_end": False,
            "record_trace": True,
        }
        reference_full = reference_model.prefill(hidden.clone(), **common)
        candidate_full = candidate.prefill(hidden.clone(), **common)
        _compare(reference_full, candidate_full, dtype=dtype)

        # Every split, including an empty prefix and empty tail, uses public
        # state carry and compares each chunk's trace independently.
        for split in range(5):
            reference_first = reference_model.prefill(
                hidden[:, :split],
                execution[:, :split],
                _SEQUENCE_IDS,
                positions[:, :split],
                lm_target_mask=lm_mask[:, :split],
                routing_stats_mask=route_mask[:, :split],
                detach_at_end=False,
                record_trace=True,
            )
            candidate_first = candidate.prefill(
                hidden[:, :split],
                execution[:, :split],
                _SEQUENCE_IDS,
                positions[:, :split],
                lm_target_mask=lm_mask[:, :split],
                routing_stats_mask=route_mask[:, :split],
                detach_at_end=False,
                record_trace=True,
            )
            _compare(reference_first, candidate_first, dtype=dtype)
            reference_second = reference_model.prefill(
                hidden[:, split:],
                execution[:, split:],
                _SEQUENCE_IDS,
                positions[:, split:],
                state=reference_first.state,
                lm_target_mask=lm_mask[:, split:],
                routing_stats_mask=route_mask[:, split:],
                detach_at_end=False,
                record_trace=True,
            )
            candidate_second = candidate.prefill(
                hidden[:, split:],
                execution[:, split:],
                _SEQUENCE_IDS,
                positions[:, split:],
                state=candidate_first.state,
                lm_target_mask=lm_mask[:, split:],
                routing_stats_mask=route_mask[:, split:],
                detach_at_end=False,
                record_trace=True,
            )
            _compare(reference_second, candidate_second, dtype=dtype)
            _compare(
                reference_full.output,
                torch.cat(
                    (candidate_first.output, candidate_second.output), dim=1
                ),
                dtype=dtype,
                path=f"split[{split}].output",
            )
            _compare(
                reference_full.state,
                candidate_second.state,
                dtype=dtype,
                path=f"split[{split}].state",
            )
            merged = candidate_first.balance_stats.merge(
                candidate_second.balance_stats
            )
            _compare(
                reference_full.balance_stats,
                merged,
                dtype=dtype,
                path=f"split[{split}].balance",
            )

        reference_state: Optional[StateStore] = None
        candidate_state: Optional[StateStore] = None
        decode_outputs = []
        decode_stats: Optional[BalanceStats] = None
        for token in range(4):
            reference_step = reference_model.interpret_token(
                hidden[:, token],
                execution[:, token],
                _SEQUENCE_IDS,
                positions[:, token],
                state=reference_state,
                lm_target_mask=lm_mask[:, token],
                routing_stats_mask=route_mask[:, token],
                detach_at_end=False,
                record_trace=True,
            )
            candidate_step = candidate.decode(
                hidden[:, token],
                execution[:, token],
                _SEQUENCE_IDS,
                positions[:, token],
                state=candidate_state,
                lm_target_mask=lm_mask[:, token],
                routing_stats_mask=route_mask[:, token],
                detach_at_end=False,
                record_trace=True,
            )
            _compare(reference_step, candidate_step, dtype=dtype)
            reference_state = reference_step.state
            candidate_state = candidate_step.state
            decode_outputs.append(candidate_step.output)
            decode_stats = (
                candidate_step.balance_stats
                if decode_stats is None
                else decode_stats.merge(candidate_step.balance_stats)
            )
        _compare(
            reference_full.output,
            torch.stack(decode_outputs, dim=1),
            dtype=dtype,
            path="decode.output",
        )
        _compare(
            reference_full.state,
            candidate_state,
            dtype=dtype,
            path="decode.state",
        )
        _compare(
            reference_full.balance_stats,
            decode_stats,
            dtype=dtype,
            path="decode.balance",
        )

        assert reference_state is not None and candidate_state is not None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(case.input_seed ^ 0x6E5E7)
        reset_hidden = torch.randn(
            (2, case.plan.d_model),
            generator=generator,
            dtype=dtype,
        )
        reversed_ids = (_SEQUENCE_IDS[1], _SEQUENCE_IDS[0])
        reset_positions = torch.tensor([3, 0], dtype=torch.int64)
        reference_reset = reference_model.interpret_token(
            reset_hidden,
            torch.ones((2,), dtype=torch.bool),
            reversed_ids,
            reset_positions,
            state=reference_state,
            reset_sequence_ids=(_SEQUENCE_IDS[0],),
            detach_at_end=False,
            record_trace=True,
        )
        candidate_reset = candidate.decode(
            reset_hidden,
            torch.ones((2,), dtype=torch.bool),
            reversed_ids,
            reset_positions,
            state=candidate_state,
            reset_sequence_ids=(_SEQUENCE_IDS[0],),
            detach_at_end=False,
            record_trace=True,
        )
        _compare(reference_reset, candidate_reset, dtype=dtype)

        empty_hidden = reset_hidden.new_empty((2, 0, case.plan.d_model))
        empty_mask = torch.empty((2, 0), dtype=torch.bool)
        empty_positions = torch.empty((2, 0), dtype=torch.int64)
        reference_empty = reference_model.prefill(
            empty_hidden,
            empty_mask,
            reversed_ids,
            empty_positions,
            state=reference_reset.state,
            detach_at_end=False,
            record_trace=True,
        )
        candidate_empty = candidate.prefill(
            empty_hidden,
            empty_mask,
            reversed_ids,
            empty_positions,
            state=candidate_reset.state,
            detach_at_end=False,
            record_trace=True,
        )
        _compare(reference_empty, candidate_empty, dtype=dtype)
        self.assertEqual(candidate_empty.output.shape[1], 0)
        assert candidate_empty.trace is not None
        self.assertEqual(candidate_empty.trace.node_events, ())
        self.assertEqual(candidate_empty.trace.output_events, ())

    def test_representative_packed_and_specialized_lifecycle_scenarios(self) -> None:
        by_ordinal = {case.ordinal: case for case in _corpus()}
        packed_cases = (by_ordinal[176], by_ordinal[19])
        self.assertEqual(
            {
                feature
                for case in packed_cases
                for feature in case.features
                if feature.startswith("profile:")
            },
            {
                "profile:N/content",
                "profile:SD/content",
                "profile:SD/pre",
                "profile:BO/content",
                "profile:BO/pre",
                "profile:BO/post",
            },
        )
        self.assertEqual(
            {
                feature
                for case in packed_cases
                for feature in case.features
                if feature.startswith("state:")
            },
            {"state:none", "state:ema", "state:gdn", "state:attention_window"},
        )
        scenarios = (
            *((case, "packed") for case in packed_cases),
            (by_ordinal[23], SINGLE_LAYER_V1),
            (by_ordinal[144], HB_LINE_V1),
        )
        for case, candidate_kind in scenarios:
            with self.subTest(case=case.case_id, executor=candidate_kind):
                if candidate_kind == "packed":
                    self.assertTrue(inspect_packed_support(case.plan).accepted)
                elif candidate_kind == SINGLE_LAYER_V1:
                    self.assertTrue(single_layer_v1_support(case.plan).supported)
                else:
                    self.assertTrue(hb_line_v1_support(case.plan).supported)
                self._exercise_lifecycle(case, candidate_kind)
                _record_execution_receipt(
                    kind="lifecycle-scenario",
                    executor=(
                        PACKED_EXECUTOR_ID
                        if candidate_kind == "packed"
                        else candidate_kind
                    ),
                    case_id=case.case_id,
                    dtype="float64",
                    scenarios=[
                        "full-prefill",
                        "all-five-chunk-boundaries-including-empty-ends",
                        "token-by-token-decode",
                        "row-reorder",
                        "sequence-reset",
                        "empty-tail",
                        "no-detach-state-carry",
                    ],
                    source_test_id=(
                        "CandidateLifecycleEquivalenceTests."
                        "test_representative_packed_and_specialized_lifecycle_scenarios"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
