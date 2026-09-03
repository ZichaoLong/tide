from __future__ import annotations

import hashlib
import unittest
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor

from tide.engine import ExecutionResult, SettleGraph, StateStore
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    Tolerance,
    compare_nested,
    validate_trace_invariants,
)
from tide.generators import (
    DEFAULT_INVALID_PLAN_CORPUS_SIZE,
    DEFAULT_PLAN_CORPUS_SEED,
    DEFAULT_PLAN_CORPUS_SIZE,
    PlanCorpusCase,
    generate_invalid_plan_corpus,
    generate_plan_corpus,
)
from tide.failures import failure_envelope_from_exception
from tide.ops import AttentionState
from tide.plan import Plan, PlanValidationError, bind_dtypes


_BATCH = 2
_LENGTH = 3


def _tolerance(dtype: torch.dtype) -> Tolerance:
    return (
        CPU_FLOAT64_TOLERANCE
        if dtype == torch.float64
        else SAME_BACKEND_FLOAT32_TOLERANCE
    )


def _inputs(
    case: PlanCorpusCase, dtype: torch.dtype
) -> Tuple[Tensor, Tensor, Tuple[str, ...], Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed)
    source = torch.randn(
        (_BATCH, _LENGTH, case.plan.d_model),
        device="cpu",
        dtype=torch.float64,
        generator=generator,
    )
    hidden = source.to(dtype=dtype)
    execution_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        device="cpu",
        dtype=torch.bool,
    )
    sequence_ids = ("sequence.a", "sequence.b")
    token_positions = torch.tensor(
        [[0, 1, 2], [0, 1, 99]], device="cpu", dtype=torch.int64
    )
    lm_target_mask = torch.tensor(
        [[False, True, True], [False, True, False]],
        device="cpu",
        dtype=torch.bool,
    )
    routing_stats_mask = torch.tensor(
        [[True, False, True], [True, True, False]],
        device="cpu",
        dtype=torch.bool,
    )
    return (
        hidden,
        execution_mask,
        sequence_ids,
        token_positions,
        lm_target_mask,
        routing_stats_mask,
    )


def _requested_k(
    plan: Plan, *, ordinal: int
) -> Optional[Mapping[str, Tensor]]:
    supplied: Dict[str, Tensor] = {}
    for region_index, region in enumerate(plan.regions):
        if region.k_requested["type"] != "input":
            continue
        values = torch.empty(
            (_BATCH, _LENGTH), device="cpu", dtype=torch.int64
        )
        for row in range(_BATCH):
            for column in range(_LENGTH):
                values[row, column] = 1 + (
                    (ordinal + region_index + row + column) % region.k_max
                )
        supplied[region.region_id] = values
    return supplied or None


def _executed_tokens() -> Tuple[Tuple[str, int], ...]:
    return (
        ("sequence.a", 0),
        ("sequence.a", 1),
        ("sequence.a", 2),
        ("sequence.b", 0),
        ("sequence.b", 1),
    )


def _models(
    case: PlanCorpusCase, dtype: torch.dtype
) -> Tuple[SettleGraph, SettleGraph]:
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    typed_plan = bind_dtypes(
        case.plan,
        hidden=dtype_name,
        parameter=dtype_name,
        state=dtype_name,
        readout=dtype_name,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.parameter_seed)
        token_major = SettleGraph(typed_plan).to(device="cpu", dtype=dtype)
        region_major = SettleGraph(typed_plan).to(device="cpu", dtype=dtype)
    region_major.load_state_dict(token_major.state_dict(), strict=True)
    if any(
        parameter.device.type != "cpu" for parameter in token_major.parameters()
    ):
        raise AssertionError("token-major corpus model is not on CPU")
    if any(
        parameter.device.type != "cpu" for parameter in region_major.parameters()
    ):
        raise AssertionError("region-major corpus model is not on CPU")
    token_major.eval()
    region_major.eval()
    return token_major, region_major


def _run_pair(
    case: PlanCorpusCase,
    dtype: torch.dtype,
    *,
    requires_grad: bool,
    record_trace: bool,
) -> Tuple[ExecutionResult, ExecutionResult, Tensor, Tensor, SettleGraph, SettleGraph]:
    token_major, region_major = _models(case, dtype)
    inputs = _inputs(case, dtype)
    source_hidden = inputs[0]
    left_hidden = source_hidden.clone().requires_grad_(requires_grad)
    right_hidden = source_hidden.clone().requires_grad_(requires_grad)
    common = dict(
        execution_mask=inputs[1],
        sequence_ids=inputs[2],
        token_positions=inputs[3],
        requested_k=_requested_k(case.plan, ordinal=case.ordinal),
        lm_target_mask=inputs[4],
        routing_stats_mask=inputs[5],
        detach_at_end=False,
        record_trace=record_trace,
    )
    left = token_major.prefill(left_hidden, **common)
    right = region_major.prefill_region_major(right_hidden, **common)
    return left, right, left_hidden, right_hidden, token_major, region_major


def _state_objective(store: StateStore, reference: Tensor) -> Tensor:
    terms = []
    for key in sorted(store.values):
        state = store.values[key]
        if isinstance(state, Tensor):
            terms.append(state.sum())
        elif isinstance(state, AttentionState):
            terms.extend((state.keys.sum(), state.values.sum()))
    return torch.stack(terms).sum() if terms else reference.sum() * 0.0


def _gradient_record(model: SettleGraph, hidden: Tensor) -> Mapping[str, object]:
    return {
        "hidden": hidden.grad,
        "parameters": {
            name: parameter.grad for name, parameter in model.named_parameters()
        },
    }


def _floating_gradients(values: Mapping[str, object]) -> Iterable[Tensor]:
    hidden = values["hidden"]
    if isinstance(hidden, Tensor):
        yield hidden
    parameters = values["parameters"]
    assert isinstance(parameters, Mapping)
    for value in parameters.values():
        if isinstance(value, Tensor):
            yield value


class DeterministicPlanCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = generate_plan_corpus()

    def test_corpus_is_deterministic_valid_expanded_and_broad(self) -> None:
        self.assertEqual(len(self.corpus), DEFAULT_PLAN_CORPUS_SIZE)
        repeated = generate_plan_corpus(DEFAULT_PLAN_CORPUS_SEED)
        identities = tuple(
            (case.case_id, case.plan.canonical_hash()) for case in self.corpus
        )
        self.assertEqual(
            identities,
            tuple((case.case_id, case.plan.canonical_hash()) for case in repeated),
        )
        self.assertEqual(len({case.case_id for case in self.corpus}), len(self.corpus))
        self.assertEqual(
            len({case.plan.canonical_hash() for case in self.corpus}),
            len(self.corpus),
        )
        self.assertNotEqual(
            tuple(case.plan.canonical_hash() for case in self.corpus),
            tuple(
                case.plan.canonical_hash()
                for case in generate_plan_corpus(DEFAULT_PLAN_CORPUS_SEED + 1)
            ),
        )

        digest = hashlib.sha256(
            "\n".join(plan_hash for _, plan_hash in identities).encode("ascii")
        ).hexdigest()
        # This is a development-corpus identity, not qualification evidence.
        self.assertEqual(
            digest,
            "bc58b58e4cfdabc0ad863e3cbf86f820a750b93cfb4d0a023c02b14b0218af2d",
        )

        for case in self.corpus:
            with self.subTest(case=case.case_id):
                self.assertIs(case.plan.validate(), case.plan)
                self.assertEqual(
                    len(case.plan.topological_regions), len(case.plan.regions)
                )

        coverage = frozenset(
            feature for case in self.corpus for feature in case.features
        )
        expected = {
            "motif:singleton",
            "motif:single-layer-r2",
            "motif:single-layer-r8",
            "motif:chain",
            "motif:diamond",
            "motif:unequal-path",
            "motif:multi-entry-terminal",
            "motif:mixed-regions",
            "motif:forced-backbone",
            "motif:small-hb",
            "motif:generated-dag",
            "profile:N/content",
            "profile:SD/content",
            "profile:SD/pre",
            "profile:BO/content",
            "profile:BO/pre",
            "profile:BO/post",
            "state:none",
            "state:ema",
            "state:gdn",
            "state:attention_window",
            "emit:hard",
            "emit:hst",
            "emit:softp",
            "aggregate:mean",
            "aggregate:edge_softmax",
            "aggregate:edge_linear_mean",
            "output-aggregate:mean",
            "output-aggregate:node_softmax",
            "k:fixed",
            "k:input",
            "budget:top-1",
            "budget:top-2",
            "budget:all",
            "routing:forced-active",
            "boundary:multi-entry",
            "boundary:multi-terminal",
            "topology:hb",
        }
        self.assertFalse(expected - coverage, sorted(expected - coverage))
        self.assertEqual(sum(case.vjp for case in self.corpus), 6)

        fixed_single_layer_budgets = {
            (case.motif, len(region.node_ids), int(region.k_requested["value"]))
            for case in self.corpus
            for region in case.plan.regions
            if case.motif in {"single-layer-r2", "single-layer-r8"}
            and region.k_requested["type"] == "fixed"
        }
        self.assertTrue(
            {
                ("single-layer-r2", 2, 1),
                ("single-layer-r2", 2, 2),
                ("single-layer-r8", 8, 1),
                ("single-layer-r8", 8, 2),
                ("single-layer-r8", 8, 8),
            }
            <= fixed_single_layer_budgets
        )

    def test_cpu_float32_and_float64_schedules_match(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tolerance = _tolerance(dtype)
            for case in self.corpus:
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    left, right, _, _, _, _ = _run_pair(
                        case,
                        dtype,
                        requires_grad=False,
                        record_trace=True,
                    )
                    self.assertIsNotNone(left.trace)
                    self.assertIsNotNone(right.trace)
                    assert left.trace is not None and right.trace is not None
                    validate_trace_invariants(
                        case.plan,
                        left.trace,
                        _executed_tokens(),
                        tolerance=tolerance,
                    )
                    validate_trace_invariants(
                        case.plan,
                        right.trace,
                        _executed_tokens(),
                        tolerance=tolerance,
                    )
                    compare_nested(left, right, tolerance=tolerance).require_pass()
                    self.assertEqual(left.output.dtype, dtype)
                    self.assertEqual(right.output.dtype, dtype)
                    self.assertEqual(left.output.device.type, "cpu")
                    self.assertEqual(right.output.device.type, "cpu")

    def test_selected_cases_match_vjp(self) -> None:
        for dtype in (torch.float32, torch.float64):
            tolerance = _tolerance(dtype)
            for case in (item for item in self.corpus if item.vjp):
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    left, right, left_hidden, right_hidden, left_model, right_model = (
                        _run_pair(
                            case,
                            dtype,
                            requires_grad=True,
                            record_trace=False,
                        )
                    )
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(case.input_seed ^ 0x5A17)
                    cotangent = torch.randn(
                        left.output.shape,
                        device="cpu",
                        generator=generator,
                        dtype=torch.float64,
                    ).to(dtype=dtype)
                    left_objective = (
                        (left.output * cotangent).sum()
                        + 0.07 * left.balance_loss
                        + 0.03 * _state_objective(left.state, left.output)
                    )
                    right_objective = (
                        (right.output * cotangent).sum()
                        + 0.07 * right.balance_loss
                        + 0.03 * _state_objective(right.state, right.output)
                    )
                    left_objective.backward()
                    right_objective.backward()
                    left_gradients = _gradient_record(left_model, left_hidden)
                    right_gradients = _gradient_record(right_model, right_hidden)
                    compare_nested(
                        left_gradients,
                        right_gradients,
                        tolerance=tolerance,
                    ).require_pass()
                    gradients = tuple(_floating_gradients(left_gradients))
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                    parameter_gradients = gradients[1:]
                    self.assertTrue(
                        any(bool((value != 0).any().item()) for value in parameter_gradients)
                    )


class InvalidPlanCorpusTests(unittest.TestCase):
    def test_named_single_mutants_fail_with_stable_envelopes(self) -> None:
        corpus = generate_invalid_plan_corpus()
        self.assertEqual(len(corpus), DEFAULT_INVALID_PLAN_CORPUS_SIZE)
        repeated = generate_invalid_plan_corpus(DEFAULT_PLAN_CORPUS_SEED)
        self.assertEqual(
            tuple(
                (
                    case.case_id,
                    case.mutation_kind,
                    case.base_plan_hash,
                    case.expected_codes,
                )
                for case in corpus
            ),
            tuple(
                (
                    case.case_id,
                    case.mutation_kind,
                    case.base_plan_hash,
                    case.expected_codes,
                )
                for case in repeated
            ),
        )
        self.assertEqual(
            {case.mutation_kind for case in corpus},
            {
                "cycle",
                "duplicate-edge-id",
                "intra-region-edge",
                "hidden-shape",
                "fixed-k-zero",
                "profile-timing",
                "unstable-node-id",
                "wrong-terminal-set",
            },
        )
        for case in corpus:
            with self.subTest(case=case.case_id):
                with self.assertRaises(PlanValidationError) as raised:
                    case.plan.validate()
                self.assertEqual(
                    raised.exception.failure_codes, case.expected_codes
                )
                envelope = failure_envelope_from_exception(raised.exception)
                self.assertEqual(envelope.codes, case.expected_codes)
                self.assertEqual(envelope.phase, "plan")


if __name__ == "__main__":
    unittest.main()
