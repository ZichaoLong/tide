from __future__ import annotations

import dataclasses
import math
import unittest
from unittest import mock
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor

import tide.engine as engine_module
import tide.packed as packed_module
import tide.specialized as specialized_module

from tide.builders import (
    build_chain,
    build_single_layer,
    build_singleton,
    build_small_hb,
)
from tide.engine import (
    DynamicReachabilityError,
    ExecutionContractError,
    ExecutionResult,
    SettleGraph,
    StateStore,
    UnsupportedPlanError,
)
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    compare_nested,
    validate_trace_invariants,
)
from tide.failures import (
    ExecutionFailed,
    FailureEnvelope,
    capture_execution,
    compare_failure_envelopes,
)
from tide.generators import (
    PlanCorpusCase,
    generate_core_v1_candidate_corpus,
    generate_plan_corpus,
)
from tide.ops import AttentionState, safe_module_key
from tide.packed import PackedSettleGraph
from tide.specialized import (
    HB_LINE_V1,
    SINGLE_LAYER_V1,
    SpecializedExecutor,
    hb_line_v1_support,
    single_layer_v1_support,
)


_SEQUENCE_IDS = ("sequence.a", "sequence.b")
_EXECUTION_MASK = torch.tensor(
    [[True, True, True], [True, True, False]], dtype=torch.bool
)
_POSITIONS = torch.tensor([[0, 1, 2], [0, 1, 99]], dtype=torch.int64)
_ROUTING_MASK = torch.tensor(
    [[True, False, True], [True, True, False]], dtype=torch.bool
)
_EXECUTED = (
    ("sequence.a", 0),
    ("sequence.a", 1),
    ("sequence.a", 2),
    ("sequence.b", 0),
    ("sequence.b", 1),
)


def _hidden(case: PlanCorpusCase) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed)
    return torch.randn(
        (2, 3, case.plan.d_model),
        generator=generator,
        dtype=torch.float64,
    )


def _model(case: PlanCorpusCase) -> SettleGraph:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.parameter_seed)
        return SettleGraph(case.plan).to(dtype=torch.float64)


def _common(*, trace: bool = False) -> Mapping[str, object]:
    return {
        "execution_mask": _EXECUTION_MASK,
        "sequence_ids": _SEQUENCE_IDS,
        "token_positions": _POSITIONS,
        "routing_stats_mask": _ROUTING_MASK,
        "detach_at_end": False,
        "record_trace": trace,
    }


def _compare(left: object, right: object, path: str) -> None:
    compare_nested(
        left,
        right,
        tolerance=CPU_FLOAT64_TOLERANCE,
        path=path,
    ).require_pass()


def _state_objective(state: StateStore, reference: Tensor) -> Tensor:
    terms = []
    for key in sorted(state.values):
        value = state.values[key]
        if isinstance(value, Tensor):
            terms.append(value.sum())
        elif isinstance(value, AttentionState):
            terms.extend((value.keys.sum(), value.values.sum()))
    return torch.stack(terms).sum() if terms else reference.sum() * 0.0


def _gradients(
    result: ExecutionResult, model: SettleGraph, hidden: Tensor
) -> Tuple[Optional[Tensor], ...]:
    objective = (
        result.output.square().sum()
        + result.balance_loss
        + _state_objective(result.state, hidden)
    )
    return torch.autograd.grad(
        objective,
        (hidden, *model.parameters()),
        allow_unused=True,
    )


def _accepted_cases() -> Iterable[Tuple[PlanCorpusCase, str]]:
    for case in generate_plan_corpus():
        if single_layer_v1_support(case.plan).supported:
            yield case, SINGLE_LAYER_V1
        if hb_line_v1_support(case.plan).supported:
            yield case, HB_LINE_V1


class SpecializedSupportTests(unittest.TestCase):
    def test_versioned_predicates_accept_only_their_static_topologies(self) -> None:
        flat = build_single_layer(receiver_count=8, k=2)
        hb = build_small_hb()
        self.assertTrue(single_layer_v1_support(flat).supported)
        self.assertFalse(hb_line_v1_support(flat).supported)
        self.assertTrue(hb_line_v1_support(hb).supported)
        self.assertFalse(single_layer_v1_support(hb).supported)
        hb_executor = SpecializedExecutor(SettleGraph(hb), HB_LINE_V1)
        self.assertEqual(
            tuple(line for line, _, _ in hb_executor.hb_line_schedule),
            tuple(range(6)),
        )
        self.assertEqual(
            tuple(phase for _, _, phase in hb_executor.hb_line_schedule),
            ("expand", "expand", "plateau", "plateau", "contract", "contract"),
        )

        chain = build_chain(length=3)
        flat_report = single_layer_v1_support(chain)
        hb_report = hb_line_v1_support(chain)
        self.assertFalse(flat_report.supported)
        self.assertFalse(hb_report.supported)
        self.assertTrue(flat_report.reasons)
        self.assertTrue(hb_report.reasons)
        with self.assertRaisesRegex(UnsupportedPlanError, "single-layer.v1"):
            SpecializedExecutor(SettleGraph(chain), SINGLE_LAYER_V1)
        with self.assertRaisesRegex(UnsupportedPlanError, "hb-line.v1"):
            SpecializedExecutor(SettleGraph(chain), HB_LINE_V1)

    def test_predicates_reject_input_k_before_execution(self) -> None:
        flat = build_single_layer(receiver_count=2, k=2)
        region = dataclasses.replace(
            flat.regions[0],
            k_requested={
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "requested_k",
                "minimum": 1,
                "maximum": 2,
            },
        )
        flat = dataclasses.replace(flat, regions=(region,)).validate()
        decision = single_layer_v1_support(flat)
        self.assertFalse(decision.supported)
        self.assertTrue(any("fixed K" in reason for reason in decision.reasons))
        with self.assertRaises(UnsupportedPlanError):
            SpecializedExecutor(SettleGraph(flat), SINGLE_LAYER_V1)

        hb = build_small_hb()
        target = next(
            region for region in hb.regions if len(region.node_ids) == 2
        )
        replacement = dataclasses.replace(
            target,
            k_requested={
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "requested_k",
                "minimum": 1,
                "maximum": target.k_max,
            },
        )
        hb = dataclasses.replace(
            hb,
            regions=tuple(
                replacement if region.region_id == target.region_id else region
                for region in hb.regions
            ),
        ).validate()
        self.assertFalse(hb_line_v1_support(hb).supported)
        with self.assertRaises(UnsupportedPlanError):
            SpecializedExecutor(SettleGraph(hb), HB_LINE_V1)

    def test_predicates_reject_plan_valid_but_unimplemented_local_formulas(self) -> None:
        for base, predicate in (
            (
                build_single_layer(receiver_count=2, k=1),
                single_layer_v1_support,
            ),
            (build_small_hb(), hb_line_v1_support),
        ):
            with self.subTest(plan=base.plan_id):
                custom_node = dataclasses.replace(
                    base.nodes[0],
                    emit={"type": "custom", "formula": "return hidden"},
                )
                custom = dataclasses.replace(
                    base,
                    nodes=(custom_node, *base.nodes[1:]),
                ).validate()
                decision = predicate(custom)
                self.assertFalse(decision.supported)
                self.assertTrue(
                    any(
                        "reference capability rejected" in reason
                        for reason in decision.reasons
                    )
                )

    def test_wrapper_reuses_owner_parameters_without_a_second_namespace(self) -> None:
        model = SettleGraph(build_single_layer(receiver_count=2, k=1))
        before = {name: id(value) for name, value in model.named_parameters()}
        executor = SpecializedExecutor(model, SINGLE_LAYER_V1)
        self.assertIs(executor.model, model)
        self.assertTrue(executor.support_report().supported)
        self.assertIn(model.plan.canonical_hash(), executor.schedule_identity)
        self.assertEqual(executor.hb_line_schedule, ())
        self.assertEqual(
            before, {name: id(value) for name, value in model.named_parameters()}
        )
        self.assertFalse(isinstance(executor, torch.nn.Module))

    def test_decode_defaults_to_a_detached_chunk_boundary_and_allows_opt_out(
        self,
    ) -> None:
        model = SettleGraph(build_singleton()).to(dtype=torch.float64)
        executor = SpecializedExecutor(model, SINGLE_LAYER_V1)
        arguments = (
            torch.randn((1, model.plan.d_model), dtype=torch.float64),
            torch.ones((1,), dtype=torch.bool),
            ("sequence.a",),
            torch.zeros((1,), dtype=torch.int64),
        )
        with mock.patch.object(
            executor, "prefill", wraps=executor.prefill
        ) as prefill:
            executor.decode(*arguments)
            self.assertTrue(prefill.call_args.kwargs["detach_at_end"])
            executor.decode(*arguments, detach_at_end=False)
            self.assertFalse(prefill.call_args.kwargs["detach_at_end"])


class SpecializedEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.accepted = tuple(_accepted_cases())

    def test_development_corpus_acceptance_is_nonempty_and_predeclared(self) -> None:
        kinds = [kind for _, kind in self.accepted]
        self.assertIn(SINGLE_LAYER_V1, kinds)
        self.assertIn(HB_LINE_V1, kinds)
        self.assertGreaterEqual(len(self.accepted), 2)

    def test_all_statically_accepted_cases_match_output_state_stats_and_trace(self) -> None:
        for case, kind in self.accepted:
            with self.subTest(case=case.case_id, specialization=kind):
                model = _model(case)
                executor = SpecializedExecutor(model, kind)
                source = _hidden(case)
                reference = model.prefill(source.clone(), **_common(trace=True))
                specialized = executor.prefill(
                    source.clone(), **_common(trace=True)
                )
                _compare(reference.output, specialized.output, "output")
                _compare(reference.state, specialized.state, "state")
                _compare(
                    reference.balance_stats,
                    specialized.balance_stats,
                    "balance_stats",
                )
                _compare(reference.trace, specialized.trace, "trace")
                assert specialized.trace is not None
                validate_trace_invariants(
                    case.plan,
                    specialized.trace,
                    _EXECUTED,
                    tolerance=CPU_FLOAT64_TOLERANCE,
                )

    def test_all_statically_accepted_cases_match_vjp_connectedness_and_values(self) -> None:
        for case, kind in self.accepted:
            with self.subTest(case=case.case_id, specialization=kind):
                reference_model = _model(case)
                specialized_model = _model(case)
                specialized_model.load_state_dict(
                    reference_model.state_dict(), strict=True
                )
                source = _hidden(case)
                left_hidden = source.clone().requires_grad_(True)
                right_hidden = source.clone().requires_grad_(True)
                left = reference_model.prefill(left_hidden, **_common())
                right = SpecializedExecutor(
                    specialized_model, kind
                ).prefill(right_hidden, **_common())
                _compare(
                    _gradients(left, reference_model, left_hidden),
                    _gradients(right, specialized_model, right_hidden),
                    "vjp",
                )
        self._assert_single_layer_output_event_vjp_is_event_local()

    def _assert_single_layer_output_event_vjp_is_event_local(self) -> None:
        case = generate_core_v1_candidate_corpus()[35]
        self.assertEqual(
            case.case_id,
            "core-v1-candidate.ql-0035.single-layer-r8.k1",
        )
        self.assertEqual(case.plan.output_aggregate["type"], "node_softmax")
        self.assertTrue(single_layer_v1_support(case.plan).supported)
        source = _hidden(case)

        def execute_calls(
            executor: object,
            hidden: Tensor,
            boundaries: Tuple[Tuple[int, int], ...],
            *,
            decode: bool,
        ) -> Tuple[ExecutionResult, ...]:
            state: Optional[StateStore] = None
            results = []
            for start, end in boundaries:
                if decode:
                    self.assertEqual(end, start + 1)
                    method = (
                        executor.interpret_token
                        if isinstance(executor, SettleGraph)
                        else executor.decode
                    )
                    result = method(
                        hidden[:, start],
                        _EXECUTION_MASK[:, start],
                        _SEQUENCE_IDS,
                        _POSITIONS[:, start],
                        state=state,
                        routing_stats_mask=_ROUTING_MASK[:, start],
                        detach_at_end=False,
                        record_trace=True,
                    )
                else:
                    result = executor.prefill(
                        hidden[:, start:end],
                        _EXECUTION_MASK[:, start:end],
                        _SEQUENCE_IDS,
                        _POSITIONS[:, start:end],
                        state=state,
                        routing_stats_mask=_ROUTING_MASK[:, start:end],
                        detach_at_end=False,
                        record_trace=True,
                    )
                state = result.state
                results.append(result)
            return tuple(results)

        modes = (
            ("full", ((0, 3),), False),
            ("split-1", ((0, 1), (1, 3)), False),
            ("split-2", ((0, 2), (2, 3)), False),
            ("decode", ((0, 1), (1, 2), (2, 3)), True),
        )
        for mode, boundaries, decode in modes:
            reference_model = _model(case)
            packed_model = _model(case)
            specialized_model = _model(case)
            packed_model.load_state_dict(reference_model.state_dict(), strict=True)
            specialized_model.load_state_dict(
                reference_model.state_dict(), strict=True
            )
            executors = (
                reference_model,
                PackedSettleGraph(packed_model),
                SpecializedExecutor(specialized_model, SINGLE_LAYER_V1),
            )
            call_groups = tuple(
                execute_calls(
                    executor,
                    source.clone(),
                    boundaries,
                    decode=decode,
                )
                for executor in executors
            )
            score_groups = tuple(
                tuple(
                    (
                        node_id,
                        model.output_scores[safe_module_key(node_id)],
                    )
                    for node_id in case.plan.terminal_node_ids
                )
                for model in (
                    reference_model,
                    packed_model,
                    specialized_model,
                )
            )
            for call_index, results in enumerate(zip(*call_groups)):
                reference, packed, specialized = results
                for name, candidate in (
                    ("packed", packed),
                    (SINGLE_LAYER_V1, specialized),
                ):
                    compare_nested(
                        reference,
                        candidate,
                        tolerance=CPU_FLOAT64_TOLERANCE,
                        require_same_requires_grad=True,
                        path=f"{mode}[{call_index}].{name}",
                    ).require_pass()
                assert reference.trace is not None
                assert packed.trace is not None
                assert specialized.trace is not None
                for event_index, events in enumerate(
                    zip(
                        reference.trace.output_events,
                        packed.trace.output_events,
                        specialized.trace.output_events,
                    )
                ):
                    gradient_groups = []
                    for event, scores in zip(events, score_groups):
                        cotangent = torch.arange(
                            1,
                            event.output.numel() + 1,
                            dtype=event.output.dtype,
                        ).reshape_as(event.output)
                        gradient_groups.append(
                            torch.autograd.grad(
                                (event.output * cotangent).sum(),
                                tuple(parameter for _, parameter in scores),
                                allow_unused=True,
                                retain_graph=True,
                            )
                        )
                    reference_gradients, packed_gradients, specialized_gradients = (
                        gradient_groups
                    )
                    _compare(
                        reference_gradients,
                        packed_gradients,
                        f"{mode}[{call_index}].output[{event_index}].packed-vjp",
                    )
                    _compare(
                        reference_gradients,
                        specialized_gradients,
                        f"{mode}[{call_index}].output[{event_index}].specialized-vjp",
                    )
                    active_ids = {
                        node_id for node_id, _ in events[0].terminal_messages
                    }
                    self.assertEqual(len(active_ids), 1)
                    self.assertEqual(
                        tuple(
                            node_id
                            for (node_id, _), gradient in zip(
                                score_groups[0], reference_gradients
                            )
                            if gradient is not None
                        ),
                        tuple(sorted(active_ids)),
                    )

    def test_specialized_schedules_do_not_call_any_generic_eager_scheduler(self) -> None:
        for plan, kind in (
            (build_single_layer(receiver_count=2, k=1), SINGLE_LAYER_V1),
            (build_small_hb(), HB_LINE_V1),
        ):
            with self.subTest(specialization=kind):
                model = SettleGraph(plan).to(dtype=torch.float64)
                executor = SpecializedExecutor(model, kind)
                hidden = torch.randn((2, 3, plan.d_model), dtype=torch.float64)
                with (
                    mock.patch.object(
                        SettleGraph,
                        "interpret_token",
                        side_effect=AssertionError("generic token scheduler called"),
                    ),
                    mock.patch.object(
                        SettleGraph,
                        "prefill",
                        side_effect=AssertionError("generic prefill called"),
                    ),
                    mock.patch.object(
                        SettleGraph,
                        "prefill_region_major",
                        side_effect=AssertionError("generic region scheduler called"),
                    ),
                    mock.patch.object(
                        SettleGraph,
                        "_interpret_sample",
                        side_effect=AssertionError("generic sample scheduler called"),
                    ),
                ):
                    result = executor.prefill(
                        hidden,
                        _EXECUTION_MASK,
                        _SEQUENCE_IDS,
                        _POSITIONS,
                    )
                self.assertEqual(result.output.shape, hidden.shape)

    def test_forced_singleton_skips_selector_read_score_and_topk(self) -> None:
        model = SettleGraph(build_singleton()).to(dtype=torch.float64)
        executor = SpecializedExecutor(model, SINGLE_LAYER_V1)
        receiver = model.receiver("node.0")
        selector = model.selector("region.0")
        hidden = torch.randn((1, 2, 4), dtype=torch.float64)
        with (
            mock.patch.object(
                receiver,
                "selector_read",
                side_effect=AssertionError("selector Read must be absent"),
            ),
            mock.patch.object(
                selector,
                "forward",
                side_effect=AssertionError("Score must be absent"),
            ),
            mock.patch.object(
                specialized_module,
                "_batched_selector_read",
                side_effect=AssertionError("batched selector Read must be absent"),
            ),
            mock.patch.object(
                specialized_module,
                "_batched_score",
                side_effect=AssertionError("batched Score must be absent"),
            ),
            mock.patch.object(
                specialized_module,
                "_stable_topk_mask",
                side_effect=AssertionError("Top-K must be absent"),
            ),
        ):
            result = executor.prefill(
                hidden,
                torch.ones((1, 2), dtype=torch.bool),
                ("sequence.a",),
                torch.tensor([[0, 1]], dtype=torch.int64),
                record_trace=True,
            )
        assert result.trace is not None
        for event in result.trace.region_events:
            self.assertTrue(event.forced_active)
            self.assertIsNone(event.logits)
            self.assertIsNone(event.requested_k)
            self.assertIsNone(event.effective_k)
            self.assertIsNone(event.top_k_node_ids)
            torch.testing.assert_close(
                event.probabilities, torch.ones_like(event.probabilities)
            )
        for event in result.trace.node_events:
            self.assertIsNone(event.selector_read)
            self.assertIsNone(event.logit)

    def test_single_layer_mlp_fp32_prefill_preserves_eager_boundary_route(
        self,
    ) -> None:
        base = build_single_layer(receiver_count=2, k=1, d_model=7)
        region = dataclasses.replace(
            base.regions[0],
            score={
                "type": "mlp",
                "formula_id": "TEST-SCORE-MLP-V1",
                "hidden_dim": 11,
                "bias": True,
                "shared_parameters": False,
            },
        )
        plan = dataclasses.replace(base, regions=(region,)).validate()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            model = SettleGraph(plan).to(dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10_001)
        hidden = torch.randn(
            (1, 3, plan.d_model), generator=generator, dtype=torch.float32
        )

        # Make candidate 1 equal the implementation-shaped batched endpoint
        # of candidate 0 at event 1.  The eager per-event MLP evaluates
        # candidate 0 one ulp below this value on the exercised CPU path, so
        # changing either Linear boundary turns a strict winner into a tie.
        selector = model.selector(region.region_id)
        normalized = {
            node_id: model.receiver(node_id).normalize_input(hidden[0])
            for node_id in region.node_ids
        }
        readouts = torch.stack(
            [
                model.receiver(node_id).selector_read(
                    normalized[node_id], None
                )
                for node_id in region.node_ids
            ],
            dim=1,
        )
        first_key = safe_module_key(region.node_ids[0])
        first_batched = selector.output_layers[first_key](
            torch.nn.functional.silu(
                selector.hidden_layers[first_key](readouts[:, 0])
            )
        ).squeeze(-1)
        second_output = selector.output_layers[
            safe_module_key(region.node_ids[1])
        ]
        with torch.no_grad():
            second_output.weight.zero_()
            assert second_output.bias is not None
            second_output.bias.copy_(
                first_batched[1].detach().reshape_as(second_output.bias)
            )

        kwargs = {
            "execution_mask": torch.ones((1, 3), dtype=torch.bool),
            "sequence_ids": ("sequence.a",),
            "token_positions": torch.arange(3, dtype=torch.int64).reshape(1, 3),
            "detach_at_end": False,
            "record_trace": True,
        }
        expected = model.prefill(hidden, **kwargs)
        assert expected.trace is not None
        boundary = expected.trace.region_events[1]
        assert boundary.logits is not None
        self.assertEqual(boundary.top_k_node_ids, (region.node_ids[1],))
        self.assertLess(
            float(boundary.logits[0].detach()),
            float(boundary.logits[1].detach()),
        )

        for executor_name, executor in (
            ("packed", PackedSettleGraph(model)),
            ("single-layer.v1", SpecializedExecutor(model, SINGLE_LAYER_V1)),
        ):
            with self.subTest(executor=executor_name):
                actual = executor.prefill(hidden, **kwargs)
                assert actual.trace is not None
                self.assertEqual(
                    len(actual.trace.region_events),
                    len(expected.trace.region_events),
                )
                for expected_event, actual_event in zip(
                    expected.trace.region_events,
                    actual.trace.region_events,
                ):
                    torch.testing.assert_close(
                        actual_event.logits,
                        expected_event.logits,
                        rtol=0.0,
                        atol=0.0,
                    )
                    self.assertEqual(
                        actual_event.top_k_node_ids,
                        expected_event.top_k_node_ids,
                    )

    def test_hb_full_chunk_split_chunks_and_decode_are_equivalent(self) -> None:
        case, _ = next(
            item for item in self.accepted if item[1] == HB_LINE_V1
        )
        model = _model(case)
        executor = SpecializedExecutor(model, HB_LINE_V1)
        hidden = _hidden(case)
        full = executor.prefill(hidden.clone(), **_common())

        first = executor.prefill(
            hidden[:, :2],
            _EXECUTION_MASK[:, :2],
            _SEQUENCE_IDS,
            _POSITIONS[:, :2],
            routing_stats_mask=_ROUTING_MASK[:, :2],
            detach_at_end=False,
        )
        second = executor.prefill(
            hidden[:, 2:],
            _EXECUTION_MASK[:, 2:],
            _SEQUENCE_IDS,
            _POSITIONS[:, 2:],
            state=first.state,
            routing_stats_mask=_ROUTING_MASK[:, 2:],
            detach_at_end=False,
        )
        _compare(
            full.output,
            torch.cat((first.output, second.output), dim=1),
            "split.output",
        )
        _compare(full.state, second.state, "split.state")
        _compare(
            full.balance_stats,
            first.balance_stats.merge(second.balance_stats),
            "split.balance",
        )

        decode_state: Optional[StateStore] = None
        decode_outputs = []
        decode_balance = None
        for column in range(hidden.shape[1]):
            step = executor.decode(
                hidden[:, column],
                _EXECUTION_MASK[:, column],
                _SEQUENCE_IDS,
                _POSITIONS[:, column],
                state=decode_state,
                routing_stats_mask=_ROUTING_MASK[:, column],
                detach_at_end=False,
            )
            decode_state = step.state
            decode_outputs.append(step.output)
            decode_balance = (
                step.balance_stats
                if decode_balance is None
                else decode_balance.merge(step.balance_stats)
            )
        _compare(full.output, torch.stack(decode_outputs, dim=1), "decode.output")
        _compare(full.state, decode_state, "decode.state")
        _compare(full.balance_stats, decode_balance, "decode.balance")

    def test_decode_default_detaches_and_explicit_false_keeps_cross_call_vjp(
        self,
    ) -> None:
        case, _ = next(
            item for item in self.accepted if item[1] == HB_LINE_V1
        )

        def execute_pair(
            api: str, detach_at_end: Optional[bool]
        ) -> Tuple[ExecutionResult, ExecutionResult, Optional[Tensor]]:
            model = _model(case)
            executor = SpecializedExecutor(model, HB_LINE_V1)
            hidden = _hidden(case)
            first_hidden = hidden[:, 0].clone().requires_grad_(True)
            first_kwargs = (
                {} if detach_at_end is None else {"detach_at_end": detach_at_end}
            )
            if api == "decode":
                first = executor.decode(
                    first_hidden,
                    _EXECUTION_MASK[:, 0],
                    _SEQUENCE_IDS,
                    _POSITIONS[:, 0],
                    **first_kwargs,
                )
                second = executor.decode(
                    hidden[:, 1],
                    _EXECUTION_MASK[:, 1],
                    _SEQUENCE_IDS,
                    _POSITIONS[:, 1],
                    state=first.state,
                    detach_at_end=False,
                )
            else:
                self.assertEqual(api, "prefill")
                first = executor.prefill(
                    first_hidden.unsqueeze(1),
                    _EXECUTION_MASK[:, :1],
                    _SEQUENCE_IDS,
                    _POSITIONS[:, :1],
                    **first_kwargs,
                )
                second = executor.prefill(
                    hidden[:, 1:2],
                    _EXECUTION_MASK[:, 1:2],
                    _SEQUENCE_IDS,
                    _POSITIONS[:, 1:2],
                    state=first.state,
                    detach_at_end=False,
                )
            objective = second.output.square().sum() + _state_objective(
                second.state, second.output
            )
            gradient = torch.autograd.grad(
                objective, first_hidden, allow_unused=True
            )[0]
            return first, second, gradient

        for api in ("prefill", "decode"):
            with self.subTest(api=api):
                default_first, default_second, default_gradient = execute_pair(
                    api, None
                )
                live_first, live_second, live_gradient = execute_pair(api, False)
                _compare(default_first.output, live_first.output, f"{api}.first.output")
                _compare(default_first.state, live_first.state, f"{api}.first.state")
                _compare(
                    default_second.output,
                    live_second.output,
                    f"{api}.second.output",
                )
                _compare(
                    default_second.state,
                    live_second.state,
                    f"{api}.second.state",
                )
                self.assertIsNone(default_gradient)
                self.assertIsNotNone(live_gradient)
                assert live_gradient is not None
                self.assertGreater(float(live_gradient.abs().sum()), 0.0)

        model = _model(case)
        executor = SpecializedExecutor(model, HB_LINE_V1)
        hidden = _hidden(case)[:1, :1]
        state = StateStore(next_position={"sequence.a": 0})
        state_before = StateStore(next_position={"sequence.a": 0})
        invalid_values = (None, 0, 1, torch.tensor(True))
        for api in ("prefill", "decode"):
            for invalid in invalid_values:
                with self.subTest(api=api, invalid=repr(invalid)), mock.patch.object(
                    specialized_module, "_prepare_prefill"
                ) as prepare:
                    with self.assertRaisesRegex(
                        ExecutionContractError,
                        "detach_at_end must be a boolean",
                    ):
                        if api == "prefill":
                            executor.prefill(
                                hidden,
                                torch.ones((1, 1), dtype=torch.bool),
                                ("sequence.a",),
                                torch.zeros((1, 1), dtype=torch.int64),
                                state=state,
                                detach_at_end=invalid,  # type: ignore[arg-type]
                            )
                        else:
                            executor.decode(
                                hidden[:, 0],
                                torch.ones((1,), dtype=torch.bool),
                                ("sequence.a",),
                                torch.zeros((1,), dtype=torch.int64),
                                state=state,
                                detach_at_end=invalid,  # type: ignore[arg-type]
                            )
                    prepare.assert_not_called()
                    _compare(state_before, state, f"{api}.invalid.entry-state")

    def test_empty_tail_has_no_events_and_preserves_published_state(self) -> None:
        for plan, kind in (
            (build_single_layer(receiver_count=2, k=1), SINGLE_LAYER_V1),
            (build_small_hb(), HB_LINE_V1),
        ):
            with self.subTest(specialization=kind):
                model = SettleGraph(plan).to(dtype=torch.float64)
                executor = SpecializedExecutor(model, kind)
                prefix = executor.decode(
                    torch.randn((2, plan.d_model), dtype=torch.float64),
                    torch.ones((2,), dtype=torch.bool),
                    _SEQUENCE_IDS,
                    torch.zeros((2,), dtype=torch.int64),
                    detach_at_end=False,
                )
                empty = executor.prefill(
                    torch.empty((2, 0, plan.d_model), dtype=torch.float64),
                    torch.empty((2, 0), dtype=torch.bool),
                    _SEQUENCE_IDS,
                    torch.empty((2, 0), dtype=torch.int64),
                    state=prefix.state,
                    detach_at_end=False,
                    record_trace=True,
                )
                self.assertEqual(empty.output.shape, (2, 0, plan.d_model))
                _compare(prefix.state, empty.state, "empty.state")
                self.assertFalse(empty.balance_stats.regions)
                assert empty.trace is not None
                self.assertEqual(empty.trace.node_events, ())
                self.assertEqual(empty.trace.region_events, ())
                self.assertEqual(empty.trace.output_events, ())

    def test_prefill_preflight_failures_match_stable_envelopes(self) -> None:
        plan = build_singleton()
        model = SettleGraph(plan).to(dtype=torch.float64)
        executors = (
            ("eager", model.prefill),
            ("packed", PackedSettleGraph(model).prefill),
            (
                SINGLE_LAYER_V1,
                SpecializedExecutor(model, SINGLE_LAYER_V1).prefill,
            ),
        )
        hidden = torch.randn((2, 2, plan.d_model), dtype=torch.float64)
        execution_mask = torch.tensor(
            [[True, True], [True, False]], dtype=torch.bool
        )
        positions = torch.tensor([[0, 1], [0, 99]], dtype=torch.int64)
        base = {
            "hidden": hidden,
            "execution_mask": execution_mask,
            "sequence_ids": ("sequence.a", "sequence.b"),
            "token_positions": positions,
        }
        cases = (
            (
                "lm-mask-outside-execution",
                {"lm_target_mask": torch.ones((2, 2), dtype=torch.bool)},
                "input.mask",
            ),
            (
                "routing-mask-outside-execution",
                {"routing_stats_mask": torch.ones((2, 2), dtype=torch.bool)},
                "input.mask",
            ),
            (
                "duplicate-sequence-id",
                {
                    "sequence_ids": ("sequence.a", "sequence.a"),
                    "token_positions": torch.tensor(
                        [[0, 1], [0, 99]], dtype=torch.int64
                    ),
                },
                "input.schema",
            ),
            (
                "position-replay",
                {
                    "hidden": hidden[:1],
                    "execution_mask": torch.ones((1, 2), dtype=torch.bool),
                    "sequence_ids": ("sequence.a",),
                    "token_positions": torch.tensor(
                        [[1, 1]], dtype=torch.int64
                    ),
                    "state": StateStore(next_position={"sequence.a": 1}),
                },
                "input.position",
            ),
            (
                "position-skip",
                {
                    "hidden": hidden[:1],
                    "execution_mask": torch.ones((1, 2), dtype=torch.bool),
                    "sequence_ids": ("sequence.a",),
                    "token_positions": torch.tensor(
                        [[1, 3]], dtype=torch.int64
                    ),
                    "state": StateStore(next_position={"sequence.a": 1}),
                },
                "input.position",
            ),
        )
        for case_name, overrides, code in cases:
            arguments = dict(base)
            arguments.update(overrides)
            expected = FailureEnvelope.create("input", code)
            for executor_name, execute in executors:
                with self.subTest(case=case_name, executor=executor_name):
                    captured = capture_execution(
                        lambda execute=execute, arguments=arguments: execute(
                            **arguments
                        ),
                        codes=code,
                    )
                    self.assertIsInstance(captured, ExecutionFailed)
                    assert isinstance(captured, ExecutionFailed)
                    compare_failure_envelopes(expected, captured.envelope)

    def test_invalid_state_shape_and_owner_alias_match_stable_envelopes(
        self,
    ) -> None:
        case, _ = next(
            (case, kind)
            for case, kind in self.accepted
            if kind == HB_LINE_V1
            and any(node.update["type"] != "none" for node in case.plan.nodes)
        )
        model = _model(case)
        executors = (
            ("eager", model.prefill),
            ("packed", PackedSettleGraph(model).prefill),
            (HB_LINE_V1, SpecializedExecutor(model, HB_LINE_V1).prefill),
        )
        tensor_nodes = tuple(
            node
            for node in case.plan.nodes
            if node.update["type"] not in {"none", "attention_window"}
        )
        wrong_shape_node = tensor_nodes[0]
        wrong_shape_state = StateStore(
            values={
                ("sequence.a", wrong_shape_node.node_id): torch.zeros(
                    wrong_shape_node.state_shape + (1,), dtype=torch.float64
                )
            },
            next_position={"sequence.a": 0},
        )
        first, second = next(
            (left, right)
            for index, left in enumerate(tensor_nodes)
            for right in tensor_nodes[index + 1 :]
            if left.state_shape == right.state_shape
        )
        element_count = math.prod(first.state_shape)
        backing = torch.arange(
            2 * element_count, dtype=torch.float64
        )
        owner_alias_state = StateStore(
            values={
                ("sequence.a", first.node_id): backing[:element_count].reshape(
                    first.state_shape
                ),
                ("sequence.a", second.node_id): backing[element_count:].reshape(
                    second.state_shape
                ),
            },
            next_position={"sequence.a": 0},
        )
        cases = (
            ("wrong-shape", wrong_shape_state, "state.schema"),
            ("owner-alias", owner_alias_state, "state.owner_alias"),
        )
        for case_name, state, code in cases:
            state_before = StateStore(
                values={key: value.clone() for key, value in state.values.items()},
                selector_history=dict(state.selector_history),
                next_position=dict(state.next_position),
            )
            expected = FailureEnvelope.create("state", code)
            arguments = dict(_common())
            arguments["state"] = state
            for executor_name, execute in executors:
                with self.subTest(case=case_name, executor=executor_name):
                    captured = capture_execution(
                        lambda execute=execute: execute(
                            _hidden(case), **arguments
                        ),
                        codes=code,
                    )
                    self.assertIsInstance(captured, ExecutionFailed)
                    assert isinstance(captured, ExecutionFailed)
                    compare_failure_envelopes(expected, captured.envelope)
                    _compare(state_before, state, f"{case_name}.entry-state")

    def test_late_empty_terminal_failure_preserves_entry_state(self) -> None:
        case, _ = next(
            (case, kind)
            for case, kind in self.accepted
            if kind == HB_LINE_V1
            and any(node.update["type"] != "none" for node in case.plan.nodes)
        )
        model = _model(case)
        hidden = _hidden(case)
        prefix = model.prefill(
            hidden[:, :1],
            _EXECUTION_MASK[:, :1],
            _SEQUENCE_IDS,
            _POSITIONS[:, :1],
            detach_at_end=False,
        )
        entry_state = prefix.state
        entry_snapshot = StateStore(
            values={
                key: (
                    value.clone()
                    if isinstance(value, Tensor)
                    else AttentionState(
                        value.positions.clone(),
                        value.keys.clone(),
                        value.values.clone(),
                    )
                )
                for key, value in entry_state.values.items()
            },
            selector_history=dict(entry_state.selector_history),
            next_position=dict(entry_state.next_position),
        )
        arguments = {
            "hidden": hidden[:, 1:],
            "execution_mask": _EXECUTION_MASK[:, 1:],
            "sequence_ids": _SEQUENCE_IDS,
            "token_positions": _POSITIONS[:, 1:],
            "state": entry_state,
            "detach_at_end": False,
        }

        def select_nothing(scores: Tensor, k: int) -> Tensor:
            del k
            return torch.zeros_like(scores, dtype=torch.bool)

        operations = (
            (
                "eager",
                model.prefill,
                mock.patch.object(
                    engine_module,
                    "deterministic_topk_mask",
                    side_effect=select_nothing,
                ),
            ),
            (
                "packed",
                PackedSettleGraph(model).prefill,
                mock.patch.object(
                    packed_module,
                    "_aggregate_terminals",
                    side_effect=DynamicReachabilityError(
                        "injected late empty-terminal failure"
                    ),
                ),
            ),
            (
                HB_LINE_V1,
                SpecializedExecutor(model, HB_LINE_V1).prefill,
                mock.patch.object(
                    specialized_module,
                    "_stable_topk_mask",
                    side_effect=select_nothing,
                ),
            ),
        )
        expected = FailureEnvelope.create(
            "execution", "execution.empty_terminal"
        )
        for executor_name, execute, injection in operations:
            with self.subTest(executor=executor_name), injection as injected:
                captured = capture_execution(
                    lambda execute=execute: execute(**arguments)
                )
                self.assertIsInstance(captured, ExecutionFailed)
                assert isinstance(captured, ExecutionFailed)
                compare_failure_envelopes(expected, captured.envelope)
                self.assertTrue(injected.called)
                _compare(
                    entry_snapshot,
                    entry_state,
                    f"{executor_name}.late-failure.entry-state",
                )

if __name__ == "__main__":
    unittest.main()
