from __future__ import annotations

import dataclasses
import unittest
from unittest import mock
from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor

import tide.specialized as specialized_module

from tide.builders import (
    build_chain,
    build_single_layer,
    build_singleton,
    build_small_hb,
)
from tide.engine import ExecutionResult, SettleGraph, StateStore, UnsupportedPlanError
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    compare_nested,
    validate_trace_invariants,
)
from tide.generators import PlanCorpusCase, generate_plan_corpus
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

        def first_input_gradient(detach_at_end: Optional[bool]):
            model = _model(case)
            executor = SpecializedExecutor(model, HB_LINE_V1)
            hidden = _hidden(case)
            first_hidden = hidden[:, 0].clone().requires_grad_(True)
            first_kwargs = (
                {} if detach_at_end is None else {"detach_at_end": detach_at_end}
            )
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
            objective = second.output.square().sum() + _state_objective(
                second.state, hidden[:, 1]
            )
            return torch.autograd.grad(
                objective, first_hidden, allow_unused=True
            )[0]

        self.assertIsNone(first_input_gradient(None))
        connected = first_input_gradient(False)
        self.assertIsNotNone(connected)
        assert connected is not None
        self.assertGreater(float(connected.abs().sum()), 0.0)

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

if __name__ == "__main__":
    unittest.main()
