from __future__ import annotations

import dataclasses
import math
import unittest
from unittest import mock

import torch
from torch import Tensor
import torch.nn.functional as F

from tide.builders import build_diamond, build_single_layer, build_singleton
from tide.engine import ExecutionResult, SettleGraph, StateStore, UnsupportedPlanError
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    compare_nested,
    validate_trace_invariants,
)
from tide.generators import (
    PlanCorpusCase,
    generate_core_v1_candidate_corpus,
    generate_plan_corpus,
)
from tide.ops import AttentionState, safe_module_key
from tide.packed import PackedSettleGraph, inspect_packed_support
from tide.plan import bind_dtypes


_BATCH = 2
_LENGTH = 3
_SEQUENCE_IDS = ("packed.sequence.a", "packed.sequence.b")
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
    ("packed.sequence.a", 0),
    ("packed.sequence.a", 1),
    ("packed.sequence.a", 2),
    ("packed.sequence.b", 0),
    ("packed.sequence.b", 1),
)


def _tolerance(dtype: torch.dtype):
    return (
        CPU_FLOAT64_TOLERANCE
        if dtype == torch.float64
        else SAME_BACKEND_FLOAT32_TOLERANCE
    )


def _accepted_cases() -> tuple[PlanCorpusCase, ...]:
    return tuple(
        case for case in generate_plan_corpus()
        if inspect_packed_support(case.plan).supported
    )


def _model(case: PlanCorpusCase, dtype: torch.dtype) -> SettleGraph:
    dtype_name = "float64" if dtype == torch.float64 else "float32"
    typed = bind_dtypes(
        case.plan,
        hidden=dtype_name,
        parameter=dtype_name,
        state=dtype_name,
        readout=dtype_name,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case.parameter_seed)
        model = SettleGraph(typed).to(device="cpu", dtype=dtype)
    model.eval()
    return model


def _hidden(case: PlanCorpusCase, dtype: torch.dtype) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed)
    return torch.randn(
        (_BATCH, _LENGTH, case.plan.d_model),
        generator=generator,
        dtype=torch.float64,
    ).to(dtype=dtype)


def _call_arguments() -> dict[str, object]:
    return {
        "execution_mask": _EXECUTION_MASK,
        "sequence_ids": _SEQUENCE_IDS,
        "token_positions": _TOKEN_POSITIONS,
        "lm_target_mask": _LM_TARGET_MASK,
        "routing_stats_mask": _ROUTING_STATS_MASK,
        "detach_at_end": False,
    }


def _state_objective(store: StateStore, reference: Tensor) -> Tensor:
    result = reference.sum() * 0.0
    for key in sorted(store.values):
        value = store.values[key]
        if isinstance(value, Tensor):
            result = result + value.sum()
        elif isinstance(value, AttentionState):
            result = result + value.keys.sum() + value.values.sum()
    return result


def _gradient_record(model: SettleGraph, hidden: Tensor) -> dict[str, object]:
    return {
        "hidden": hidden.grad,
        "parameters": {
            name: parameter.grad for name, parameter in model.named_parameters()
        },
    }


def _competitive_sd_pre_case(state_kind: str) -> PlanCorpusCase:
    """Specialize one mixed HB fixture to a competitive SD/pre state family."""

    base = next(
        case for case in generate_plan_corpus()
        if case.case_id == "corpus.004.small-hb.base-r2"
    )
    region_id = {
        "ema": "region.l0002.r0001",
        "gdn": "region.l0003.r0000",
        "attention_window": "region.l0003.r0001",
    }[state_kind]
    nodes = []
    for node in base.plan.nodes:
        if node.region_id == region_id and state_kind == "gdn":
            node = dataclasses.replace(
                node,
                selector_read={
                    "type": "content_state_summary_linear",
                    "formula_id": "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
                    "out_dim": node.selector_read_shape[0],
                    "output_shape": list(node.selector_read_shape),
                },
            )
        nodes.append(node)
    regions = tuple(
        dataclasses.replace(region, profile="SD", selector_timing="pre")
        if region.region_id == region_id
        else region
        for region in base.plan.regions
    )
    plan = dataclasses.replace(
        base.plan, nodes=tuple(nodes), regions=regions
    ).validate()
    return dataclasses.replace(
        base,
        case_id=f"packed.sd-pre.{state_kind}",
        plan=plan,
        features=frozenset((*base.features, f"state:{state_kind}", "profile:SD/pre")),
    )


def _sd_pre_route_boundary_plan(
    *,
    d_model: int,
    state_shape: tuple[int, ...],
    update: dict[str, object],
    summary: bool = False,
):
    """Build the common two-node carrier for exact-route regressions."""

    # Reuse the public builder's canonical single-layer IDs and topology.
    plan = build_single_layer(receiver_count=2, k=1, d_model=d_model)
    nodes = tuple(
        dataclasses.replace(
            node,
            state_shape=state_shape,
            state_owner=node.node_id,
            update=update,
            selector_read_shape=(1,),
            selector_read={
                "type": (
                    "content_state_summary_linear"
                    if summary
                    else "content_state_linear"
                ),
                "formula_id": (
                    "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1"
                    if summary
                    else "TEST-READ-PROJ-V1"
                ),
                "out_dim": 1,
                "output_shape": [1],
            },
            ffn_read={
                "type": "zero",
                "formula_id": "read.ffn.zero.v1",
                "output_shape": [d_model],
            },
            node_compute={
                "type": "identity",
                "formula_id": "node.identity.v1",
                "output_shape": [d_model],
            },
            emit={
                "type": "hard",
                "formula_id": "emit.hard.v1",
                "output_shape": [d_model],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0],
        profile="SD",
        selector_timing="pre",
        score={
            "type": "read_sum",
            "formula_id": "score.read-sum.v1",
            "context_dim": 0,
        },
    )
    return dataclasses.replace(plan, nodes=nodes, regions=(region,)).validate()


def _trace_routes(result: ExecutionResult) -> tuple[tuple[str, ...], ...]:
    assert result.trace is not None
    return tuple(event.active_node_ids for event in result.trace.region_events)


def _state_boundary_probe_plan(state_kind: str):
    if state_kind == "ema":
        return _sd_pre_route_boundary_plan(
            d_model=3,
            state_shape=(1,),
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 1,
                "decay": 0.7,
                "learnable_decay": False,
                "state_shape": [1],
            },
        )
    if state_kind == "gdn":
        return _sd_pre_route_boundary_plan(
            d_model=3,
            state_shape=(3, 1),
            update={
                "type": "gdn",
                "formula_id": "state.gdn.v1",
                "key_dim": 3,
                "value_dim": 1,
                "norm_eps": 1e-12,
                "state_shape": [3, 1],
            },
        )
    if state_kind == "attention_window":
        return _sd_pre_route_boundary_plan(
            d_model=2,
            state_shape=(3, 3, 4),
            update={
                "type": "attention_window",
                "formula_id": "state.attention-window.v1",
                "key_dim": 3,
                "value_dim": 4,
                "window": 3,
                "norm_eps": 1e-12,
                "state_shape": [3, 3, 4],
            },
            summary=True,
        )
    raise AssertionError(state_kind)


def _differentiable_probe_state(
    state_kind: str, dtype: torch.dtype
) -> tuple[StateStore, tuple[Tensor, ...], tuple[str, ...]]:
    values = {}
    leaves = []
    names = []
    generator = torch.Generator(device="cpu").manual_seed(7721)
    for node_id in ("node.0000", "node.0001"):
        key = ("state-probe", node_id)
        if state_kind == "ema":
            state = torch.randn((1,), generator=generator, dtype=dtype).requires_grad_()
            values[key] = state
            leaves.append(state)
            names.append(f"initial:{node_id}:tensor")
        elif state_kind == "gdn":
            state = torch.randn(
                (3, 1), generator=generator, dtype=dtype
            ).requires_grad_()
            values[key] = state
            leaves.append(state)
            names.append(f"initial:{node_id}:tensor")
        else:
            keys = torch.randn(
                (2, 3), generator=generator, dtype=dtype
            ).requires_grad_()
            state_values = torch.randn(
                (2, 4), generator=generator, dtype=dtype
            ).requires_grad_()
            values[key] = AttentionState(
                torch.tensor([0, 1]), keys, state_values
            )
            leaves.extend((keys, state_values))
            names.extend(
                (
                    f"initial:{node_id}:keys",
                    f"initial:{node_id}:values",
                )
            )
    return (
        StateStore(values, next_position={"state-probe": 2}),
        tuple(leaves),
        tuple(names),
    )


def _clone_differentiable_probe_state(
    source: StateStore,
) -> tuple[StateStore, tuple[Tensor, ...]]:
    values = {}
    leaves = []
    for key, state in sorted(source.values.items()):
        if isinstance(state, Tensor):
            clone = state.detach().clone().requires_grad_()
            values[key] = clone
            leaves.append(clone)
        else:
            keys = state.keys.detach().clone().requires_grad_()
            state_values = state.values.detach().clone().requires_grad_()
            values[key] = AttentionState(state.positions.clone(), keys, state_values)
            leaves.extend((keys, state_values))
    return (
        StateStore(values, next_position=dict(source.next_position)),
        tuple(leaves),
    )


def _public_vjp_objectives(
    result: ExecutionResult, prefix: str = ""
) -> tuple[tuple[str, Tensor], ...]:
    objectives = [(prefix + "output", result.output.sum())]
    for key, state in sorted(result.state.values.items()):
        if isinstance(state, Tensor) and state.requires_grad:
            objectives.append((prefix + f"state:{key}:tensor", state.sum()))
        elif isinstance(state, AttentionState):
            if state.keys.requires_grad:
                objectives.append((prefix + f"state:{key}:keys", state.keys.sum()))
            if state.values.requires_grad:
                objectives.append(
                    (prefix + f"state:{key}:values", state.values.sum())
                )
    for region_id, stats in result.balance_stats.regions.items():
        if stats.soft_sum.requires_grad:
            objectives.append(
                (prefix + f"balance:{region_id}", stats.soft_sum.sum())
            )
    if result.trace is not None:
        trace_objectives = []
        for index, event in enumerate(result.trace.region_events):
            if event.logits is not None and event.logits.requires_grad:
                trace_objectives.append(
                    (prefix + f"logit:{index}", event.logits.sum())
                )
            if (
                event.probabilities is not None
                and event.probabilities.requires_grad
            ):
                trace_objectives.append(
                    (
                        prefix + f"probability:{index}",
                        event.probabilities.sum(),
                    )
                )
        if trace_objectives:
            objectives.extend((trace_objectives[0], trace_objectives[-1]))
    return tuple(objectives)


def _long_inputs(
    case: PlanCorpusCase, length: int, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.input_seed ^ (length << 8))
    hidden = torch.randn(
        (_BATCH, length, case.plan.d_model),
        generator=generator,
        dtype=torch.float64,
    ).to(dtype)
    mask = torch.ones((_BATCH, length), dtype=torch.bool)
    if length >= 4:
        mask[0, 2] = False
        mask[1, 1] = False
    if length >= 7:
        mask[1, 5] = False
    positions = mask.to(torch.int64).cumsum(dim=1) - 1
    positions = positions.masked_fill(~mask, 10_000)
    return hidden, mask, positions


def _assert_trace_routes_rederive_from_logits(
    test: unittest.TestCase, result: ExecutionResult
) -> None:
    assert result.trace is not None
    for event in result.trace.region_events:
        if event.logits is None:
            continue
        assert event.effective_k is not None
        order = sorted(
            range(len(event.candidate_node_ids)),
            key=lambda index: (-float(event.logits[index].item()), index),
        )
        expected = tuple(
            event.candidate_node_ids[index]
            for index in sorted(order[: event.effective_k])
        )
        test.assertEqual(event.active_node_ids, expected)
        test.assertEqual(event.top_k_node_ids, expected)


class PackedSupportTests(unittest.TestCase):
    def test_static_support_partition_is_stable_and_no_fallback_exists(self) -> None:
        corpus = generate_plan_corpus()
        accepted = tuple(case for case in corpus if inspect_packed_support(case.plan).supported)
        rejected = tuple(case for case in corpus if not inspect_packed_support(case.plan).supported)
        self.assertEqual(len(accepted), 35)
        self.assertEqual(len(rejected), 13)
        self.assertTrue(
            all(
                all(region.k_requested["type"] == "fixed" for region in case.plan.regions)
                for case in accepted
            )
        )
        issue_codes = {
            issue.code
            for case in rejected
            for issue in inspect_packed_support(case.plan).issues
        }
        self.assertEqual(
            issue_codes,
            {"packed.input-k-extension"},
        )
        for case in rejected:
            with self.subTest(case=case.case_id):
                with self.assertRaises(UnsupportedPlanError):
                    PackedSettleGraph(SettleGraph(case.plan))

    def test_schedule_identity_and_parameter_owner_are_stable(self) -> None:
        case = _accepted_cases()[0]
        model = _model(case, torch.float64)
        first = PackedSettleGraph(model)
        second = PackedSettleGraph(model)
        self.assertEqual(first.schedule_identity, second.schedule_identity)
        self.assertEqual(
            tuple(model.named_parameters()), tuple(first.named_parameters())
        )
        self.assertEqual(set(model.state_dict()), set(first.state_dict()))

    def test_reference_norm_gaps_are_rejected_by_the_static_predicate(self) -> None:
        base = build_singleton(d_model=2)
        custom_norm = {"type": "custom", "formula": "return x", "eps": 1e-6}
        for field in ("input_norm", "ffn_norm"):
            with self.subTest(field=field):
                node = dataclasses.replace(
                    base.nodes[0], **{field: custom_norm}
                )
                plan = dataclasses.replace(base, nodes=(node,)).validate()
                report = inspect_packed_support(plan)
                self.assertFalse(report.supported)
                self.assertIn(
                    "packed.reference-capability",
                    {issue.code for issue in report.issues},
                )
                with self.assertRaises(UnsupportedPlanError):
                    SettleGraph(plan)


class PackedForwardTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = _accepted_cases()

    def test_all_supported_development_cases_match_forward_state_balance_and_trace(self) -> None:
        for dtype in (torch.float64, torch.float32):
            tolerance = _tolerance(dtype)
            for case in self.cases:
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    model = _model(case, dtype)
                    packed = PackedSettleGraph(model)
                    hidden = _hidden(case, dtype)
                    reference = model.prefill(
                        hidden, record_trace=True, **_call_arguments()
                    )
                    actual = packed.prefill(
                        hidden, record_trace=True, **_call_arguments()
                    )
                    compare_nested(
                        reference, actual, tolerance=tolerance
                    ).require_pass()
                    assert actual.trace is not None
                    validate_trace_invariants(
                        case.plan,
                        actual.trace,
                        _EXECUTED_TOKENS,
                        tolerance=tolerance,
                    )
                    _assert_trace_routes_rederive_from_logits(self, actual)
                    profile = packed.last_profile
                    self.assertIsNotNone(profile)
                    assert profile is not None
                    self.assertEqual(profile.python_token_hot_loops, 0)
                    self.assertEqual(profile.python_batch_row_hot_loops, 0)
                    self.assertEqual(profile.python_node_event_hot_loops, 0)
                    self.assertEqual(profile.eager_scheduler_calls, 0)
                    self.assertTrue(profile.trace_materialized)

    def test_forced_singleton_trace_has_no_selector_or_topk_artifacts(self) -> None:
        case = next(
            case for case in self.cases
            if case.motif == "singleton"
        )
        model = _model(case, torch.float64)
        result = PackedSettleGraph(model).prefill(
            _hidden(case, torch.float64), record_trace=True, **_call_arguments()
        )
        assert result.trace is not None
        for event in result.trace.region_events:
            self.assertTrue(event.forced_active)
            self.assertIsNone(event.logits)
            self.assertIsNone(event.requested_k)
            self.assertIsNone(event.effective_k)
            self.assertIsNone(event.top_k_node_ids)
            torch.testing.assert_close(event.probabilities, torch.ones_like(event.probabilities))
        self.assertTrue(all(event.selector_read is None for event in result.trace.node_events))

    def test_instrumentation_proves_no_eager_call_through(self) -> None:
        case = next(case for case in self.cases if case.motif == "diamond")
        model = _model(case, torch.float64)
        packed = PackedSettleGraph(model)
        failure = AssertionError("an eager scheduler was called")
        with (
            mock.patch.object(model, "prefill", side_effect=failure),
            mock.patch.object(model, "prefill_region_major", side_effect=failure),
            mock.patch.object(model, "interpret_token", side_effect=failure),
            mock.patch.object(model, "_interpret_sample", side_effect=failure),
        ):
            result = packed.prefill(_hidden(case, torch.float64), **_call_arguments())
        self.assertEqual(result.output.shape, (_BATCH, _LENGTH, case.plan.d_model))
        assert packed.last_profile is not None
        self.assertEqual(packed.last_profile.eager_scheduler_calls, 0)

    def test_sd_pre_exact_tie_near_boundary_and_unreached_forced_region(self) -> None:
        base = next(
            case for case in generate_plan_corpus()
            if case.case_id == "corpus.001.unequal-path.base-r0"
        )
        epsilon = torch.finfo(torch.float64).eps
        for label, values, expected in (
            (
                "exact-tie",
                {"node.long.0": 0.0, "node.short": 0.0},
                "node.long.0",
            ),
            (
                "near-boundary-short",
                {"node.long.0": 0.0, "node.short": epsilon},
                "node.short",
            ),
        ):
            with self.subTest(label=label):
                regions = tuple(
                    dataclasses.replace(
                        region,
                        k_max=1,
                        k_requested={
                            "type": "fixed",
                            "formula_id": "k.fixed.v1",
                            "value": 1,
                        },
                        score={
                            "type": "fixed",
                            "formula_id": "score.fixed-by-node.v1",
                            "values_by_node": values,
                        },
                    )
                    if region.region_id == "region.split"
                    else region
                    for region in base.plan.regions
                )
                case = dataclasses.replace(
                    base,
                    case_id=f"packed.sd-pre.{label}",
                    plan=dataclasses.replace(base.plan, regions=regions).validate(),
                )
                model = _model(case, torch.float64)
                hidden = _hidden(case, torch.float64)
                reference = model.prefill(
                    hidden, record_trace=True, **_call_arguments()
                )
                actual = PackedSettleGraph(model).prefill(
                    hidden, record_trace=True, **_call_arguments()
                )
                compare_nested(
                    reference, actual, tolerance=CPU_FLOAT64_TOLERANCE
                ).require_pass()
                _assert_trace_routes_rederive_from_logits(self, actual)
                assert actual.trace is not None
                split_events = tuple(
                    event for event in actual.trace.region_events
                    if event.region_id == "region.split"
                )
                self.assertTrue(split_events)
                self.assertTrue(
                    all(event.active_node_ids == (expected,) for event in split_events)
                )
                if label == "near-boundary-short":
                    long_events = tuple(
                        event for event in actual.trace.region_events
                        if event.region_id == "region.long"
                    )
                    self.assertTrue(long_events)
                    for event in long_events:
                        self.assertEqual(event.candidate_node_ids, ())
                        self.assertEqual(event.active_node_ids, ())
                        self.assertFalse(event.forced_active)
                        self.assertIsNone(event.logits)
                        self.assertIsNone(event.probabilities)
                        self.assertIsNone(event.requested_k)
                        self.assertIsNone(event.effective_k)
                        self.assertIsNone(event.top_k_node_ids)
                    long_nodes = tuple(
                        event for event in actual.trace.node_events
                        if event.node_id == "node.long.1"
                    )
                    self.assertTrue(long_nodes)
                    self.assertTrue(
                        all(
                            not event.reached
                            and not event.observed
                            and not event.active
                            for event in long_nodes
                        )
                    )

    def test_sd_pre_state_replay_preserves_natural_route_at_fp32_boundaries(self) -> None:
        """Packed discovery/replay must use the semantic operator ordering.

        Each fixture places the second event's two scores between values that
        used to differ only because packed execution changed a reduction or
        Linear boundary.  A tolerance-only comparison cannot catch that bug:
        the small numerical change flips the exact Top-K membership.
        """

        first_id, second_id = "node.0000", "node.0001"

        with self.subTest(state="ema"):
            plan = _sd_pre_route_boundary_plan(
                d_model=3,
                state_shape=(1,),
                update={
                    "type": "ema",
                    "formula_id": "state.ema.v1",
                    "state_dim": 1,
                    "decay": 0.0,
                    "learnable_decay": False,
                    "state_shape": [1],
                },
            )
            self.assertEqual(
                plan.logical_hash(),
                "ffa050dfaa7f93eea618967ae99b5d960b9b0fe7d52ab7c1080b4f9de0dbafc8",
            )
            self.assertTrue(inspect_packed_support(plan).supported)
            model = SettleGraph(plan).float().eval()
            first = model.receivers[safe_module_key(first_id)]
            second = model.receivers[safe_module_key(second_id)]
            target_normalized = torch.tensor(
                [-0.10141713172197342, 0.08364854753017426, 0.3988964557647705]
            )
            observe_weight = torch.tensor(
                [[0.13018068671226501, -0.06468359380960464, 0.21639209985733032]]
            )
            observe_bias = torch.tensor([-0.17453885078430176])
            token0 = torch.ones(3)
            with torch.no_grad():
                for module in (first, second):
                    assert module.ema_observe is not None
                    assert module.selector_read_linear is not None
                    module.ema_observe.weight.zero_()
                    module.ema_observe.bias.zero_()
                    module.selector_read_linear.weight.zero_()
                    module.selector_read_linear.bias.zero_()
                scale = torch.rsqrt(torch.tensor(1.0 + first.input_norm.eps))
                first.input_norm.weight.copy_(target_normalized / scale)
                second.input_norm.weight.copy_(target_normalized / scale)
                first.ema_observe.weight.copy_(observe_weight)
                first.ema_observe.bias.copy_(observe_bias)
                first.selector_read_linear.weight[0, -1] = 1.0

            normalized = first.normalize_input(token0)
            former_grouped = torch.tanh(
                torch.einsum(
                    "i,oi->o", normalized, first.ema_observe.weight
                )
                + first.ema_observe.bias
            )
            semantic = torch.tanh(
                F.linear(
                    normalized,
                    first.ema_observe.weight,
                    first.ema_observe.bias,
                )
            )
            self.assertEqual(former_grouped.item(), -0.10642942786216736)
            self.assertEqual(semantic.item(), -0.10642946511507034)
            with torch.no_grad():
                second.selector_read_linear.bias.copy_(former_grouped)

            hidden = torch.stack(
                (token0, torch.tensor([2.0, -1.0, 0.5]))
            ).unsqueeze(0)
            mask = torch.ones((1, 2), dtype=torch.bool)
            positions = torch.tensor([[0, 1]])
            common = dict(detach_at_end=False, record_trace=True)
            eager = model.prefill(hidden, mask, ("route.ema",), positions, **common)
            packed = PackedSettleGraph(model).prefill(
                hidden, mask, ("route.ema",), positions, **common
            )
            self.assertEqual(_trace_routes(eager), ((first_id,), (second_id,)))
            self.assertEqual(_trace_routes(packed), _trace_routes(eager))

        with self.subTest(state="gdn"):
            plan = _sd_pre_route_boundary_plan(
                d_model=3,
                state_shape=(3, 1),
                update={
                    "type": "gdn",
                    "formula_id": "state.gdn.v1",
                    "key_dim": 3,
                    "value_dim": 1,
                    "norm_eps": 1e-12,
                    "state_shape": [3, 1],
                },
            )
            self.assertEqual(
                plan.logical_hash(),
                "34c8438ec18c3254be67219aa015c4cafdb7b0e4ee8d968e644771f66d52494b",
            )
            self.assertTrue(inspect_packed_support(plan).supported)
            model = SettleGraph(plan).float().eval()
            first = model.receivers[safe_module_key(first_id)]
            second = model.receivers[safe_module_key(second_id)]
            initial_first = torch.tensor(
                [[-8.35875415802002], [45.4123649597168], [58.850608825683594]]
            )
            key_target = torch.tensor(
                [0.2886396646499634, 0.08684342354536057, -0.9534912109375]
            )
            value_target = torch.tensor([0.8685739636421204])
            eta_target = torch.tensor(0.9978919625282288)
            gamma_target = torch.tensor(0.6968797445297241)
            token0 = torch.tensor([1.0, 2.0, -1.0])
            with torch.no_grad():
                for module in (first, second):
                    assert module.gdn_key is not None
                    assert module.gdn_value is not None
                    assert module.gdn_eta is not None
                    assert module.gdn_gamma is not None
                    assert module.gdn_beta is not None
                    assert module.selector_read_linear is not None
                    module.input_norm.weight.fill_(1.0)
                    module.gdn_key.weight.zero_()
                    module.gdn_value.weight.zero_()
                    module.gdn_eta.weight.zero_()
                    module.gdn_gamma.weight.zero_()
                    module.gdn_beta.zero_()
                    module.selector_read_linear.weight.zero_()
                    module.selector_read_linear.bias.zero_()
                normalized = first.normalize_input(token0)
                denominator = normalized.square().sum()
                first.gdn_key.weight.copy_(
                    key_target[:, None] * normalized[None, :] / denominator
                )
                first.gdn_value.weight.copy_(
                    value_target[:, None] * normalized[None, :] / denominator
                )
                first.gdn_eta.bias.copy_(torch.logit(eta_target))
                first.gdn_gamma.bias.copy_(
                    torch.log(torch.expm1(-torch.log(gamma_target)))
                )
                first.selector_read_linear.weight[0, 4] = 1.0

            normalized = first.normalize_input(token0)
            key = F.normalize(first.gdn_key(normalized), dim=-1, eps=1e-12)
            value = first.gdn_value(normalized)
            eta = torch.sigmoid(first.gdn_eta(normalized)).squeeze()
            gamma = torch.exp(-F.softplus(first.gdn_gamma(normalized).squeeze()))
            affine_a = gamma * (
                torch.eye(3) - eta * key[:, None] * key[None, :]
            )
            affine_b = eta * key[:, None] * value[None, :]
            former_affine = affine_a @ initial_first + affine_b
            decayed = gamma * initial_first
            semantic = decayed + eta * torch.outer(
                key, value - decayed.transpose(-2, -1).matmul(key)
            )
            self.assertEqual(former_affine[1, 0].item(), 35.0185661315918)
            self.assertEqual(semantic[1, 0].item(), 35.01856231689453)
            with torch.no_grad():
                second.selector_read_linear.bias.copy_(former_affine[1, 0])

            hidden = torch.stack(
                (token0, torch.tensor([-2.0, 1.0, 3.0]))
            ).unsqueeze(0)
            mask = torch.ones((1, 2), dtype=torch.bool)
            positions = torch.tensor([[0, 1]])
            state = StateStore(
                values={
                    ("route.gdn", first_id): initial_first,
                    ("route.gdn", second_id): torch.zeros_like(initial_first),
                },
                next_position={"route.gdn": 0},
            )
            common = dict(state=state, detach_at_end=False, record_trace=True)
            eager = model.prefill(hidden, mask, ("route.gdn",), positions, **common)
            packed = PackedSettleGraph(model).prefill(
                hidden, mask, ("route.gdn",), positions, **common
            )
            self.assertEqual(_trace_routes(eager), ((first_id,), (second_id,)))
            self.assertEqual(_trace_routes(packed), _trace_routes(eager))

        with self.subTest(state="attention-window"):
            plan = _sd_pre_route_boundary_plan(
                d_model=2,
                state_shape=(3, 3, 4),
                update={
                    "type": "attention_window",
                    "formula_id": "state.attention-window.v1",
                    "key_dim": 3,
                    "value_dim": 4,
                    "window": 3,
                    "norm_eps": 1e-12,
                    "state_shape": [3, 3, 4],
                },
                summary=True,
            )
            self.assertEqual(
                plan.logical_hash(),
                "107ac028a06938673e4479cb1e9e4c473a1f103bcf0e8e2113c77fadc198c4e1",
            )
            self.assertTrue(inspect_packed_support(plan).supported)
            model = SettleGraph(plan).float().eval()
            first = model.receivers[safe_module_key(first_id)]
            second = model.receivers[safe_module_key(second_id)]
            components = torch.tensor(
                [
                    [
                        -3.0892306490670762e-09,
                        -3.940413506597906e-08,
                        -4.309537171565125e-09,
                        -1.1994727877606692e-08,
                        1.0369884506644667e-08,
                        1.4455182828498891e-08,
                        1.6525312229731526e-09,
                    ],
                    [
                        1.5923404816930997e-08,
                        1.6806474434361007e-08,
                        -2.593550929574917e-09,
                        1.4358975342076974e-08,
                        3.066588316613661e-09,
                        -8.564941644806368e-09,
                        1.5546669729360474e-08,
                    ],
                ]
            )
            left_aligned = torch.zeros((3, 7))
            right_aligned = torch.zeros((3, 7))
            left_aligned[:2] = components
            right_aligned[-2:] = components
            divisor = math.sqrt(14.0)
            former_summary = torch.linalg.vector_norm(
                left_aligned.flatten()
            ) / divisor
            semantic = torch.linalg.vector_norm(
                right_aligned.flatten()
            ) / divisor
            self.assertEqual(former_summary.item(), 1.4921875290951903e-08)
            self.assertEqual(semantic.item(), 1.4921873514595063e-08)
            with torch.no_grad():
                for module in (first, second):
                    assert module.selector_read_linear is not None
                    module.input_norm.weight.fill_(1.0)
                    module.selector_read_linear.weight.zero_()
                    module.selector_read_linear.bias.zero_()
                first.selector_read_linear.weight[0, -1] = 1.0
                second.selector_read_linear.bias.copy_(former_summary)

            sequence_id = "route.attention"
            state = StateStore(
                values={
                    (sequence_id, first_id): AttentionState(
                        torch.tensor([0, 1]),
                        components[:, :3],
                        components[:, 3:],
                    )
                },
                next_position={sequence_id: 2},
            )
            hidden = torch.tensor([[[1.0, -1.0]]])
            mask = torch.tensor([[True]])
            positions = torch.tensor([[2]])
            common = dict(state=state, detach_at_end=False, record_trace=True)
            eager = model.prefill(hidden, mask, (sequence_id,), positions, **common)
            packed = PackedSettleGraph(model).prefill(
                hidden, mask, (sequence_id,), positions, **common
            )
            self.assertEqual(_trace_routes(eager), ((second_id,),))
            self.assertEqual(_trace_routes(packed), _trace_routes(eager))

    def test_upstream_node_compute_preserves_downstream_fp32_tie_route(self) -> None:
        plan = build_diamond(d_model=2, branch_k=1)
        nodes = []
        for node in plan.nodes:
            if node.node_id == "node.root":
                node = dataclasses.replace(
                    node,
                    node_compute={
                        "type": "double_residual_swiglu",
                        "formula_id": "TEST-NODE-SWIGLU-V1",
                        "hidden_dim": 4,
                        "bias": True,
                        "output_shape": [2],
                    },
                )
            elif node.region_id == "region.branches":
                node = dataclasses.replace(
                    node,
                    selector_read_shape=(1,),
                    selector_read={
                        "type": "content_linear",
                        "formula_id": "TEST-READ-PROJ-V1",
                        "out_dim": 1,
                        "output_shape": [1],
                    },
                )
            nodes.append(node)
        regions = tuple(
            dataclasses.replace(
                region,
                score={
                    "type": "read_sum",
                    "formula_id": "score.read-sum.v1",
                    "context_dim": 0,
                },
            )
            if region.region_id == "region.branches"
            else region
            for region in plan.regions
        )
        plan = dataclasses.replace(
            plan, nodes=tuple(nodes), regions=regions
        ).validate()
        self.assertEqual(
            plan.logical_hash(),
            "c701ac7f96bd2b7cad827a726f3499c9cd340da43f387544595df23fac8c25c3",
        )
        self.assertTrue(inspect_packed_support(plan).supported)

        torch.manual_seed(3)
        model = SettleGraph(plan).float().eval()
        root = model.receivers[safe_module_key("node.root")]
        first = model.receivers[safe_module_key("node.branch.a")]
        second = model.receivers[safe_module_key("node.branch.b")]
        with torch.no_grad():
            for module in (first, second):
                assert module.selector_read_linear is not None
                module.input_norm.weight.fill_(1.0)
                module.selector_read_linear.weight.zero_()
                module.selector_read_linear.bias.zero_()
            first.selector_read_linear.weight[0, 1] = 1.0

        generator = torch.Generator(device="cpu").manual_seed(10003)
        hidden = torch.randn((1, 1, 2), generator=generator)
        token = hidden[0, 0]
        ffn_input = root.ffn_norm(token)
        assert root.gate_proj is not None
        assert root.up_proj is not None
        assert root.down_proj is not None
        expansion = F.silu(root.gate_proj(ffn_input)) * root.up_proj(ffn_input)
        semantic_down = F.linear(
            expansion, root.down_proj.weight, root.down_proj.bias
        )
        former_down = torch.addmv(
            root.down_proj.bias, root.down_proj.weight, expansion
        )
        semantic_logit = first.selector_read(
            first.normalize_input(token + semantic_down), None
        ).sum()
        former_logit = first.selector_read(
            first.normalize_input(token + former_down), None
        ).sum()
        self.assertEqual(semantic_logit.item(), 0.30305951833724976)
        self.assertEqual(former_logit.item(), 0.303059458732605)
        with torch.no_grad():
            assert second.selector_read_linear is not None
            second.selector_read_linear.bias.copy_(semantic_logit)

        common = dict(
            execution_mask=torch.ones((1, 1), dtype=torch.bool),
            sequence_ids=("compute-route",),
            token_positions=torch.zeros((1, 1), dtype=torch.int64),
            record_trace=True,
        )
        eager = model.prefill(hidden, **common)
        packed = PackedSettleGraph(model).prefill(hidden, **common)
        assert eager.trace is not None and packed.trace is not None
        eager_branch = next(
            event
            for event in eager.trace.region_events
            if event.region_id == "region.branches"
        )
        packed_branch = next(
            event
            for event in packed.trace.region_events
            if event.region_id == "region.branches"
        )
        torch.testing.assert_close(
            eager_branch.logits,
            torch.tensor([0.30305951833724976, 0.30305951833724976]),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(eager_branch.active_node_ids, ("node.branch.a",))
        self.assertTrue(torch.equal(packed_branch.logits, eager_branch.logits))
        self.assertEqual(packed_branch.active_node_ids, eager_branch.active_node_ids)


class PackedChunkDecodeAndGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = _accepted_cases()

    def _assert_retained_vjps(
        self,
        label: str,
        reference_objectives: tuple[tuple[str, Tensor], ...],
        packed_objectives: tuple[tuple[str, Tensor], ...],
        reference_inputs: tuple[Tensor, ...],
        packed_inputs: tuple[Tensor, ...],
        input_names: tuple[str, ...],
    ) -> None:
        self.assertEqual(
            tuple(name for name, _ in reference_objectives),
            tuple(name for name, _ in packed_objectives),
        )
        objective_pairs = (*zip(reference_objectives, packed_objectives),)
        # Query output once more after every other root.  The packed tracker
        # must replace the prior structural root set on a retained graph.
        objective_pairs = (
            *objective_pairs,
            (("output.repeat", reference_objectives[0][1]),
             ("output.repeat", packed_objectives[0][1])),
        )
        for (objective_name, reference), (_, packed) in objective_pairs:
            with self.subTest(probe=label, objective=objective_name):
                reference_gradients = torch.autograd.grad(
                    reference,
                    reference_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                packed_gradients = torch.autograd.grad(
                    packed,
                    packed_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                compare_nested(
                    dict(zip(input_names, reference_gradients)),
                    dict(zip(input_names, packed_gradients)),
                    tolerance=CPU_FLOAT64_TOLERANCE,
                ).require_pass()

    def test_differentiable_initial_state_and_cross_chunk_objectives_match(self) -> None:
        for state_kind in ("ema", "gdn", "attention_window"):
            plan = _state_boundary_probe_plan(state_kind)

            with self.subTest(state=state_kind, phase="initial-state"):
                torch.manual_seed(991)
                reference_model = SettleGraph(plan).double().eval()
                packed_model = SettleGraph(plan).double().eval()
                packed_model.load_state_dict(reference_model.state_dict(), strict=True)
                reference_state, reference_leaves, state_names = (
                    _differentiable_probe_state(state_kind, torch.float64)
                )
                packed_state, packed_leaves = _clone_differentiable_probe_state(
                    reference_state
                )
                generator = torch.Generator(device="cpu").manual_seed(552)
                hidden = torch.randn(
                    (1, 3, plan.d_model),
                    generator=generator,
                    dtype=torch.float64,
                )
                reference_hidden = hidden.clone().requires_grad_()
                packed_hidden = hidden.clone().requires_grad_()
                mask = torch.ones((1, 3), dtype=torch.bool)
                positions = torch.tensor([[2, 3, 4]])
                common = dict(
                    detach_at_end=False,
                    record_trace=True,
                )
                reference = reference_model.prefill(
                    reference_hidden,
                    mask,
                    ("state-probe",),
                    positions,
                    state=reference_state,
                    **common,
                )
                packed = PackedSettleGraph(packed_model).prefill(
                    packed_hidden,
                    mask,
                    ("state-probe",),
                    positions,
                    state=packed_state,
                    **common,
                )
                parameter_names = tuple(
                    name for name, _ in reference_model.named_parameters()
                )
                self._assert_retained_vjps(
                    f"{state_kind}.initial",
                    _public_vjp_objectives(reference),
                    _public_vjp_objectives(packed),
                    (reference_hidden, *reference_leaves, *reference_model.parameters()),
                    (packed_hidden, *packed_leaves, *packed_model.parameters()),
                    ("hidden", *state_names, *parameter_names),
                )

            with self.subTest(state=state_kind, phase="cross-chunk"):
                torch.manual_seed(992)
                reference_model = SettleGraph(plan).double().eval()
                packed_model = SettleGraph(plan).double().eval()
                packed_model.load_state_dict(reference_model.state_dict(), strict=True)
                packed_executor = PackedSettleGraph(packed_model)
                reference_state, reference_leaves, state_names = (
                    _differentiable_probe_state(state_kind, torch.float64)
                )
                packed_state, packed_leaves = _clone_differentiable_probe_state(
                    reference_state
                )
                generator = torch.Generator(device="cpu").manual_seed(553)
                hidden = torch.randn(
                    (1, 4, plan.d_model),
                    generator=generator,
                    dtype=torch.float64,
                )
                reference_first_hidden = hidden[:, :2].clone().requires_grad_()
                reference_second_hidden = hidden[:, 2:].clone().requires_grad_()
                packed_first_hidden = hidden[:, :2].clone().requires_grad_()
                packed_second_hidden = hidden[:, 2:].clone().requires_grad_()
                mask = torch.ones((1, 2), dtype=torch.bool)
                common = dict(detach_at_end=False, record_trace=True)
                reference_first = reference_model.prefill(
                    reference_first_hidden,
                    mask,
                    ("state-probe",),
                    torch.tensor([[2, 3]]),
                    state=reference_state,
                    **common,
                )
                packed_first = packed_executor.prefill(
                    packed_first_hidden,
                    mask,
                    ("state-probe",),
                    torch.tensor([[2, 3]]),
                    state=packed_state,
                    **common,
                )
                reference_second = reference_model.prefill(
                    reference_second_hidden,
                    mask,
                    ("state-probe",),
                    torch.tensor([[4, 5]]),
                    state=reference_first.state,
                    **common,
                )
                packed_second = packed_executor.prefill(
                    packed_second_hidden,
                    mask,
                    ("state-probe",),
                    torch.tensor([[4, 5]]),
                    state=packed_first.state,
                    **common,
                )
                reference_objectives = list(
                    _public_vjp_objectives(reference_second, "second:")
                )
                packed_objectives = list(
                    _public_vjp_objectives(packed_second, "second:")
                )
                reference_objectives.append(
                    (
                        "combined:first-output+second-state",
                        reference_first.output.sum()
                        + reference_objectives[1][1],
                    )
                )
                packed_objectives.append(
                    (
                        "combined:first-output+second-state",
                        packed_first.output.sum() + packed_objectives[1][1],
                    )
                )
                parameter_names = tuple(
                    name for name, _ in reference_model.named_parameters()
                )
                self._assert_retained_vjps(
                    f"{state_kind}.cross-chunk",
                    tuple(reference_objectives),
                    tuple(packed_objectives),
                    (
                        reference_first_hidden,
                        reference_second_hidden,
                        *reference_leaves,
                        *reference_model.parameters(),
                    ),
                    (
                        packed_first_hidden,
                        packed_second_hidden,
                        *packed_leaves,
                        *packed_model.parameters(),
                    ),
                    (
                        "hidden:first",
                        "hidden:second",
                        *state_names,
                        *parameter_names,
                    ),
                )

    def test_each_node_event_logit_vjp_matches_none_connectivity(self) -> None:
        case = generate_core_v1_candidate_corpus()[64]
        self.assertEqual(case.case_id, "core-v1-candidate.ql-0064.diamond.k1")
        self.assertEqual(
            case.plan.canonical_hash(),
            "23613b480036e1b660e408088f316aff446bc7e061b04370beaaa9090e048819",
        )
        self.assertTrue(inspect_packed_support(case.plan).supported)

        reference_model = _model(case, torch.float64)
        packed_model = _model(case, torch.float64)
        packed_model.load_state_dict(reference_model.state_dict(), strict=True)
        source = _hidden(case, torch.float64)
        reference_hidden = source.clone().requires_grad_()
        packed_hidden = source.clone().requires_grad_()
        reference = reference_model.prefill(
            reference_hidden,
            record_trace=True,
            **_call_arguments(),
        )
        packed = PackedSettleGraph(packed_model).prefill(
            packed_hidden,
            record_trace=True,
            **_call_arguments(),
        )
        assert reference.trace is not None and packed.trace is not None
        self.assertEqual(
            len(reference.trace.node_events), len(packed.trace.node_events)
        )

        reference_objectives = []
        packed_objectives = []
        for index, (reference_event, packed_event) in enumerate(
            zip(reference.trace.node_events, packed.trace.node_events)
        ):
            event_key = (
                reference_event.sequence_id,
                reference_event.token_position,
                reference_event.region_id,
                reference_event.node_id,
            )
            self.assertEqual(
                event_key,
                (
                    packed_event.sequence_id,
                    packed_event.token_position,
                    packed_event.region_id,
                    packed_event.node_id,
                ),
            )
            self.assertEqual(
                reference_event.logit is None, packed_event.logit is None
            )
            if reference_event.logit is None:
                continue
            assert packed_event.logit is not None
            objective_name = f"node-event:{index}:{':'.join(map(str, event_key))}"
            reference_objectives.append(
                (objective_name, reference_event.logit.sum())
            )
            packed_objectives.append(
                (objective_name, packed_event.logit.sum())
            )

        self.assertTrue(reference_objectives)
        parameter_names = tuple(
            name for name, _ in reference_model.named_parameters()
        )
        self._assert_retained_vjps(
            case.case_id,
            tuple(reference_objectives),
            tuple(packed_objectives),
            (reference_hidden, *reference_model.parameters()),
            (packed_hidden, *packed_model.parameters()),
            ("hidden", *parameter_names),
        )

    def test_ema_gdn_and_attention_full_chunk_split_and_decode_match(self) -> None:
        for state_kind in ("ema", "gdn", "attention_window"):
            case = next(
                case for case in self.cases
                if f"state:{state_kind}" in case.features
            )
            with self.subTest(state=state_kind, case=case.case_id):
                model = _model(case, torch.float64)
                packed = PackedSettleGraph(model)
                hidden = _hidden(case, torch.float64)
                full = packed.prefill(hidden, **_call_arguments())
                first_args = {
                    "execution_mask": _EXECUTION_MASK[:, :1],
                    "sequence_ids": _SEQUENCE_IDS,
                    "token_positions": _TOKEN_POSITIONS[:, :1],
                    "detach_at_end": False,
                }
                first = packed.prefill(hidden[:, :1], **first_args)
                second = packed.prefill(
                    hidden[:, 1:],
                    _EXECUTION_MASK[:, 1:],
                    _SEQUENCE_IDS,
                    _TOKEN_POSITIONS[:, 1:],
                    state=first.state,
                    detach_at_end=False,
                )
                torch.testing.assert_close(
                    torch.cat((first.output, second.output), dim=1),
                    full.output,
                    atol=1e-10,
                    rtol=1e-8,
                )
                compare_nested(
                    full.state, second.state, tolerance=CPU_FLOAT64_TOLERANCE
                ).require_pass()

                decode_state = None
                decode_outputs = []
                for token in range(_LENGTH):
                    decoded = packed.decode(
                        hidden[:, token],
                        _EXECUTION_MASK[:, token],
                        _SEQUENCE_IDS,
                        _TOKEN_POSITIONS[:, token],
                        state=decode_state,
                        detach_at_end=False,
                    )
                    decode_state = decoded.state
                    decode_outputs.append(decoded.output)
                torch.testing.assert_close(
                    torch.stack(decode_outputs, dim=1),
                    full.output,
                    atol=1e-10,
                    rtol=1e-8,
                )
                compare_nested(
                    full.state, decode_state, tolerance=CPU_FLOAT64_TOLERANCE
                ).require_pass()

                empty = packed.prefill(
                    hidden[:, :0],
                    _EXECUTION_MASK[:, :0],
                    _SEQUENCE_IDS,
                    _TOKEN_POSITIONS[:, :0],
                    state=second.state,
                    detach_at_end=False,
                )
                self.assertEqual(empty.output.shape[1], 0)
                compare_nested(
                    second.state, empty.state, tolerance=CPU_FLOAT64_TOLERANCE
                ).require_pass()

    def test_mask_hole_and_zero_decay_match_token_reference(self) -> None:
        case = next(
            case for case in self.cases
            if case.motif == "singleton" and "state:ema" in case.features
        )
        node = case.plan.nodes[0]
        update = {
            "type": "ema",
            "formula_id": "state.ema.v1",
            "state_dim": int(node.update["state_dim"]),
            "decay": 0.0,
            "learnable_decay": bool(node.update["learnable_decay"]),
            "state_shape": list(node.state_shape),
        }
        plan = dataclasses.replace(
            case.plan,
            nodes=(dataclasses.replace(node, update=update),),
        ).validate()
        case = dataclasses.replace(case, plan=plan)
        model = _model(case, torch.float64)
        packed = PackedSettleGraph(model)
        hidden = torch.randn((1, 4, plan.d_model), dtype=torch.float64)
        mask = torch.tensor([[True, False, True, True]])
        positions = torch.tensor([[0, 99, 1, 2]])
        kwargs = dict(
            execution_mask=mask,
            sequence_ids=("mask-hole",),
            token_positions=positions,
            detach_at_end=False,
            record_trace=True,
        )
        reference = model.prefill(hidden, **kwargs)
        actual = packed.prefill(hidden, **kwargs)
        compare_nested(
            reference, actual, tolerance=CPU_FLOAT64_TOLERANCE
        ).require_pass()

    def test_chunk_detach_cuts_only_the_cross_chunk_state_gradient(self) -> None:
        case = next(
            case for case in self.cases
            if case.motif == "singleton" and "state:ema" in case.features
        )
        for detach in (False, True):
            with self.subTest(detach=detach):
                reference_model = _model(case, torch.float64)
                packed_model = _model(case, torch.float64)
                packed_model.load_state_dict(reference_model.state_dict(), strict=True)
                packed = PackedSettleGraph(packed_model)
                first_source = _hidden(case, torch.float64)[:, :1]
                second_source = _hidden(case, torch.float64)[:, 1:2]
                left_first = first_source.clone().requires_grad_(True)
                right_first = first_source.clone().requires_grad_(True)
                left_second = second_source.clone().requires_grad_(True)
                right_second = second_source.clone().requires_grad_(True)
                first_kwargs = dict(
                    execution_mask=_EXECUTION_MASK[:, :1],
                    sequence_ids=_SEQUENCE_IDS,
                    token_positions=_TOKEN_POSITIONS[:, :1],
                    detach_at_end=detach,
                )
                left_state = reference_model.prefill(left_first, **first_kwargs).state
                right_state = packed.prefill(right_first, **first_kwargs).state
                second_kwargs = dict(
                    execution_mask=_EXECUTION_MASK[:, 1:2],
                    sequence_ids=_SEQUENCE_IDS,
                    token_positions=_TOKEN_POSITIONS[:, 1:2],
                    detach_at_end=False,
                )
                left = reference_model.prefill(
                    left_second, state=left_state, **second_kwargs
                )
                right = packed.prefill(
                    right_second, state=right_state, **second_kwargs
                )
                left.output.sum().backward()
                right.output.sum().backward()
                compare_nested(
                    {"first": left_first.grad, "second": left_second.grad},
                    {"first": right_first.grad, "second": right_second.grad},
                    tolerance=CPU_FLOAT64_TOLERANCE,
                ).require_pass()
                if detach:
                    self.assertIsNone(right_first.grad)
                else:
                    self.assertIsNotNone(right_first.grad)
                    assert right_first.grad is not None
                    self.assertTrue(bool((right_first.grad != 0).any().item()))

    def test_supported_vjp_cases_match_values_and_none_connectivity(self) -> None:
        selected = [case for case in self.cases if case.vjp]
        # Add a Top-2/8 regression where most receiver compute parameters are
        # structurally disconnected from an output-only objective.
        selected.append(
            next(
                case for case in self.cases
                if case.case_id == "corpus.023.single-layer-r8.top-2-r2"
            )
        )
        for dtype in (torch.float64, torch.float32):
            for case in selected:
                with self.subTest(dtype=str(dtype), case=case.case_id):
                    reference_model = _model(case, dtype)
                    packed_model = _model(case, dtype)
                    packed_model.load_state_dict(reference_model.state_dict(), strict=True)
                    packed = PackedSettleGraph(packed_model)
                    source = _hidden(case, dtype)
                    left_hidden = source.clone().requires_grad_(True)
                    right_hidden = source.clone().requires_grad_(True)
                    left = reference_model.prefill(left_hidden, **_call_arguments())
                    right = packed.prefill(right_hidden, **_call_arguments())
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(case.input_seed ^ 0x5A17)
                    cotangent = torch.randn(
                        left.output.shape, generator=generator, dtype=torch.float64
                    ).to(dtype)
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
                    compare_nested(
                        _gradient_record(reference_model, left_hidden),
                        _gradient_record(packed_model, right_hidden),
                        tolerance=_tolerance(dtype),
                    ).require_pass()

    def test_competitive_sd_pre_state_families_cover_long_chunk_decode_and_vjp(self) -> None:
        for state_kind, length in (
            ("ema", 17),
            ("gdn", 7),
            ("attention_window", 7),
        ):
            case = _competitive_sd_pre_case(state_kind)
            for dtype in (torch.float64, torch.float32):
                with self.subTest(state=state_kind, dtype=str(dtype), phase="vjp"):
                    reference_model = _model(case, dtype)
                    packed_model = _model(case, dtype)
                    packed_model.load_state_dict(reference_model.state_dict(), strict=True)
                    packed = PackedSettleGraph(packed_model)
                    source, mask, positions = _long_inputs(case, length, dtype)
                    left_hidden = source.clone().requires_grad_(True)
                    right_hidden = source.clone().requires_grad_(True)
                    common = dict(
                        execution_mask=mask,
                        sequence_ids=_SEQUENCE_IDS,
                        token_positions=positions,
                        routing_stats_mask=mask,
                        detach_at_end=False,
                        record_trace=True,
                    )
                    reference = reference_model.prefill(left_hidden, **common)
                    actual = packed.prefill(right_hidden, **common)
                    compare_nested(
                        reference, actual, tolerance=_tolerance(dtype)
                    ).require_pass()
                    _assert_trace_routes_rederive_from_logits(self, actual)

                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(case.input_seed ^ 0x71A9)
                    cotangent = torch.randn(
                        reference.output.shape,
                        generator=generator,
                        dtype=torch.float64,
                    ).to(dtype)
                    left_objective = (
                        (reference.output * cotangent).sum()
                        + 0.07 * reference.balance_loss
                        + 0.03 * _state_objective(reference.state, reference.output)
                    )
                    right_objective = (
                        (actual.output * cotangent).sum()
                        + 0.07 * actual.balance_loss
                        + 0.03 * _state_objective(actual.state, actual.output)
                    )
                    left_objective.backward()
                    right_objective.backward()
                    compare_nested(
                        _gradient_record(reference_model, left_hidden),
                        _gradient_record(packed_model, right_hidden),
                        tolerance=_tolerance(dtype),
                    ).require_pass()
                    for gradient in _gradient_record(
                        packed_model, right_hidden
                    )["parameters"].values():
                        if gradient is not None:
                            self.assertTrue(bool(torch.isfinite(gradient).all().item()))

            with self.subTest(state=state_kind, phase="chunk-decode"):
                model = _model(case, torch.float64)
                packed = PackedSettleGraph(model)
                hidden, mask, positions = _long_inputs(case, length, torch.float64)
                full = packed.prefill(
                    hidden,
                    mask,
                    _SEQUENCE_IDS,
                    positions,
                    detach_at_end=False,
                )
                split_points = (1, 3) if state_kind == "attention_window" else (1,)
                for split in split_points:
                    first = packed.prefill(
                        hidden[:, :split],
                        mask[:, :split],
                        _SEQUENCE_IDS,
                        positions[:, :split],
                        detach_at_end=False,
                    )
                    second = packed.prefill(
                        hidden[:, split:],
                        mask[:, split:],
                        _SEQUENCE_IDS,
                        positions[:, split:],
                        state=first.state,
                        detach_at_end=False,
                    )
                    torch.testing.assert_close(
                        torch.cat((first.output, second.output), dim=1),
                        full.output,
                        atol=1e-10,
                        rtol=1e-8,
                    )
                    compare_nested(
                        full.state,
                        second.state,
                        tolerance=CPU_FLOAT64_TOLERANCE,
                    ).require_pass()

                decode_state = None
                decode_outputs = []
                for token_index in range(length):
                    token = packed.decode(
                        hidden[:, token_index],
                        mask[:, token_index],
                        _SEQUENCE_IDS,
                        positions[:, token_index],
                        state=decode_state,
                    )
                    decode_state = token.state
                    decode_outputs.append(token.output)
                torch.testing.assert_close(
                    torch.stack(decode_outputs, dim=1),
                    full.output,
                    atol=1e-10,
                    rtol=1e-8,
                )
                compare_nested(
                    full.state,
                    decode_state,
                    tolerance=CPU_FLOAT64_TOLERANCE,
                ).require_pass()

    def test_decode_default_detaches_and_explicit_false_preserves_state_graph(self) -> None:
        case = next(
            case for case in self.cases
            if case.motif == "singleton" and "state:ema" in case.features
        )
        default_model = _model(case, torch.float64)
        default_hidden = _hidden(case, torch.float64)[:, 0].clone().requires_grad_(True)
        default = PackedSettleGraph(default_model).decode(
            default_hidden,
            _EXECUTION_MASK[:, 0],
            _SEQUENCE_IDS,
            _TOKEN_POSITIONS[:, 0],
        )
        tensor_states = [
            value for value in default.state.values.values()
            if isinstance(value, Tensor)
        ]
        self.assertTrue(tensor_states)
        self.assertTrue(all(not value.requires_grad for value in tensor_states))

        reference_model = _model(case, torch.float64)
        packed_model = _model(case, torch.float64)
        packed_model.load_state_dict(reference_model.state_dict(), strict=True)
        left_hidden = _hidden(case, torch.float64)[:, 0].clone().requires_grad_(True)
        right_hidden = _hidden(case, torch.float64)[:, 0].clone().requires_grad_(True)
        left = reference_model.interpret_token(
            left_hidden,
            _EXECUTION_MASK[:, 0],
            _SEQUENCE_IDS,
            _TOKEN_POSITIONS[:, 0],
            detach_at_end=False,
        )
        right = PackedSettleGraph(packed_model).decode(
            right_hidden,
            _EXECUTION_MASK[:, 0],
            _SEQUENCE_IDS,
            _TOKEN_POSITIONS[:, 0],
            detach_at_end=False,
        )
        left_objective = _state_objective(left.state, left.output)
        right_objective = _state_objective(right.state, right.output)
        left_objective.backward()
        right_objective.backward()
        self.assertIsNotNone(right_hidden.grad)
        compare_nested(
            _gradient_record(reference_model, left_hidden),
            _gradient_record(packed_model, right_hidden),
            tolerance=CPU_FLOAT64_TOLERANCE,
        ).require_pass()


if __name__ == "__main__":
    unittest.main()
