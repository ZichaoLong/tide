from __future__ import annotations

import dataclasses
import unittest

import torch

from tide.builders import (
    build_chain,
    build_diamond,
    build_mixed_regions,
    build_multi_entry_terminal,
    build_single_layer,
    build_singleton,
    build_small_hb,
    build_unequal_path,
)
from tide.engine import (
    BoundaryEventTrace,
    ExecutionTrace,
    NodeEventTrace,
    OutputEventTrace,
    RegionEventTrace,
    SettleGraph,
    StateWriteTrace,
)
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    TraceInvariantError,
    Tolerance,
    classify_route_boundary,
    compare_nested,
    validate_trace_invariants,
)
from tide.plan import bind_dtypes


def _replace_plan(plan, *, nodes=None, regions=None, **changes):
    return dataclasses.replace(
        plan,
        nodes=tuple(plan.nodes if nodes is None else nodes),
        regions=tuple(plan.regions if regions is None else regions),
        **changes,
    ).validate()


def _profile_plan(profile: str, timing: str):
    plan = build_single_layer(receiver_count=2, k=1, d_model=2)
    nodes = []
    for node in plan.nodes:
        stateful = profile != "N"
        read = (
            {
                "type": "content",
                "formula_id": "read.selector.content.v1",
                "output_shape": [2],
            }
            if timing == "content"
            else {
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
                "output_shape": [1],
            }
        )
        nodes.append(
            dataclasses.replace(
                node,
                state_shape=(2,) if stateful else (),
                state_owner=node.node_id if stateful else None,
                update=(
                    {
                        "type": "ema",
                        "formula_id": "state.ema.v1",
                        "state_dim": 2,
                        "decay": 0.6,
                        "state_shape": [2],
                    }
                    if stateful
                    else {"type": "none", "formula_id": "update.none.v1"}
                ),
                selector_read_shape=(2,) if timing == "content" else (1,),
                selector_read=read,
                node_compute={
                    "type": "affine_residual",
                    "formula_id": "TEST-NODE-AFFINE-V1",
                    "bias": True,
                    "output_shape": [2],
                },
            )
        )
    region = dataclasses.replace(
        plan.regions[0],
        profile=profile,
        selector_timing=timing,
        score={
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
        },
    )
    return _replace_plan(plan, nodes=nodes, regions=(region,))


def _state_variant_plan(kind: str):
    plan = build_singleton(d_model=2)
    if kind == "ema":
        state_shape = (3,)
        update = {
            "type": "ema",
            "formula_id": "state.ema.v1",
            "state_dim": 3,
            "decay": 0.4,
            "state_shape": [3],
        }
        selector_read = {
            "type": "content_state_linear",
            "formula_id": "TEST-READ-PROJ-V1",
            "out_dim": 1,
            "output_shape": [1],
        }
    elif kind == "gdn":
        state_shape = (2, 3)
        update = {
            "type": "gdn",
            "formula_id": "state.gdn.v1",
            "key_dim": 2,
            "value_dim": 3,
            "norm_eps": 1e-12,
            "state_shape": [2, 3],
        }
        selector_read = {
            "type": "content_state_linear",
            "formula_id": "TEST-READ-PROJ-V1",
            "out_dim": 1,
            "output_shape": [1],
        }
    elif kind == "attention_window":
        state_shape = (3, 2, 3)
        update = {
            "type": "attention_window",
            "formula_id": "state.attention-window.v1",
            "key_dim": 2,
            "value_dim": 3,
            "window": 3,
            "norm_eps": 1e-12,
            "state_shape": [3, 2, 3],
        }
        selector_read = {
            "type": "content_state_summary_linear",
            "formula_id": "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
            "out_dim": 1,
            "output_shape": [1],
        }
    else:  # pragma: no cover - test helper guard
        raise AssertionError(kind)
    node = dataclasses.replace(
        plan.nodes[0],
        state_shape=state_shape,
        state_owner=plan.nodes[0].node_id,
        update=update,
        selector_read_shape=(1,),
        selector_read=selector_read,
        ffn_read={
            "type": "state_default",
            "formula_id": f"read.ffn.{kind.replace('_', '-')}.v1",
            "output_shape": [2],
        },
        node_compute={
            "type": "double_residual_swiglu",
            "formula_id": "TEST-NODE-SWIGLU-V1",
            "hidden_dim": 4,
            "bias": True,
            "output_shape": [2],
        },
        emit={
            "type": "softp",
            "formula_id": "emit.softp.v1",
            "output_shape": [2],
        },
    )
    region = dataclasses.replace(
        plan.regions[0], profile="BO", selector_timing="post"
    )
    return _replace_plan(plan, nodes=(node,), regions=(region,))


class ComparatorTests(unittest.TestCase):
    def test_nested_comparator_records_worst_tensor_element(self) -> None:
        reference = {
            "value": torch.tensor([1.0, 0.0, -3.0], dtype=torch.float64),
            "route": ("node.a", "node.c"),
        }
        candidate = {
            "value": torch.tensor(
                [1.0 + 1e-11, 0.0, -3.0 - 2e-11], dtype=torch.float64
            ),
            "route": ("node.a", "node.c"),
        }
        report = compare_nested(
            reference,
            candidate,
            tolerance=CPU_FLOAT64_TOLERANCE,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.tensors), 1)
        self.assertIn(".flat[", report.tensors[0].worst_path)
        self.assertGreater(report.tensors[0].max_absolute_error, 0.0)

    def test_comparator_rejects_nonfinite_dtype_and_discrete_mismatch(self) -> None:
        nonfinite = compare_nested(
            torch.tensor([1.0]),
            torch.tensor([float("nan")]),
            tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
        )
        self.assertFalse(nonfinite.passed)
        self.assertIn("NaN or infinity", nonfinite.errors[0])

        dtype = compare_nested(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float32),
            tolerance=CPU_FLOAT64_TOLERANCE,
        )
        self.assertFalse(dtype.passed)
        self.assertIn("dtype mismatch", dtype.errors[0])

        discrete = compare_nested(
            {"active": ("a",)},
            {"active": ("b",)},
            tolerance=CPU_FLOAT64_TOLERANCE,
        )
        self.assertFalse(discrete.passed)
        with self.assertRaises(AssertionError):
            discrete.require_pass()

    def test_cross_binding_can_compare_different_floating_dtypes(self) -> None:
        report = compare_nested(
            torch.tensor([0.25, -0.5], dtype=torch.float64),
            torch.tensor([0.25, -0.5], dtype=torch.float32),
            tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
            require_same_dtype=False,
        )
        self.assertTrue(report.passed)


class RouteBoundaryTests(unittest.TestCase):
    def test_all_four_route_boundary_classes(self) -> None:
        tolerance = Tolerance(atol=1e-6, rtol=1e-5)
        all_active = classify_route_boundary(
            torch.tensor([2.0, 1.0]), ("b", "a"), 2, tolerance=tolerance
        )
        self.assertEqual(all_active.classification, "all-active")

        tie = classify_route_boundary(
            torch.tensor([1.0, 1.0]), ("b", "a"), 1, tolerance=tolerance
        )
        self.assertEqual(tie.classification, "exact-tie")
        self.assertEqual(tie.kth_node_id, "a")

        near = classify_route_boundary(
            torch.tensor([1.0 + 1e-6, 1.0]),
            ("a", "b"),
            1,
            tolerance=tolerance,
        )
        self.assertEqual(near.classification, "near-boundary")

        safe = classify_route_boundary(
            torch.tensor([2.0, -1.0]), ("a", "b"), 1, tolerance=tolerance
        )
        self.assertEqual(safe.classification, "margin-safe")


class TraceInvariantTests(unittest.TestCase):
    def test_singleton_exact_trace_has_an_independent_analytic_golden(self) -> None:
        plan = build_singleton(d_model=2)
        region = dataclasses.replace(
            plan.regions[0],
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {plan.nodes[0].node_id: 7.25},
            },
        )
        plan = dataclasses.replace(plan, regions=(region,)).validate()
        graph = SettleGraph(plan).double()
        hidden = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        actual = graph.interpret_token(
            hidden,
            torch.tensor([True]),
            ["sequence"],
            torch.tensor([0], dtype=torch.int64),
            record_trace=True,
        )
        assert actual.trace is not None

        # This expected value spells out the fixture equation directly.  It
        # intentionally does not call ReceiverModule, RegionSelector, Top-K,
        # Emit, output Aggregate, or an executor helper.
        message = hidden[0]
        normalized = message / torch.sqrt(
            message.square().mean() + torch.tensor(1e-6, dtype=torch.float64)
        )
        node_id = "node.0"
        region_id = "region.0"
        expected = ExecutionTrace(
            node_events=(
                NodeEventTrace(
                    "sequence",
                    0,
                    region_id,
                    node_id,
                    True,
                    False,
                    True,
                    message,
                    normalized,
                    None,
                    None,
                    None,
                    None,
                    None,
                    torch.tensor(1.0, dtype=torch.float64),
                    message,
                    message,
                    (("boundary:node.0", "DATA", message),),
                ),
            ),
            edge_events=(),
            boundary_events=(
                BoundaryEventTrace("sequence", 0, node_id, message),
            ),
            region_events=(
                RegionEventTrace(
                    "sequence",
                    0,
                    region_id,
                    (node_id,),
                    None,
                    torch.tensor([1.0], dtype=torch.float64),
                    None,
                    None,
                    (node_id,),
                    True,
                    None,
                ),
            ),
            state_writes=(),
            output_events=(
                OutputEventTrace(
                    "sequence", 0, ((node_id, message),), message
                ),
            ),
        )
        compare_nested(
            expected,
            actual.trace,
            tolerance=CPU_FLOAT64_TOLERANCE,
        ).require_pass()
        self.assertEqual(actual.state.next_position, {"sequence": 1})
        self.assertEqual(actual.balance_stats.regions[region_id].event_count, 1)

    def _run_both(self, plan):
        graph = SettleGraph(plan).double()
        graph.make_identity()
        hidden = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[-1.0, 0.0, 1.0], [7.0, 8.0, 9.0]],
            ],
            dtype=torch.float64,
        )
        execution_mask = torch.tensor([[True, True], [True, False]])
        positions = torch.tensor([[0, 1], [0, 99]], dtype=torch.int64)
        executed = (("seq-a", 0), ("seq-a", 1), ("seq-b", 0))
        token_major = graph.prefill(
            hidden,
            execution_mask,
            ["seq-a", "seq-b"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        region_major = graph.prefill_region_major(
            hidden,
            execution_mask,
            ["seq-a", "seq-b"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        assert token_major.trace is not None
        assert region_major.trace is not None
        return token_major.trace, region_major.trace, executed

    def test_manual_motifs_satisfy_invariants_and_match(self) -> None:
        for plan in (
            build_singleton(d_model=3),
            build_single_layer(receiver_count=2, k=1, d_model=3),
            build_single_layer(receiver_count=8, k=2, d_model=3),
            build_single_layer(receiver_count=8, k=8, d_model=3),
            build_chain(length=3, d_model=3),
            build_diamond(d_model=3, branch_k=1),
            build_diamond(d_model=3, branch_k=2),
            build_unequal_path(d_model=3),
            build_multi_entry_terminal(d_model=3),
            build_mixed_regions(d_model=3),
            build_small_hb(d_model=3),
        ):
            with self.subTest(plan=plan.plan_id):
                token_trace, region_trace, executed = self._run_both(plan)
                validate_trace_invariants(
                    plan,
                    token_trace,
                    executed,
                    tolerance=CPU_FLOAT64_TOLERANCE,
                )
                validate_trace_invariants(
                    plan,
                    region_trace,
                    executed,
                    tolerance=CPU_FLOAT64_TOLERANCE,
                )
                report = compare_nested(
                    token_trace,
                    region_trace,
                    tolerance=CPU_FLOAT64_TOLERANCE,
                )
                report.require_pass()

    def test_invariant_check_rejects_duplicate_or_wrong_edge_settlement(self) -> None:
        plan = build_chain(length=2, d_model=3)
        trace, _, executed = self._run_both(plan)
        first = trace.edge_events[0]
        malformed_edge = dataclasses.replace(first, status="CLOSED", payload=None)
        malformed = dataclasses.replace(
            trace,
            edge_events=(malformed_edge,) + trace.edge_events[1:],
        )
        with self.assertRaisesRegex(TraceInvariantError, "source activity"):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        duplicate = dataclasses.replace(
            trace,
            edge_events=trace.edge_events + (trace.edge_events[-1],),
        )
        with self.assertRaisesRegex(TraceInvariantError, "duplicate IDs"):
            validate_trace_invariants(
                plan,
                duplicate,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

    def test_invariant_check_links_forced_region_and_node_routing_values(self) -> None:
        plan = build_singleton(d_model=3)
        trace, _, executed = self._run_both(plan)
        first = trace.region_events[0]
        self.assertIsNone(first.logits)
        self.assertIsNone(first.requested_k)
        self.assertIsNone(first.effective_k)
        self.assertIsNone(first.top_k_node_ids)
        malformed_region = dataclasses.replace(
            first,
            top_k_node_ids=(first.candidate_node_ids[0],),
        )
        malformed = dataclasses.replace(
            trace,
            region_events=(malformed_region,) + trace.region_events[1:],
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "leave logits and K/Top-K fields absent"
        ):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        node = trace.node_events[0]
        assert node.probability is not None
        malformed_node = dataclasses.replace(
            node, probability=node.probability + 1.0
        )
        malformed = dataclasses.replace(
            trace,
            node_events=(malformed_node,) + trace.node_events[1:],
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "linked trace payload differs"
        ):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

    def test_invariant_check_links_parent_settlements_and_reached_flags(self) -> None:
        plan = build_chain(length=2, d_model=3)
        trace, _, executed = self._run_both(plan)
        terminal_id = plan.terminal_node_ids[0]
        node_index = next(
            index
            for index, event in enumerate(trace.node_events)
            if event.sequence_id == "seq-a"
            and event.token_position == 0
            and event.node_id == terminal_id
        )
        node_event = trace.node_events[node_index]
        parent_id, _, _ = node_event.parent_messages[0]
        malformed_node = dataclasses.replace(
            node_event,
            parent_messages=((parent_id, "CLOSED", None),),
        )
        malformed_nodes = list(trace.node_events)
        malformed_nodes[node_index] = malformed_node
        malformed = dataclasses.replace(
            trace, node_events=tuple(malformed_nodes)
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "parent_messages"
        ):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        wrong_reached = dataclasses.replace(malformed_node, reached=False)
        malformed_nodes[node_index] = wrong_reached
        malformed = dataclasses.replace(
            trace, node_events=tuple(malformed_nodes)
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "reached flag"
        ):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

    def test_invariant_check_uses_plan_for_forced_flag_and_fixed_k(self) -> None:
        singleton = build_singleton(d_model=3)
        trace, _, executed = self._run_both(singleton)
        malformed_region = dataclasses.replace(
            trace.region_events[0], forced_active=False
        )
        malformed = dataclasses.replace(
            trace,
            region_events=(malformed_region,) + trace.region_events[1:],
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "forced-active flag"
        ):
            validate_trace_invariants(
                singleton,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        fixed = build_single_layer(receiver_count=2, k=2, d_model=3)
        fixed_trace, _, fixed_executed = self._run_both(fixed)
        first = fixed_trace.region_events[0]
        assert first.top_k_node_ids is not None
        self.assertEqual(first.top_k_node_ids, first.active_node_ids)
        malformed_region = dataclasses.replace(first, requested_k=1)
        malformed = dataclasses.replace(
            fixed_trace,
            region_events=(malformed_region,) + fixed_trace.region_events[1:],
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "fixed value"
        ):
            validate_trace_invariants(
                fixed,
                malformed,
                fixed_executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        malformed_region = dataclasses.replace(
            first, top_k_node_ids=tuple(reversed(first.top_k_node_ids))
        )
        malformed = dataclasses.replace(
            fixed_trace,
            region_events=(malformed_region,) + fixed_trace.region_events[1:],
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "active IDs do not match Top-K IDs"
        ):
            validate_trace_invariants(
                fixed,
                malformed,
                fixed_executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

    def test_trace_token_keys_are_strict_and_history_none_has_no_writes(self) -> None:
        plan = build_singleton(d_model=3)
        trace, _, executed = self._run_both(plan)
        for malformed_tokens in (
            ((7, 0),),
            (("seq-a", True),),
            (("e\u0301", 0),),
        ):
            with self.subTest(executed_tokens=repr(malformed_tokens)):
                with self.assertRaises(TraceInvariantError):
                    validate_trace_invariants(
                        plan,
                        trace,
                        malformed_tokens,
                        tolerance=CPU_FLOAT64_TOLERANCE,
                    )

        first_output = trace.output_events[0]
        invalid_output = dataclasses.replace(first_output, sequence_id=7)
        malformed_trace = dataclasses.replace(
            trace,
            output_events=(invalid_output,) + trace.output_events[1:],
        )
        with self.assertRaises(TraceInvariantError):
            validate_trace_invariants(
                plan,
                malformed_trace,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

        extra_write = StateWriteTrace(
            "seq-a", 0, "selector_history", "region.0", torch.tensor(1.0)
        )
        malformed_trace = dataclasses.replace(
            trace, state_writes=(extra_write,)
        )
        with self.assertRaisesRegex(
            TraceInvariantError, "selector-history"
        ):
            validate_trace_invariants(
                plan,
                malformed_trace,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )

    def test_nonexecuted_position_must_have_no_graph_event(self) -> None:
        plan = build_singleton(d_model=3)
        trace, _, executed = self._run_both(plan)
        output = dataclasses.replace(trace.output_events[0], token_position=99)
        malformed = ExecutionTrace(
            trace.node_events,
            trace.edge_events,
            trace.boundary_events,
            trace.region_events,
            trace.state_writes,
            trace.output_events + (output,),
        )
        with self.assertRaisesRegex(TraceInvariantError, "non-executed"):
            validate_trace_invariants(
                plan,
                malformed,
                executed,
                tolerance=CPU_FLOAT64_TOLERANCE,
            )


class SemanticMatrixDifferentialTests(unittest.TestCase):
    def _compare_paths(
        self, plan, *, width: int = 2, dtype: torch.dtype = torch.float64
    ) -> None:
        torch.manual_seed(90210)
        graph = SettleGraph(plan).to(dtype=dtype)
        hidden = torch.randn((2, 4, width), dtype=dtype)
        execution = torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        )
        positions = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 40, 41]], dtype=torch.int64
        )
        executed = (
            ("long", 0),
            ("long", 1),
            ("long", 2),
            ("long", 3),
            ("short", 0),
            ("short", 1),
        )
        token_major = graph.prefill(
            hidden,
            execution,
            ["long", "short"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        region_major = graph.prefill_region_major(
            hidden,
            execution,
            ["long", "short"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        assert token_major.trace is not None
        assert region_major.trace is not None
        tolerance = (
            CPU_FLOAT64_TOLERANCE
            if dtype == torch.float64
            else SAME_BACKEND_FLOAT32_TOLERANCE
        )
        validate_trace_invariants(
            plan,
            token_major.trace,
            executed,
            tolerance=tolerance,
        )
        validate_trace_invariants(
            plan,
            region_major.trace,
            executed,
            tolerance=tolerance,
        )
        compare_nested(
            token_major,
            region_major,
            tolerance=tolerance,
        ).require_pass()

    def test_all_six_standard_profile_timing_combinations(self) -> None:
        for profile, timing in (
            ("N", "content"),
            ("SD", "content"),
            ("SD", "pre"),
            ("BO", "content"),
            ("BO", "pre"),
            ("BO", "post"),
        ):
            with self.subTest(profile=profile, timing=timing):
                self._compare_paths(_profile_plan(profile, timing))

    def test_ema_gdn_and_attention_state_paths(self) -> None:
        for kind in ("ema", "gdn", "attention_window"):
            with self.subTest(kind=kind):
                self._compare_paths(_state_variant_plan(kind))

    def test_cpu_float32_executor_matrix(self) -> None:
        plans = (
            build_single_layer(receiver_count=8, k=2, d_model=2),
            build_chain(length=4, d_model=2),
            build_diamond(d_model=2, branch_k=1),
            build_unequal_path(d_model=2),
            build_multi_entry_terminal(d_model=2),
            build_mixed_regions(d_model=2),
            build_small_hb(d_model=2),
            _profile_plan("SD", "pre"),
            _profile_plan("BO", "post"),
            _state_variant_plan("ema"),
            _state_variant_plan("gdn"),
            _state_variant_plan("attention_window"),
        )
        for plan in plans:
            with self.subTest(plan=plan.plan_id):
                self._compare_paths(plan, dtype=torch.float32)

    def test_cpu_float64_and_float32_bindings_share_source_fixture(self) -> None:
        source = torch.tensor(
            [[[0.25, -0.5], [1.25, 0.75], [-0.125, 2.0]]],
            dtype=torch.float64,
        )
        execution = torch.ones((1, 3), dtype=torch.bool)
        positions = torch.arange(3, dtype=torch.int64).reshape(1, 3)
        for kind in ("ema", "gdn", "attention_window"):
            plan = _state_variant_plan(kind)
            typed64 = bind_dtypes(
                plan,
                hidden="float64",
                parameter="float64",
                state="float64",
                readout="float64",
            )
            typed32 = bind_dtypes(
                plan,
                hidden="float32",
                parameter="float32",
                state="float32",
                readout="float32",
            )
            self.assertEqual(typed64.logical_hash(), typed32.logical_hash())
            self.assertNotEqual(typed64.typed_hash(), typed32.typed_hash())
            torch.manual_seed(441)
            graph64 = SettleGraph(typed64).double()
            graph32 = SettleGraph(typed32).float()
            graph32.load_state_dict(
                {
                    key: value.float() if value.is_floating_point() else value
                    for key, value in graph64.state_dict().items()
                }
            )
            result64 = graph64.prefill_region_major(
                source,
                execution,
                ["sequence"],
                positions,
                detach_at_end=False,
                record_trace=True,
            )
            result32 = graph32.prefill_region_major(
                source.float(),
                execution,
                ["sequence"],
                positions,
                detach_at_end=False,
                record_trace=True,
            )
            with self.subTest(kind=kind):
                compare_nested(
                    result64,
                    result32,
                    tolerance=SAME_BACKEND_FLOAT32_TOLERANCE,
                    require_same_dtype=False,
                ).require_pass()

    def test_edge_aggregate_score_emit_and_output_variants(self) -> None:
        for aggregate_type in ("edge_softmax", "edge_linear_mean"):
            base = build_diamond(d_model=2, branch_k=2)
            nodes = tuple(
                dataclasses.replace(
                    node,
                    aggregate=(
                        {
                            "type": aggregate_type,
                            "formula_id": (
                                "TEST-AGG-EDGE-SOFTMAX-V1"
                                if aggregate_type == "edge_softmax"
                                else "TEST-AGG-EDGE-AFFINE-MEAN-V1"
                            ),
                            **(
                                {"bias": True}
                                if aggregate_type == "edge_linear_mean"
                                else {}
                            ),
                            "output_shape": [2],
                        }
                        if node.node_id == "node.out"
                        else node.aggregate
                    ),
                )
                for node in base.nodes
            )
            with self.subTest(aggregate=aggregate_type):
                self._compare_paths(_replace_plan(base, nodes=nodes))

        for score_type in ("mlp", "read_sum"):
            base = build_single_layer(receiver_count=2, k=1, d_model=2)
            nodes = tuple(
                dataclasses.replace(
                    node,
                    node_compute={
                        "type": "affine_residual",
                        "formula_id": "TEST-NODE-AFFINE-V1",
                        "bias": True,
                        "output_shape": [2],
                    },
                    emit={
                        "type": "softp",
                        "formula_id": "emit.softp.v1",
                        "output_shape": [2],
                    },
                )
                for node in base.nodes
            )
            score = (
                {
                    "type": "mlp",
                    "formula_id": "TEST-SCORE-MLP-V1",
                    "hidden_dim": 3,
                    "bias": True,
                }
                if score_type == "mlp"
                else {
                    "type": "read_sum",
                    "formula_id": "score.read-sum.v1",
                }
            )
            region = dataclasses.replace(base.regions[0], score=score)
            with self.subTest(score=score_type, emit="softp"):
                self._compare_paths(
                    _replace_plan(base, nodes=nodes, regions=(region,))
                )

        base = build_multi_entry_terminal(d_model=2)
        output_plan = _replace_plan(
            base,
            output_aggregate={
                "type": "node_softmax",
                "formula_id": "TEST-AGG-TERMINAL-SOFTMAX-V1",
                "output_shape": [2],
            },
        )
        self._compare_paths(output_plan)


if __name__ == "__main__":
    unittest.main()
