from __future__ import annotations

import dataclasses
import math
import unittest
from unittest import mock

import torch

from tide.builders import (
    build_chain,
    build_diamond,
    build_single_layer,
    build_singleton,
    build_unequal_path,
)
from tide.engine import (
    ExecutionContractError,
    LocalOperationError,
    SettleGraph,
    StateStore,
    UnsupportedPlanError,
)
from tide.ops import OperationExecutionError, ReceiverModule, safe_module_key
from tide.plan import PlanValidationError, bind_dtypes


def _replace_plan(plan, *, nodes=None, regions=None):
    return dataclasses.replace(
        plan,
        nodes=tuple(nodes if nodes is not None else plan.nodes),
        regions=tuple(regions if regions is not None else plan.regions),
    ).validate()


def _ema_plan(
    *,
    receiver_count: int = 2,
    profile: str = "BO",
    timing: str = "content",
    input_k: bool = False,
    d_model: int = 2,
):
    plan = build_single_layer(
        receiver_count=receiver_count, k=1, d_model=d_model
    )
    nodes = []
    for node in plan.nodes:
        if timing == "content":
            selector_read = {
                "type": "content",
                "formula_id": "read.selector.content.v1",
                "output_shape": [d_model],
            }
            selector_shape = (d_model,)
        else:
            selector_read = {
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
                "output_shape": [1],
            }
            selector_shape = (1,)
        nodes.append(
            dataclasses.replace(
                node,
                state_shape=(d_model,),
                state_owner=node.node_id,
                update={
                    "type": "ema",
                    "formula_id": "state.ema.v1",
                    "state_dim": d_model,
                    "decay": 0.5,
                    "state_shape": [d_model],
                },
                selector_read_shape=selector_shape,
                selector_read=selector_read,
            )
        )
    region = plan.regions[0]
    k_requested = (
        {
            "type": "input",
            "formula_id": "k.input.v1",
            "field": "requested_k",
            "minimum": 1,
            "maximum": receiver_count,
        }
        if input_k
        else {"type": "fixed", "formula_id": "k.fixed.v1", "value": 1}
    )
    region = dataclasses.replace(
        region,
        profile=profile,
        selector_timing=timing,
        k_max=receiver_count if input_k else 1,
        k_requested=k_requested,
        score={
            "type": "fixed",
            "formula_id": "TEST-SCORE-CONST-V1",
            "values_by_node": {
                node.node_id: float(index)
                for index, node in enumerate(plan.nodes)
            },
        },
    )
    return _replace_plan(plan, nodes=nodes, regions=[region])


def _assert_state_equal(test, left: StateStore, right: StateStore):
    test.assertEqual(left.next_position, right.next_position)
    test.assertEqual(set(left.values), set(right.values))
    for key in left.values:
        lhs = left.values[key]
        rhs = right.values[key]
        if lhs is None or rhs is None:
            test.assertIs(lhs, rhs)
        else:
            torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


class FormulaDispatchContractTests(unittest.TestCase):
    def _attention_summary_plan(self, formula_id: str):
        plan = build_singleton(d_model=2)
        node = dataclasses.replace(
            plan.nodes[0],
            state_shape=(2, 2, 2),
            state_owner=plan.nodes[0].node_id,
            update={
                "type": "attention_window",
                "formula_id": "state.attention-window.v1",
                "key_dim": 2,
                "value_dim": 2,
                "window": 2,
                "norm_eps": 1e-12,
                "state_shape": [2, 2, 2],
            },
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_summary_linear",
                "formula_id": formula_id,
                "out_dim": 1,
                "output_shape": [1],
            },
        )
        region = dataclasses.replace(
            plan.regions[0], profile="BO", selector_timing="post"
        )
        return _replace_plan(plan, nodes=(node,), regions=(region,))

    def test_known_formula_id_is_rejected_for_the_wrong_dispatch_type(self):
        plan = build_singleton(d_model=2)
        node = dataclasses.replace(
            plan.nodes[0],
            node_compute={
                "type": "identity",
                "formula_id": "TEST-NODE-SWIGLU-V1",
                "output_shape": [2],
            },
        )
        with self.assertRaisesRegex(
            PlanValidationError,
            "node_compute type 'identity' does not support formula_id",
        ):
            _replace_plan(plan, nodes=(node,))

    def test_unknown_formula_id_is_rejected_before_execution(self):
        plan = build_singleton(d_model=2)
        node = dataclasses.replace(
            plan.nodes[0],
            emit={
                "type": "hard",
                "formula_id": "unknown.emit.formula.v99",
                "output_shape": [2],
            },
        )
        with self.assertRaisesRegex(
            PlanValidationError,
            "emit type 'hard' does not support formula_id",
        ):
            _replace_plan(plan, nodes=(node,))

    def test_missing_formula_id_cannot_bypass_the_registry(self):
        plan = build_singleton(d_model=2)
        node = dataclasses.replace(
            plan.nodes[0], emit={"type": "hard", "output_shape": [2]}
        )
        with self.assertRaisesRegex(
            PlanValidationError,
            "emit type 'hard' must declare a nonempty string formula_id",
        ):
            _replace_plan(plan, nodes=(node,))

    def test_attention_summary_read_requires_its_distinct_formula_id(self):
        with self.assertRaisesRegex(
            PlanValidationError,
            "content_state_summary_linear.*TEST-READ-PROJ-V1",
        ):
            self._attention_summary_plan("TEST-READ-PROJ-V1")

        valid = SettleGraph(
            self._attention_summary_plan(
                "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1"
            )
        )
        self.assertEqual(
            valid.receiver("node.0").selector_read_type,
            "content_state_summary_linear",
        )

    def test_reference_dispatch_rejects_noncanonical_type_aliases(self):
        base = build_singleton(d_model=2)
        alias_nodes = (
            dataclasses.replace(
                base.nodes[0],
                aggregate={
                    "type": "learned_convex",
                    "formula_id": "TEST-AGG-EDGE-SOFTMAX-V1",
                    "output_shape": [2],
                },
            ),
            dataclasses.replace(
                base.nodes[0],
                node_compute={
                    "type": "double_residual_mlp",
                    "formula_id": "TEST-NODE-SWIGLU-V1",
                    "hidden_dim": 8,
                    "bias": True,
                    "output_shape": [2],
                },
            ),
        )
        for node, operation_type in zip(
            alias_nodes, ("learned_convex", "double_residual_mlp")
        ):
            with self.subTest(operation_type=operation_type):
                with self.assertRaisesRegex(
                    PlanValidationError,
                    "is not supported by the eager reference executor",
                ):
                    _replace_plan(base, nodes=(node,))

        with self.assertRaisesRegex(
            PlanValidationError,
            "is not supported by the eager reference executor",
        ):
            dataclasses.replace(
                base,
                output_aggregate={
                    "type": "learned_convex",
                    "formula_id": "TEST-AGG-TERMINAL-SOFTMAX-V1",
                    "output_shape": [2],
                },
            ).validate()

        constant_test_score = dataclasses.replace(
            base.regions[0],
            score={
                "type": "constant",
                "formula_id": "TEST-SCORE-CONST-V1",
                "value": 0.0,
            },
        )
        with self.assertRaisesRegex(
            PlanValidationError,
            "score type 'constant' does not support formula_id",
        ):
            _replace_plan(base, regions=(constant_test_score,))

    def test_semantically_explicit_custom_formula_is_a_capability_failure(self):
        base = build_singleton(d_model=2)
        node = dataclasses.replace(
            base.nodes[0],
            emit={
                "type": "custom",
                "formula": "Emit(m, p) = m",
                "output_shape": [2],
            },
        )
        custom = _replace_plan(base, nodes=(node,))
        with self.assertRaisesRegex(
            UnsupportedPlanError, "unsupported Plan capability"
        ):
            SettleGraph(custom)


class IdentityAndTopologyTests(unittest.TestCase):
    def test_typed_low_precision_plans_fail_during_construction(self):
        logical = build_singleton(d_model=2)
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                typed = bind_dtypes(
                    logical,
                    hidden=dtype,
                    parameter=dtype,
                    state=dtype,
                    readout=dtype,
                )
                with self.assertRaisesRegex(
                    UnsupportedPlanError, f"dtype '{dtype}'.*only float32 and float64"
                ):
                    SettleGraph(typed)

    def test_typed_plan_accepts_matching_dtype_and_rejects_mismatches(self):
        logical = build_singleton(d_model=2)
        typed = bind_dtypes(
            logical,
            hidden="float64",
            parameter="float64",
            state="float64",
            readout="float64",
        )
        model = SettleGraph(typed).double()
        result = model.interpret_token(
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
        )
        self.assertEqual(result.output.dtype, torch.float64)
        self.assertEqual(result.state.next_position, {"s": 1})

        with self.assertRaisesRegex(
            ExecutionContractError, "does not match typed Plan binding"
        ):
            model.interpret_token(
                torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                torch.tensor([True]),
                ["s"],
                torch.tensor([0]),
            )

        model_with_wrong_parameters = SettleGraph(typed)
        with self.assertRaisesRegex(ExecutionContractError, "parameter .*dtype"):
            model_with_wrong_parameters.interpret_token(
                torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                torch.tensor([True]),
                ["s"],
                torch.tensor([0]),
            )

        mixed = bind_dtypes(
            logical,
            hidden="float64",
            parameter="float32",
            state="float64",
            readout="float64",
        )
        with self.assertRaises(UnsupportedPlanError):
            SettleGraph(mixed)

        stateful_logical = _ema_plan(receiver_count=1, d_model=2)
        stateful_typed = bind_dtypes(
            stateful_logical,
            hidden="float64",
            parameter="float64",
            state="float64",
            readout="float64",
        )
        stateful_model = SettleGraph(stateful_typed).double()
        wrong_state = StateStore(
            values={
                ("s", stateful_logical.nodes[0].node_id): torch.zeros(
                    2, dtype=torch.float32
                )
            },
            next_position={"s": 1},
        )
        with self.assertRaisesRegex(
            ExecutionContractError, "floating state dtype must match hidden"
        ):
            stateful_model.interpret_token(
                torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                torch.tensor([True]),
                ["s"],
                torch.tensor([1]),
                state=wrong_state,
            )

    def test_identity_is_exact_for_all_manual_topologies(self):
        plans = [
            build_singleton(d_model=3),
            build_chain(length=4, d_model=3),
            build_diamond(d_model=3, branch_k=1),
            build_diamond(d_model=3, branch_k=2),
            build_unequal_path(d_model=3),
        ]
        hidden = torch.tensor(
            [[[1.0, 2.0, -1.0], [4.0, -2.0, 0.5]]],
            dtype=torch.float64,
        )
        execution = torch.ones((1, 2), dtype=torch.bool)
        positions = torch.tensor([[0, 1]], dtype=torch.int64)
        for plan in plans:
            with self.subTest(plan=plan.plan_id):
                model = SettleGraph(plan).double()
                model.make_identity()
                result = model.prefill(
                    hidden,
                    execution,
                    ["sequence"],
                    positions,
                    detach_at_end=False,
                    record_trace=True,
                )
                torch.testing.assert_close(result.output, hidden, atol=0, rtol=0)
                self.assertEqual(result.state.next_position, {"sequence": 2})
                self.assertTrue(result.trace.output_events)
                self.assertTrue(
                    all(
                        event.status in {"DATA", "CLOSED"}
                        for event in result.trace.edge_events
                    )
                )

    def test_padding_bypasses_graph_and_has_no_events_or_state(self):
        model = SettleGraph(build_singleton(d_model=2)).double()
        hidden = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=torch.float64,
        )
        execution = torch.tensor([[True, False, True]])
        positions = torch.tensor([[0, 999, 1]])
        result = model.prefill(
            hidden,
            execution,
            ["s"],
            positions,
            routing_stats_mask=torch.tensor([[True, False, False]]),
            lm_target_mask=torch.tensor([[False, False, True]]),
            detach_at_end=False,
            record_trace=True,
        )
        torch.testing.assert_close(result.output, hidden, atol=0, rtol=0)
        self.assertEqual(result.state.next_position, {"s": 2})
        self.assertEqual(
            [event.token_position for event in result.trace.output_events], [0, 1]
        )
        self.assertEqual(result.balance_stats.regions["region.0"].event_count, 1)

    def test_trace_is_canonical_after_batch_row_reordering(self):
        model = SettleGraph(build_singleton(d_model=1)).double()
        result = model.interpret_token(
            torch.tensor([[2.0], [1.0]], dtype=torch.float64),
            torch.tensor([True, True]),
            ["z", "a"],
            torch.tensor([0, 0]),
            record_trace=True,
        )
        self.assertEqual(
            [event.sequence_id for event in result.trace.output_events],
            ["a", "z"],
        )


class LocalOperationBoundaryTests(unittest.TestCase):
    @staticmethod
    def _execute(model, executor_name):
        if executor_name == "interpret_token":
            return model.interpret_token(
                torch.ones((1, 2), dtype=torch.float64),
                torch.ones((1,), dtype=torch.bool),
                ["sequence"],
                torch.zeros((1,), dtype=torch.int64),
            )
        if executor_name == "prefill_region_major":
            return model.prefill_region_major(
                torch.ones((1, 1, 2), dtype=torch.float64),
                torch.ones((1, 1), dtype=torch.bool),
                ["sequence"],
                torch.zeros((1, 1), dtype=torch.int64),
            )
        raise AssertionError(f"unknown test executor {executor_name!r}")

    def test_operation_execution_error_becomes_executor_owned_failure(self):
        for executor_name in ("interpret_token", "prefill_region_major"):
            with self.subTest(executor=executor_name):
                model = SettleGraph(build_singleton(d_model=2)).double()
                sentinel = OperationExecutionError(
                    f"injected local failure in {executor_name}"
                )
                with mock.patch.object(
                    ReceiverModule, "aggregate", side_effect=sentinel
                ):
                    with self.assertRaises(LocalOperationError) as raised:
                        self._execute(model, executor_name)
                self.assertIs(raised.exception.__cause__, sentinel)
                self.assertEqual(str(raised.exception), str(sentinel))

    def test_unknown_operation_exception_is_not_reclassified(self):
        for executor_name in ("interpret_token", "prefill_region_major"):
            with self.subTest(executor=executor_name):
                model = SettleGraph(build_singleton(d_model=2)).double()
                sentinel = KeyError(f"injected bug in {executor_name}")
                with mock.patch.object(
                    ReceiverModule, "aggregate", side_effect=sentinel
                ):
                    with self.assertRaises(KeyError) as raised:
                        self._execute(model, executor_name)
                self.assertIs(raised.exception, sentinel)


class StateAndTransactionTests(unittest.TestCase):
    def test_runtime_stable_ids_are_strict_and_fail_without_state_publication(self):
        model = SettleGraph(build_singleton(d_model=2)).double()
        token_hidden = torch.zeros((1, 2), dtype=torch.float64)
        token_mask = torch.tensor([True])
        token_positions = torch.tensor([0], dtype=torch.int64)
        prefill_hidden = token_hidden[:, None, :]
        prefill_mask = token_mask[:, None]
        prefill_positions = token_positions[:, None]
        invalid_ids = (7, "e\u0301", " leading", "trailing ", "bad\x00id", "\ud800")

        for invalid_id in invalid_ids:
            with self.subTest(kind="interpret-sequence", value=repr(invalid_id)):
                external = StateStore()
                with self.assertRaises(ExecutionContractError):
                    model.interpret_token(
                        token_hidden,
                        token_mask,
                        [invalid_id],
                        token_positions,
                        state=external,
                    )
                self.assertEqual(external, StateStore())

            for executor in (model.prefill, model.prefill_region_major):
                with self.subTest(
                    kind="prefill-sequence",
                    executor=executor.__name__,
                    value=repr(invalid_id),
                ):
                    external = StateStore()
                    with self.assertRaises(ExecutionContractError):
                        executor(
                            prefill_hidden,
                            prefill_mask,
                            [invalid_id],
                            prefill_positions,
                            state=external,
                        )
                    self.assertEqual(external, StateStore())

                with self.subTest(
                    kind="prefill-reset",
                    executor=executor.__name__,
                    value=repr(invalid_id),
                ):
                    external = StateStore(next_position={"kept": 3})
                    snapshot = StateStore(next_position={"kept": 3})
                    with self.assertRaises(ExecutionContractError):
                        executor(
                            prefill_hidden,
                            prefill_mask,
                            ["sequence"],
                            prefill_positions,
                            state=external,
                            reset_sequence_ids=[invalid_id],
                        )
                    self.assertEqual(external, snapshot)

            with self.subTest(kind="state-reset", value=repr(invalid_id)):
                external = StateStore(next_position={"kept": 3})
                with self.assertRaises(ExecutionContractError):
                    external.reset([invalid_id])
                self.assertEqual(external.next_position, {"kept": 3})

            with self.subTest(kind="external-state", value=repr(invalid_id)):
                external = StateStore(next_position={invalid_id: 0})
                snapshot = dict(external.next_position)
                with self.assertRaises(ExecutionContractError):
                    model.interpret_token(
                        token_hidden,
                        token_mask,
                        ["sequence"],
                        token_positions,
                        state=external,
                    )
                self.assertEqual(external.next_position, snapshot)

    def test_sd_updates_only_active_while_bo_updates_all_reached(self):
        hidden = torch.tensor([[[1.0, -1.0]]], dtype=torch.float64)
        execution = torch.ones((1, 1), dtype=torch.bool)
        positions = torch.zeros((1, 1), dtype=torch.int64)
        states_by_profile = {}
        for profile in ("SD", "BO"):
            model = SettleGraph(_ema_plan(profile=profile)).double()
            for receiver in model.receivers.values():
                with torch.no_grad():
                    receiver.ema_observe.weight.copy_(torch.eye(2))
                    receiver.ema_observe.bias.zero_()
            result = model.prefill(
                hidden,
                execution,
                ["s"],
                positions,
                detach_at_end=False,
                record_trace=True,
            )
            states_by_profile[profile] = result.state
        self.assertEqual(len(states_by_profile["SD"].values), 1)
        self.assertEqual(len(states_by_profile["BO"].values), 2)
        self.assertIn(("s", "node.0001"), states_by_profile["SD"].values)

    def test_whole_prefill_equals_every_token_chunk_with_reset(self):
        plan = _ema_plan(receiver_count=2, profile="BO", d_model=2)
        model = SettleGraph(plan).double()
        torch.manual_seed(4)
        hidden = torch.randn((1, 4, 2), dtype=torch.float64)
        execution = torch.ones((1, 4), dtype=torch.bool)
        positions = torch.arange(4, dtype=torch.int64).reshape(1, 4)
        whole = model.prefill(
            hidden,
            execution,
            ["s"],
            positions,
            detach_at_end=False,
        )
        state = StateStore()
        outputs = []
        for index in range(4):
            part = model.prefill(
                hidden[:, index : index + 1],
                execution[:, index : index + 1],
                ["s"],
                positions[:, index : index + 1],
                state=state,
                detach_at_end=False,
            )
            outputs.append(part.output)
            state = part.state
        torch.testing.assert_close(
            torch.cat(outputs, dim=1), whole.output, atol=1e-12, rtol=1e-12
        )
        _assert_state_equal(self, state, whole.state)

        reset = model.prefill(
            hidden[:, :1],
            execution[:, :1],
            ["s"],
            positions[:, :1],
            state=state,
            reset_sequence_ids=["s"],
            detach_at_end=False,
        )
        fresh = model.prefill(
            hidden[:, :1],
            execution[:, :1],
            ["s"],
            positions[:, :1],
            detach_at_end=False,
        )
        torch.testing.assert_close(reset.output, fresh.output, atol=0, rtol=0)
        _assert_state_equal(self, reset.state, fresh.state)

    def test_window_attention_state_uses_global_positions_and_bounded_history(self):
        plan = _ema_plan(receiver_count=2, profile="BO", d_model=2)
        nodes = [
            dataclasses.replace(
                node,
                state_shape=(2, 2, 2),
                update={
                    "type": "attention_window",
                    "formula_id": "state.attention-window.v1",
                    "key_dim": 2,
                    "value_dim": 2,
                    "window": 2,
                    "norm_eps": 1e-12,
                    "state_shape": [2, 2, 2],
                },
            )
            for node in plan.nodes
        ]
        model = SettleGraph(_replace_plan(plan, nodes=nodes)).double()
        hidden = torch.tensor(
            [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]],
            dtype=torch.float64,
        )
        result = model.prefill(
            hidden,
            torch.ones((1, 3), dtype=torch.bool),
            ["s"],
            torch.tensor([[0, 1, 2]]),
            detach_at_end=False,
        )
        self.assertEqual(len(result.state.values), 2)
        for state in result.state.values.values():
            self.assertEqual(state.positions.tolist(), [1, 2])
            self.assertEqual(tuple(state.keys.shape), (2, 2))

    def test_gated_deltanet_state_is_finite_and_has_declared_shape(self):
        plan = _ema_plan(receiver_count=2, profile="BO", d_model=2)
        nodes = [
            dataclasses.replace(
                node,
                state_shape=(2, 3),
                update={
                    "type": "gdn",
                    "formula_id": "state.gdn.v1",
                    "key_dim": 2,
                    "value_dim": 3,
                    "norm_eps": 1e-12,
                    "state_shape": [2, 3],
                },
            )
            for node in plan.nodes
        ]
        model = SettleGraph(_replace_plan(plan, nodes=nodes)).double()
        result = model.prefill(
            torch.tensor([[[1.0, 2.0], [-1.0, 0.5]]], dtype=torch.float64),
            torch.ones((1, 2), dtype=torch.bool),
            ["s"],
            torch.tensor([[0, 1]]),
            detach_at_end=False,
        )
        for state in result.state.values.values():
            self.assertEqual(tuple(state.shape), (2, 3))
            self.assertTrue(bool(torch.isfinite(state).all()))

    def test_chunk_detach_only_cuts_cross_chunk_gradient(self):
        plan = _ema_plan(receiver_count=2, profile="BO", d_model=2)
        nodes = [
            dataclasses.replace(
                node,
                ffn_read={
                    "type": "state_default",
                    "formula_id": "read.ffn.ema.v1",
                    "output_shape": [2],
                },
                node_compute={
                    "type": "double_residual_swiglu",
                    "formula_id": "TEST-NODE-SWIGLU-V1",
                    "hidden_dim": 2,
                    "bias": True,
                    "output_shape": [2],
                },
            )
            for node in plan.nodes
        ]
        model = SettleGraph(_replace_plan(plan, nodes=nodes)).double()
        with torch.no_grad():
            for receiver in model.receivers.values():
                receiver.state_out.weight.copy_(torch.eye(2, dtype=torch.float64))
                receiver.down_proj.weight.zero_()
                receiver.down_proj.bias.zero_()

        def first_input_gradient(detach_at_end):
            first_hidden = torch.tensor(
                [[[1.0, 2.0]]], dtype=torch.float64, requires_grad=True
            )
            first = model.prefill(
                first_hidden,
                torch.tensor([[True]]),
                ["s"],
                torch.tensor([[0]]),
                detach_at_end=detach_at_end,
            )
            second = model.prefill(
                torch.tensor([[[0.5, -1.0]]], dtype=torch.float64),
                torch.tensor([[True]]),
                ["s"],
                torch.tensor([[1]]),
                state=first.state,
                detach_at_end=False,
            )
            return torch.autograd.grad(
                second.output.sum(), first_hidden, allow_unused=True
            )[0]

        self.assertIsNone(first_input_gradient(True))
        connected = first_input_gradient(False)
        self.assertIsNotNone(connected)
        self.assertGreater(float(connected.abs().sum()), 0.0)

    def test_late_invalid_runtime_k_rolls_back_whole_call(self):
        plan = _ema_plan(profile="BO", input_k=True)
        model = SettleGraph(plan).double()
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
        execution = torch.ones((1, 2), dtype=torch.bool)
        positions = torch.tensor([[0, 1]])
        external = StateStore()
        with self.assertRaisesRegex(ExecutionContractError, "outside"):
            model.prefill(
                hidden,
                execution,
                ["s"],
                positions,
                state=external,
                requested_k={"region.0": torch.tensor([[1, 3]])},
                detach_at_end=False,
            )
        self.assertEqual(external.values, {})
        self.assertEqual(external.next_position, {})

    def test_positions_masks_and_duplicate_sequence_fail_before_execution(self):
        model = SettleGraph(build_singleton(d_model=2)).double()
        hidden = torch.zeros((2, 1, 2), dtype=torch.float64)
        execution = torch.ones((2, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ExecutionContractError, "unique"):
            model.prefill(
                hidden,
                execution,
                ["same", "same"],
                torch.zeros((2, 1), dtype=torch.int64),
            )
        with self.assertRaisesRegex(ExecutionContractError, "next required"):
            model.prefill(
                hidden[:1],
                execution[:1],
                ["s"],
                torch.tensor([[1]]),
            )
        with self.assertRaisesRegex(ExecutionContractError, "subset"):
            model.prefill(
                hidden[:1],
                torch.tensor([[False]]),
                ["s"],
                torch.tensor([[0]]),
                lm_target_mask=torch.tensor([[True]]),
            )


class RoutingLossAndGradientTests(unittest.TestCase):
    def _trainable_plan(self, *, emit_type: str = "hst"):
        plan = build_single_layer(receiver_count=2, k=1, d_model=2)
        nodes = [
            dataclasses.replace(
                node,
                selector_read={
                    "type": "content",
                    "formula_id": "read.selector.content.v1",
                    "output_shape": [2],
                },
                selector_read_shape=(2,),
                node_compute={
                    "type": "affine_residual",
                    "formula_id": "TEST-NODE-AFFINE-V1",
                    "bias": True,
                    "output_shape": [2],
                },
                emit={
                    "type": emit_type,
                    "formula_id": f"emit.{emit_type}.v1",
                    **({"zeta": 1.0} if emit_type == "hst" else {}),
                    "output_shape": [2],
                },
            )
            for node in plan.nodes
        ]
        region = dataclasses.replace(
            plan.regions[0],
            score={
                "type": "linear",
                "formula_id": "TEST-SCORE-LINEAR-V1",
                "bias": True,
            },
        )
        return _replace_plan(plan, nodes=nodes, regions=[region])

    def _configure_selector(self, model):
        selector = model.selector("region.0")
        with torch.no_grad():
            for node_id in ("node.0000", "node.0001"):
                layer = selector.linears[safe_module_key(node_id)]
                layer.weight.zero_()
                layer.bias.fill_(0.0 if node_id.endswith("0") else 2.0)

    def test_hst_has_selector_gradient_and_inactive_compute_is_disconnected(self):
        model = SettleGraph(self._trainable_plan(emit_type="hst")).double()
        self._configure_selector(model)
        hidden = torch.tensor([[1.0, -2.0]], dtype=torch.float64, requires_grad=True)
        result = model.interpret_token(
            hidden,
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
            routing_stats_mask=torch.tensor([False]),
            record_trace=True,
        )
        result.output.sum().backward()
        selector = model.selector("region.0")
        self.assertIsNotNone(
            selector.linears[safe_module_key("node.0001")].bias.grad
        )
        inactive = model.receiver("node.0000").down_proj
        active = model.receiver("node.0001").down_proj
        self.assertIsNone(inactive.weight.grad)
        self.assertIsNotNone(active.weight.grad)

    def test_exact_tie_selects_lexically_first_node(self):
        plan = build_single_layer(receiver_count=2, k=1, d_model=2)
        region = dataclasses.replace(
            plan.regions[0],
            score={
                "type": "fixed",
                "formula_id": "TEST-SCORE-CONST-V1",
                "values_by_node": {
                    "node.0000": 1.0,
                    "node.0001": 1.0,
                },
            },
        )
        model = SettleGraph(_replace_plan(plan, regions=[region])).double()
        result = model.interpret_token(
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
            record_trace=True,
        )
        self.assertEqual(
            result.trace.region_events[0].active_node_ids, ("node.0000",)
        )

    def test_hard_emit_has_no_main_task_selector_gradient(self):
        model = SettleGraph(self._trainable_plan(emit_type="hard")).double()
        self._configure_selector(model)
        result = model.interpret_token(
            torch.tensor([[1.0, -2.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
            routing_stats_mask=torch.tensor([False]),
        )
        result.output.sum().backward()
        selector = model.selector("region.0")
        self.assertTrue(
            all(layer.bias.grad is None for layer in selector.linears.values())
        )

    def test_balance_sufficient_statistics_merge_and_vjp(self):
        model = SettleGraph(self._trainable_plan(emit_type="hard")).double()
        self._configure_selector(model)
        hidden = torch.tensor(
            [[[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]]],
            dtype=torch.float64,
        )
        execution = torch.ones((1, 3), dtype=torch.bool)
        positions = torch.arange(3).reshape(1, 3)
        full = model.prefill(
            hidden,
            execution,
            ["s"],
            positions,
            detach_at_end=False,
        )
        full_loss = full.balance_loss
        full_grad = torch.autograd.grad(
            full_loss,
            model.selector("region.0").linears[
                safe_module_key("node.0001")
            ].bias,
            retain_graph=False,
        )[0]

        first = model.prefill(
            hidden[:, :1],
            execution[:, :1],
            ["s"],
            positions[:, :1],
            detach_at_end=False,
        )
        second = model.prefill(
            hidden[:, 1:],
            execution[:, 1:],
            ["s"],
            positions[:, 1:],
            state=first.state,
            detach_at_end=False,
        )
        merged = first.balance_stats.merge(second.balance_stats)
        merged_grad = torch.autograd.grad(
            merged.loss(),
            model.selector("region.0").linears[
                safe_module_key("node.0001")
            ].bias,
        )[0]
        torch.testing.assert_close(merged.loss(), full_loss, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(merged_grad, full_grad, atol=1e-12, rtol=1e-12)
        stats = merged.regions["region.0"]
        self.assertEqual(stats.event_count, 3)
        self.assertEqual(stats.competition_count, 3)
        torch.testing.assert_close(
            stats.availability_sum,
            torch.tensor([1.5, 1.5], dtype=torch.float64),
            atol=0,
            rtol=0,
        )

    def test_post_selector_probability_vjp_reaches_update_but_pre_does_not(self):
        gradients = {}
        for timing in ("pre", "post"):
            plan = _ema_plan(profile="BO", timing=timing)
            plan = _replace_plan(
                plan,
                regions=[
                    dataclasses.replace(
                        plan.regions[0],
                        score={
                            "type": "linear",
                            "formula_id": "TEST-SCORE-LINEAR-V1",
                            "bias": True,
                        },
                    )
                ],
            )
            model = SettleGraph(plan).double()
            selector = model.selector("region.0")
            with torch.no_grad():
                for node_id in ("node.0000", "node.0001"):
                    read = model.receiver(node_id).selector_read_linear
                    read.weight.fill_(0.25)
                    read.bias.zero_()
                    score = selector.linears[safe_module_key(node_id)]
                    score.weight.fill_(1.0 if node_id.endswith("0") else -0.5)
                    score.bias.zero_()
            result = model.interpret_token(
                torch.tensor([[1.0, 2.0]], dtype=torch.float64),
                torch.tensor([True]),
                ["s"],
                torch.tensor([0]),
                routing_stats_mask=torch.tensor([False]),
                record_trace=True,
            )
            probability = result.trace.region_events[0].probabilities[0]
            probability.backward()
            gradients[timing] = model.receiver("node.0000").ema_observe.weight.grad
        self.assertIsNone(gradients["pre"])
        self.assertIsNotNone(gradients["post"])
        self.assertGreater(float(gradients["post"].abs().sum()), 0.0)

    def test_empty_balance_loss_preserves_executor_dtype_and_device(self):
        model = SettleGraph(build_singleton(d_model=2)).double()
        result = model.interpret_token(
            torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
            routing_stats_mask=torch.tensor([False]),
        )
        self.assertEqual(result.balance_loss.dtype, torch.float64)
        self.assertEqual(result.balance_loss.device.type, "cpu")
        self.assertEqual(result.balance_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
