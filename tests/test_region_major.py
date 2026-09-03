from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any, Mapping
import unittest
from unittest import mock

import torch
from torch import Tensor

from tide.builders import (
    build_chain,
    build_diamond,
    build_single_layer,
    build_singleton,
    build_unequal_path,
)
from tide.engine import ExecutionContractError, SettleGraph, StateStore
from tide.ops import safe_module_key


def _assert_nested_close(
    case: unittest.TestCase,
    expected: Any,
    actual: Any,
    *,
    atol: float = 1e-11,
    rtol: float = 1e-11,
) -> None:
    if isinstance(expected, Tensor):
        case.assertIsInstance(actual, Tensor)
        if torch.is_floating_point(expected):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return
    if dataclasses.is_dataclass(expected):
        case.assertTrue(dataclasses.is_dataclass(actual))
        case.assertIs(type(actual), type(expected))
        for field in dataclasses.fields(expected):
            _assert_nested_close(
                case,
                getattr(expected, field.name),
                getattr(actual, field.name),
                atol=atol,
                rtol=rtol,
            )
        return
    if isinstance(expected, Mapping):
        case.assertIsInstance(actual, Mapping)
        case.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_nested_close(
                case, expected[key], actual[key], atol=atol, rtol=rtol
            )
        return
    if isinstance(expected, (tuple, list)):
        case.assertIsInstance(actual, type(expected))
        case.assertEqual(len(actual), len(expected))
        for expected_item, actual_item in zip(expected, actual):
            _assert_nested_close(
                case, expected_item, actual_item, atol=atol, rtol=rtol
            )
        return
    case.assertEqual(actual, expected)


def _ema_singleton_plan():
    plan = build_singleton(d_model=2)
    node = plan.nodes[0]
    node = replace(
        node,
        state_shape=(2,),
        state_owner=node.node_id,
        update={
            "type": "ema",
            "formula_id": "state.ema.v1",
            "state_dim": 2,
            "decay": 0.5,
            "state_shape": [2],
        },
        ffn_read={
            "type": "state_default",
            "formula_id": "read.ffn.ema.v1",
            "output_shape": [2],
        },
        node_compute={
            "type": "double_residual_swiglu",
            "formula_id": "TEST-NODE-SWIGLU-V1",
            "hidden_dim": 3,
            "bias": True,
            "output_shape": [2],
        },
    )
    region = replace(plan.regions[0], profile="BO", selector_timing="content")
    return replace(
        plan,
        plan_id="ema-singleton",
        nodes=(node,),
        regions=(region,),
    ).validate()


def _configure_ema_graph(graph: SettleGraph) -> None:
    receiver = graph.receiver(graph.plan.nodes[0].node_id)
    with torch.no_grad():
        receiver.ema_observe.weight.copy_(torch.eye(2, dtype=torch.float64))
        receiver.ema_observe.bias.copy_(
            torch.tensor([0.1, -0.2], dtype=torch.float64)
        )
        receiver.state_out.weight.copy_(
            torch.tensor([[1.0, 0.25], [-0.5, 0.75]], dtype=torch.float64)
        )
        receiver.down_proj.weight.zero_()
        receiver.down_proj.bias.zero_()


def _stateful_input_k_plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=2)
    nodes = []
    for node in plan.nodes:
        nodes.append(
            replace(
                node,
                state_shape=(2,),
                state_owner=node.node_id,
                update={
                    "type": "ema",
                    "formula_id": "state.ema.v1",
                    "state_dim": 2,
                    "decay": 0.5,
                    "state_shape": [2],
                },
            )
        )
    region = replace(
        plan.regions[0],
        profile="BO",
        selector_timing="content",
        k_max=2,
        k_requested={
            "type": "input",
            "formula_id": "k.input.v1",
            "field": "requested_k",
            "minimum": 1,
            "maximum": 2,
        },
    )
    return replace(
        plan,
        plan_id="stateful-input-k",
        nodes=tuple(nodes),
        regions=(region,),
    ).validate()


def _optional_downstream_input_k_plan():
    plan = build_unequal_path(d_model=2)
    nodes = tuple(
        replace(node, forced_active=False)
        if node.node_id == "node.long.1"
        else node
        for node in plan.nodes
    )
    regions = []
    for region in plan.regions:
        if region.region_id == "region.split":
            region = replace(
                region,
                k_max=1,
                k_requested={
                    "type": "fixed",
                    "formula_id": "k.fixed.v1",
                    "value": 1,
                },
                score={
                    "type": "linear",
                    "formula_id": "TEST-SCORE-LINEAR-V1",
                    "bias": True,
                },
            )
        elif region.region_id == "region.long":
            region = replace(
                region,
                k_max=1,
                k_requested={
                    "type": "input",
                    "formula_id": "k.input.v1",
                    "field": "requested_k",
                    "minimum": 1,
                    "maximum": 1,
                },
            )
        regions.append(region)
    return replace(plan, nodes=nodes, regions=tuple(regions)).validate()


def _unreached_forced_singleton_plan():
    plan = build_unequal_path(d_model=2)
    regions = tuple(
        replace(
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
                "values_by_node": {
                    "node.long.0": -1.0,
                    "node.short": 1.0,
                },
            },
        )
        if region.region_id == "region.split"
        else region
        for region in plan.regions
    )
    return replace(
        plan,
        plan_id="unreached-forced-singleton",
        regions=regions,
    ).validate()


def _backward_plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=3)
    nodes = tuple(
        replace(
            node,
            node_compute={
                "type": "affine_residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
                "bias": True,
                "output_shape": [3],
            },
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 0.7,
                "output_shape": [3],
            },
        )
        for node in plan.nodes
    )
    region = replace(
        plan.regions[0],
        score={
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
        },
    )
    return replace(
        plan,
        plan_id="region-major-backward",
        nodes=nodes,
        regions=(region,),
    ).validate()


class RegionMajorDifferentialTests(unittest.TestCase):
    def test_forced_singleton_skips_read_score_softmax_and_topk(self):
        hidden = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64)
        execution_mask = torch.ones((1, 1), dtype=torch.bool)
        positions = torch.zeros((1, 1), dtype=torch.int64)

        fixed_plan = build_singleton(d_model=2)
        fixed_region = replace(
            fixed_plan.regions[0],
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {fixed_plan.nodes[0].node_id: 7.25},
            },
        )
        fixed_plan = replace(
            fixed_plan,
            plan_id="forced-singleton-fixed-score",
            regions=(fixed_region,),
        ).validate()
        read_sum_plan = build_singleton(d_model=2)
        read_sum_region = replace(
            read_sum_plan.regions[0],
            score={
                "type": "read_sum",
                "formula_id": "score.read-sum.v1",
            },
        )
        read_sum_plan = replace(
            read_sum_plan,
            plan_id="forced-singleton-read-sum-score",
            regions=(read_sum_region,),
        ).validate()

        for plan in (fixed_plan, read_sum_plan):
            with self.subTest(score=plan.regions[0].score["type"]):
                graph = SettleGraph(plan).double()
                receiver = graph.receiver("node.0")
                selector = graph.selector("region.0")
                with mock.patch.object(
                    receiver,
                    "selector_read",
                    wraps=receiver.selector_read,
                ) as selector_read, mock.patch.object(
                    selector,
                    "forward",
                    wraps=selector.forward,
                ) as score, mock.patch(
                    "tide.engine.torch.softmax",
                    side_effect=AssertionError(
                        "forced-active selection must not depend on softmax"
                    ),
                ), mock.patch(
                    "tide.engine.deterministic_topk_mask",
                    side_effect=AssertionError(
                        "forced-active selection must not depend on Top-K"
                    ),
                ):
                    token_major = graph.prefill(
                        hidden,
                        execution_mask,
                        ["seq"],
                        positions,
                        detach_at_end=False,
                        record_trace=True,
                    )
                    region_major = graph.prefill_region_major(
                        hidden,
                        execution_mask,
                        ["seq"],
                        positions,
                        detach_at_end=False,
                        record_trace=True,
                    )

                self.assertEqual(selector_read.call_count, 0)
                self.assertEqual(score.call_count, 0)
                _assert_nested_close(self, token_major, region_major)
                for result in (token_major, region_major):
                    assert result.trace is not None
                    region_event = result.trace.region_events[0]
                    node_event = result.trace.node_events[0]
                    self.assertIsNone(region_event.logits)
                    torch.testing.assert_close(
                        region_event.probabilities,
                        torch.ones(1, dtype=torch.float64),
                        atol=0,
                        rtol=0,
                    )
                    self.assertIsNone(region_event.requested_k)
                    self.assertIsNone(region_event.effective_k)
                    self.assertIsNone(region_event.top_k_node_ids)
                    self.assertEqual(region_event.active_node_ids, ("node.0",))
                    self.assertTrue(region_event.forced_active)
                    self.assertIsNone(node_event.selector_read)
                    self.assertIsNone(node_event.logit)
                    torch.testing.assert_close(
                        node_event.probability,
                        torch.tensor(1.0, dtype=torch.float64),
                        atol=0,
                        rtol=0,
                    )

    def test_forced_singleton_does_not_read_nonfinite_unused_score(self):
        plan = build_singleton(d_model=2)
        region = replace(
            plan.regions[0],
            score={
                "type": "linear",
                "formula_id": "TEST-SCORE-LINEAR-V1",
                "bias": True,
            },
        )
        plan = replace(
            plan,
            plan_id="forced-singleton-nonfinite-score",
            regions=(region,),
        ).validate()
        graph = SettleGraph(plan).double()
        score = graph.selector("region.0").linears[safe_module_key("node.0")]
        with torch.no_grad():
            score.weight.zero_()
            score.bias.fill_(float("inf"))

        hidden = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64)
        execution_mask = torch.ones((1, 1), dtype=torch.bool)
        positions = torch.zeros((1, 1), dtype=torch.int64)
        for executor in (graph.prefill, graph.prefill_region_major):
            with self.subTest(executor=executor.__name__):
                result = executor(
                    hidden,
                    execution_mask,
                    ["seq"],
                    positions,
                    detach_at_end=False,
                    record_trace=True,
                )
                torch.testing.assert_close(result.output, hidden, atol=0, rtol=0)
                assert result.trace is not None
                self.assertIsNone(result.trace.region_events[0].logits)

    def test_identity_chain_and_diamond_match_token_major_exact_trace(self):
        torch.manual_seed(11)
        hidden = torch.randn(2, 3, 4, dtype=torch.float64)
        execution_mask = torch.ones((2, 3), dtype=torch.bool)
        positions = torch.arange(3, dtype=torch.int64).repeat(2, 1)

        for plan in (
            build_chain(length=4, d_model=4),
            build_diamond(d_model=4, branch_k=1),
            build_diamond(d_model=4, branch_k=2),
        ):
            with self.subTest(plan=plan.plan_id):
                graph = SettleGraph(plan).double()
                graph.make_identity()
                token_major = graph.prefill(
                    hidden,
                    execution_mask,
                    ["seq-b", "seq-a"],
                    positions,
                    detach_at_end=False,
                    record_trace=True,
                )
                with mock.patch.object(
                    SettleGraph,
                    "interpret_token",
                    side_effect=AssertionError(
                        "region-major execution called interpret_token"
                    ),
                ):
                    region_major = graph.prefill_region_major(
                        hidden,
                        execution_mask,
                        ["seq-b", "seq-a"],
                        positions,
                        detach_at_end=False,
                        record_trace=True,
                    )

                torch.testing.assert_close(
                    region_major.output, hidden, atol=0, rtol=0
                )
                _assert_nested_close(self, token_major, region_major)

    def test_ema_state_and_post_update_outputs_match(self):
        torch.manual_seed(23)
        graph = SettleGraph(_ema_singleton_plan()).double()
        _configure_ema_graph(graph)
        hidden = torch.tensor(
            [
                [[1.0, 0.5], [0.25, -1.0], [2.0, 1.5], [-0.5, 0.75]],
                [[-1.0, 0.25], [0.75, 1.25], [0.5, -0.75], [1.5, 0.5]],
            ],
            dtype=torch.float64,
        )
        execution_mask = torch.ones((2, 4), dtype=torch.bool)
        positions = torch.arange(4, dtype=torch.int64).repeat(2, 1)

        token_major = graph.prefill(
            hidden,
            execution_mask,
            ["left", "right"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        region_major = graph.prefill_region_major(
            hidden,
            execution_mask,
            ["left", "right"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )

        _assert_nested_close(self, token_major, region_major)
        self.assertFalse(torch.equal(region_major.output, hidden))
        for value in region_major.state.values.values():
            self.assertIsInstance(value, Tensor)
            self.assertGreater(float(value.abs().sum().item()), 0.0)

    def test_masks_and_chunk_carry_match_full_prefill(self):
        graph = SettleGraph(_ema_singleton_plan()).double()
        _configure_ema_graph(graph)
        hidden = torch.tensor(
            [
                [[0.5, 1.0], [1.0, -0.5], [0.25, 0.75], [-1.0, 0.5], [2.0, -1.0]],
                [[-0.5, 0.25], [1.5, 0.75], [9.0, 8.0], [7.0, 6.0], [5.0, 4.0]],
            ],
            dtype=torch.float64,
        )
        execution_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, False, False, False]]
        )
        lm_target_mask = torch.tensor(
            [[False, True, True, True, True], [False, True, False, False, False]]
        )
        routing_stats_mask = torch.tensor(
            [[True, False, True, False, True], [False, True, False, False, False]]
        )
        positions = torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 90, 91, 92]], dtype=torch.int64
        )
        sequence_ids = ["long", "short"]

        token_major = graph.prefill(
            hidden,
            execution_mask,
            sequence_ids,
            positions,
            lm_target_mask=lm_target_mask,
            routing_stats_mask=routing_stats_mask,
        )
        full = graph.prefill_region_major(
            hidden,
            execution_mask,
            sequence_ids,
            positions,
            lm_target_mask=lm_target_mask,
            routing_stats_mask=routing_stats_mask,
        )
        first = graph.prefill_region_major(
            hidden[:, :2],
            execution_mask[:, :2],
            sequence_ids,
            positions[:, :2],
            lm_target_mask=lm_target_mask[:, :2],
            routing_stats_mask=routing_stats_mask[:, :2],
        )
        second = graph.prefill_region_major(
            hidden[:, 2:],
            execution_mask[:, 2:],
            sequence_ids,
            positions[:, 2:],
            state=first.state,
            lm_target_mask=lm_target_mask[:, 2:],
            routing_stats_mask=routing_stats_mask[:, 2:],
        )
        empty_token_major = graph.prefill(
            hidden[:, 5:],
            execution_mask[:, 5:],
            sequence_ids,
            positions[:, 5:],
            state=second.state,
            lm_target_mask=lm_target_mask[:, 5:],
            routing_stats_mask=routing_stats_mask[:, 5:],
        )
        empty_region_major = graph.prefill_region_major(
            hidden[:, 5:],
            execution_mask[:, 5:],
            sequence_ids,
            positions[:, 5:],
            state=second.state,
            lm_target_mask=lm_target_mask[:, 5:],
            routing_stats_mask=routing_stats_mask[:, 5:],
        )

        _assert_nested_close(self, token_major, full)
        torch.testing.assert_close(
            torch.cat((first.output, second.output), dim=1),
            full.output,
            atol=1e-11,
            rtol=1e-11,
        )
        _assert_nested_close(self, full.state, second.state)
        _assert_nested_close(
            self, full.balance_stats, first.balance_stats.merge(second.balance_stats)
        )
        torch.testing.assert_close(
            full.output[1, 2:], hidden[1, 2:], atol=0, rtol=0
        )
        self.assertEqual(full.state.next_position, {"long": 5, "short": 2})
        self.assertEqual(empty_region_major.output.shape, (2, 0, 2))
        _assert_nested_close(self, empty_token_major, empty_region_major)
        _assert_nested_close(self, second.state, empty_region_major.state)

    def test_failure_discards_staged_state_positions_and_reset(self):
        graph = SettleGraph(_stateful_input_k_plan()).double()
        node_ids = tuple(node.node_id for node in graph.plan.nodes)
        original = StateStore(
            values={
                ("seq", node_id): torch.tensor(
                    [index + 1.0, -(index + 2.0)], dtype=torch.float64
                )
                for index, node_id in enumerate(node_ids)
            },
            next_position={"seq": 8},
        )
        snapshot = StateStore(
            values={key: value.clone() for key, value in original.values.items()},
            next_position=dict(original.next_position),
        )
        hidden = torch.tensor(
            [[[1.0, 0.5], [0.25, -1.0]]], dtype=torch.float64
        )
        execution_mask = torch.ones((1, 2), dtype=torch.bool)
        positions = torch.tensor([[0, 1]], dtype=torch.int64)
        requested_k = {
            "region.0": torch.tensor([[1, 3]], dtype=torch.int64)
        }

        with self.assertRaisesRegex(ExecutionContractError, "outside"):
            graph.prefill_region_major(
                hidden,
                execution_mask,
                ["seq"],
                positions,
                state=original,
                requested_k=requested_k,
                reset_sequence_ids=["seq"],
                detach_at_end=False,
            )

        _assert_nested_close(self, snapshot, original)

    def test_empty_candidate_event_does_not_read_runtime_k(self):
        graph = SettleGraph(_optional_downstream_input_k_plan()).double()
        split_selector = graph.selector("region.split")
        with torch.no_grad():
            long_score = split_selector.linears[safe_module_key("node.long.0")]
            short_score = split_selector.linears[safe_module_key("node.short")]
            long_score.weight.copy_(
                torch.tensor([[1.0, 0.0]], dtype=torch.float64)
            )
            short_score.weight.copy_(
                torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
            )
            long_score.bias.zero_()
            short_score.bias.zero_()

        execution_mask = torch.ones((1, 1), dtype=torch.bool)
        positions = torch.zeros((1, 1), dtype=torch.int64)
        invalid_placeholder = {
            "region.long": torch.tensor([[2]], dtype=torch.int64)
        }

        # Negative first coordinate selects the short branch, so region.long
        # has no candidates.  Its out-of-range integer is only a placeholder
        # and must not be resolved or range-checked for this event.
        empty_hidden = torch.tensor([[[-1.0, 0.5]]], dtype=torch.float64)
        token_major = graph.prefill(
            empty_hidden,
            execution_mask,
            ["empty"],
            positions,
            requested_k=invalid_placeholder,
            detach_at_end=False,
            record_trace=True,
        )
        region_major = graph.prefill_region_major(
            empty_hidden,
            execution_mask,
            ["empty"],
            positions,
            requested_k=invalid_placeholder,
            detach_at_end=False,
            record_trace=True,
        )
        _assert_nested_close(self, token_major, region_major)
        long_event = next(
            event
            for event in token_major.trace.region_events
            if event.region_id == "region.long"
        )
        self.assertEqual(long_event.candidate_node_ids, ())
        self.assertIsNone(long_event.requested_k)
        self.assertIsNone(long_event.effective_k)
        self.assertIsNone(long_event.top_k_node_ids)

        # Positive first coordinate selects the long branch.  The same value
        # is now consumed and must fail instead of being clipped to k_max.
        reached_hidden = torch.tensor([[[1.0, 0.5]]], dtype=torch.float64)
        for executor in (graph.prefill, graph.prefill_region_major):
            with self.subTest(executor=executor.__name__):
                with self.assertRaisesRegex(ExecutionContractError, "outside"):
                    executor(
                        reached_hidden,
                        execution_mask,
                        ["reached"],
                        positions,
                        requested_k=invalid_placeholder,
                        detach_at_end=False,
                    )

    def test_unreached_forced_singleton_has_an_empty_absent_trace(self):
        graph = SettleGraph(_unreached_forced_singleton_plan()).double()
        hidden = torch.tensor([[[1.0, -0.5]]], dtype=torch.float64)
        execution_mask = torch.ones((1, 1), dtype=torch.bool)
        positions = torch.zeros((1, 1), dtype=torch.int64)
        token_major = graph.prefill(
            hidden,
            execution_mask,
            ["seq"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        region_major = graph.prefill_region_major(
            hidden,
            execution_mask,
            ["seq"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        _assert_nested_close(self, token_major, region_major)
        for result in (token_major, region_major):
            assert result.trace is not None
            region_event = next(
                event
                for event in result.trace.region_events
                if event.region_id == "region.long"
            )
            self.assertEqual(region_event.candidate_node_ids, ())
            self.assertFalse(region_event.forced_active)
            self.assertIsNone(region_event.logits)
            self.assertIsNone(region_event.probabilities)
            self.assertIsNone(region_event.requested_k)
            self.assertIsNone(region_event.effective_k)
            self.assertIsNone(region_event.top_k_node_ids)
            self.assertEqual(region_event.active_node_ids, ())
            node_event = next(
                event
                for event in result.trace.node_events
                if event.node_id == "node.long.1"
            )
            self.assertFalse(node_event.reached)
            self.assertFalse(node_event.active)
            self.assertIsNone(node_event.selector_read)
            self.assertIsNone(node_event.logit)
            self.assertIsNone(node_event.probability)

    def test_backward_matches_token_major_for_inputs_and_parameters(self):
        torch.manual_seed(37)
        graph = SettleGraph(_backward_plan()).double()
        selector = graph.selector("region.0")
        with torch.no_grad():
            for index, node_id in enumerate(graph.plan.regions[0].node_ids):
                linear = selector.linears[safe_module_key(node_id)]
                linear.weight.zero_()
                linear.bias.fill_(2.0 if index == 0 else -2.0)
                receiver = graph.receiver(node_id)
                receiver.down_proj.weight.copy_(
                    (index + 1.0) * torch.eye(3, dtype=torch.float64)
                )
                receiver.down_proj.bias.fill_(0.1 * (index + 1.0))

        source = torch.randn(2, 3, 3, dtype=torch.float64)
        execution_mask = torch.ones((2, 3), dtype=torch.bool)
        positions = torch.arange(3, dtype=torch.int64).repeat(2, 1)
        cotangent = torch.randn(2, 3, 3, dtype=torch.float64)

        token_input = source.clone().requires_grad_(True)
        token_result = graph.prefill(
            token_input,
            execution_mask,
            ["a", "b"],
            positions,
            detach_at_end=False,
        )
        token_loss = (token_result.output * cotangent).sum()
        token_loss = token_loss + 0.3 * token_result.balance_loss
        token_loss.backward()
        token_input_grad = token_input.grad.detach().clone()
        token_parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in graph.named_parameters()
        }

        graph.zero_grad(set_to_none=True)
        region_input = source.clone().requires_grad_(True)
        region_result = graph.prefill_region_major(
            region_input,
            execution_mask,
            ["a", "b"],
            positions,
            detach_at_end=False,
        )
        region_loss = (region_result.output * cotangent).sum()
        region_loss = region_loss + 0.3 * region_result.balance_loss
        region_loss.backward()
        region_parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in graph.named_parameters()
        }

        torch.testing.assert_close(
            region_result.output, token_result.output, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            region_input.grad, token_input_grad, atol=1e-11, rtol=1e-11
        )
        _assert_nested_close(self, token_parameter_grads, region_parameter_grads)


if __name__ == "__main__":
    unittest.main()
