from __future__ import annotations

import math
import types
import unittest

import torch
import torch.nn.functional as F

from tide.ops import (
    AttentionState,
    ReceiverModule,
    RegionSelector,
    deterministic_topk_mask,
    safe_module_key,
)


def receiver_spec(**overrides):
    values = {
        "aggregate": {"type": "mean"},
        "update": {"type": "none"},
        "selector_read": {"type": "content_norm", "out_dim": 1},
        "ffn_read": {"type": "zero"},
        "node_compute": {"type": "identity"},
        "emit": {"type": "hard"},
        "state_shape": (),
        "selector_read_shape": (1,),
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class AggregateFormulaTests(unittest.TestCase):
    def test_entry_boundary_is_exact_identity_for_edge_affine(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                aggregate={"type": "edge_linear_mean", "bias": True}
            ),
        ).double()
        value = torch.tensor([2.0, -3.0], dtype=torch.float64)
        actual = module.aggregate([value], ["boundary:n0"])
        self.assertIs(actual, value)
        self.assertEqual(len(module.edge_transforms), 0)

    def test_edge_softmax_matches_independent_formula(self):
        module = ReceiverModule(
            2, receiver_spec(aggregate={"type": "learned_convex"})
        ).double()
        module.ensure_edge_transforms(["e0", "e1"])
        with torch.no_grad():
            module.edge_scores[safe_module_key("e0")].fill_(math.log(3.0))
            module.edge_scores[safe_module_key("e1")].fill_(0.0)
        left = torch.tensor([4.0, -2.0], dtype=torch.float64)
        right = torch.tensor([0.0, 6.0], dtype=torch.float64)
        actual = module.aggregate([left, right], ["e0", "e1"])
        expected = 0.75 * left + 0.25 * right
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_edge_affine_mean_matches_manual_formula_and_order(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                aggregate={"type": "edge_linear_mean", "bias": True}
            ),
        ).double()
        module.ensure_edge_transforms(["edge-b", "edge-a"])
        first = module.edge_transforms[safe_module_key("edge-b")]
        second = module.edge_transforms[safe_module_key("edge-a")]
        with torch.no_grad():
            first.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))
            first.bias.copy_(torch.tensor([1.0, 2.0]))
            second.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
            second.bias.copy_(torch.tensor([-1.0, 3.0]))
        x = torch.tensor([1.0, 4.0], dtype=torch.float64)
        y = torch.tensor([-2.0, 5.0], dtype=torch.float64)
        actual = module.aggregate([x, y], ["edge-b", "edge-a"])
        expected = ((first.weight @ x + first.bias) + (second.weight @ y + second.bias)) / 2
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


class StateFormulaTests(unittest.TestCase):
    def test_ema_proposal_matches_formula(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                update={"type": "ema", "state_dim": 2, "decay": 0.25},
                state_shape=(2,),
            ),
        ).double()
        with torch.no_grad():
            module.ema_observe.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
            module.ema_observe.bias.copy_(torch.tensor([0.5, -0.5]))
        old = torch.tensor([2.0, -4.0], dtype=torch.float64)
        current = torch.tensor([0.25, 0.5], dtype=torch.float64)
        actual = module.proposal(old, current)
        expected_observation = torch.tanh(
            torch.tensor([0.75, 0.5], dtype=torch.float64)
        )
        expected = 0.25 * old + 0.75 * expected_observation
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_attention_state_records_positions_and_evicts(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                update={
                    "type": "attention_window",
                    "key_dim": 2,
                    "value_dim": 2,
                    "window": 2,
                },
                state_shape=(2, 2, 2),
            ),
        ).double()
        with torch.no_grad():
            module.attn_key.weight.copy_(torch.eye(2))
            module.attn_value.weight.copy_(torch.eye(2))
        state = module.initial_state(torch.zeros(2, dtype=torch.float64))
        self.assertIsInstance(state, AttentionState)
        state = module.proposal(
            state, torch.tensor([1.0, 0.0], dtype=torch.float64), token_position=4
        )
        state = module.proposal(
            state, torch.tensor([0.0, 2.0], dtype=torch.float64), token_position=5
        )
        state = module.proposal(
            state, torch.tensor([3.0, 0.0], dtype=torch.float64), token_position=6
        )
        self.assertEqual(state.positions.tolist(), [5, 6])
        torch.testing.assert_close(
            state.keys,
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            module.proposal(
                state,
                torch.tensor([1.0, 0.0], dtype=torch.float64),
                token_position=6,
            )

    def test_full_state_selector_read_is_affine_projection(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                update={"type": "ema", "state_dim": 2, "decay": 0.5},
                state_shape=(2,),
                selector_read={"type": "content_state_linear", "out_dim": 1},
            ),
        ).double()
        with torch.no_grad():
            module.selector_read_linear.weight.copy_(
                torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            )
            module.selector_read_linear.bias.fill_(5.0)
        actual = module.selector_read(
            torch.tensor([1.0, -1.0], dtype=torch.float64),
            torch.tensor([2.0, 0.5], dtype=torch.float64),
        )
        expected = torch.tensor([12.0], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


class SelectorAndEmitTests(unittest.TestCase):
    def test_fixed_score_uses_static_node_ids_for_reached_subset(self):
        selector = RegionSelector(
            1,
            {"type": "fixed", "values": [5.0, 1.0, 3.0]},
            ["a", "b", "c"],
        ).double()
        actual = selector(
            torch.zeros((2, 1), dtype=torch.float64), ["a", "c"]
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([5.0, 3.0], dtype=torch.float64),
            atol=0,
            rtol=0,
        )

    def test_linear_score_parameters_are_node_specific_by_default(self):
        selector = RegionSelector(
            2, {"type": "linear", "bias": True}, ["a", "b"]
        ).double()
        with torch.no_grad():
            selector.linears[safe_module_key("a")].weight.copy_(
                torch.tensor([[1.0, 0.0]])
            )
            selector.linears[safe_module_key("a")].bias.fill_(2.0)
            selector.linears[safe_module_key("b")].weight.copy_(
                torch.tensor([[0.0, -1.0]])
            )
            selector.linears[safe_module_key("b")].bias.fill_(3.0)
        actual = selector(
            torch.tensor([[4.0, 9.0], [7.0, 5.0]], dtype=torch.float64),
            ["a", "b"],
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([6.0, -2.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_topk_ties_follow_candidate_order(self):
        scores = torch.tensor([2.0, 2.0, 3.0, 2.0])
        mask = deterministic_topk_mask(scores, 3)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_hard_straight_through_forward_and_probability_gradient(self):
        module = ReceiverModule(
            2,
            receiver_spec(emit={"type": "hst", "zeta": 0.5}),
        ).double()
        hidden = torch.tensor([1.0, -2.0], dtype=torch.float64)
        computed = torch.tensor([4.0, 6.0], dtype=torch.float64, requires_grad=True)
        probability = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        emitted = module.emit(hidden, computed, probability)
        torch.testing.assert_close(emitted, computed, atol=0, rtol=0)
        emitted.sum().backward()
        expected_probability_grad = 0.5 * (computed.detach() - hidden).sum()
        torch.testing.assert_close(
            probability.grad, expected_probability_grad, atol=1e-12, rtol=1e-12
        )


class NodeComputeFormulaTests(unittest.TestCase):
    def test_affine_node_matches_test_formula(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                node_compute={"type": "affine_residual", "bias": True}
            ),
        ).double()
        with torch.no_grad():
            module.down_proj.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))
            module.down_proj.bias.copy_(torch.tensor([1.0, -2.0]))
        hidden = torch.tensor([4.0, 5.0], dtype=torch.float64)
        normalized = torch.tensor([1.5, -0.5], dtype=torch.float64)
        actual = module.compute(hidden, normalized, None)
        expected = hidden + module.down_proj.weight @ normalized + module.down_proj.bias
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_swiglu_node_matches_manual_double_residual(self):
        module = ReceiverModule(
            2,
            receiver_spec(
                node_compute={
                    "type": "double_residual_swiglu",
                    "hidden_dim": 2,
                    "bias": True,
                    "ffn_norm_eps": 1e-6,
                }
            ),
        ).double()
        with torch.no_grad():
            module.ffn_norm.weight.copy_(torch.tensor([1.25, 0.75]))
            module.gate_proj.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
            module.gate_proj.bias.copy_(torch.tensor([0.25, -0.5]))
            module.up_proj.weight.copy_(torch.tensor([[0.5, -1.0], [2.0, 1.0]]))
            module.up_proj.bias.copy_(torch.tensor([1.0, 0.0]))
            module.down_proj.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 2.0]]))
            module.down_proj.bias.copy_(torch.tensor([-1.0, 0.5]))
        hidden = torch.tensor([2.0, -1.0], dtype=torch.float64)
        normalized = torch.zeros_like(hidden)
        variance = hidden.square().mean()
        normed = hidden * torch.rsqrt(variance + 1e-6) * module.ffn_norm.weight
        gate = module.gate_proj.weight @ normed + module.gate_proj.bias
        up = module.up_proj.weight @ normed + module.up_proj.bias
        expected = hidden + module.down_proj.weight @ (F.silu(gate) * up) + module.down_proj.bias
        actual = module.compute(hidden, normalized, None)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
