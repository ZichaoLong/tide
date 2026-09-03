from __future__ import annotations

import unittest

import torch

from tide.placement import (
    PARATTN,
    PARBLK,
    PARMLP,
    POST,
    PlacementContractError,
    apply_placement,
)


class RecordingTransform:
    def __init__(self, scale: float, bias: torch.Tensor) -> None:
        self.scale = scale
        self.bias = bias
        self.inputs = []

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden.detach().clone())
        return self.scale * hidden + self.bias


class PlacementEquationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]],
            dtype=torch.float64,
        )
        self.attention_bias = torch.tensor(
            [0.25, -0.5, 1.0], dtype=torch.float64
        )
        self.mlp_bias = torch.tensor(
            [-1.0, 0.75, 0.5], dtype=torch.float64
        )
        self.graph_bias = torch.tensor(
            [2.0, -1.0, 0.25], dtype=torch.float64
        )

    def transforms(self):
        return (
            RecordingTransform(0.5, self.attention_bias),
            RecordingTransform(-0.25, self.mlp_bias),
            RecordingTransform(1.5, self.graph_bias),
        )

    def base_values(self):
        attention_residual = 0.5 * self.hidden + self.attention_bias
        attention_output = self.hidden + attention_residual
        mlp_residual = -0.25 * attention_output + self.mlp_bias
        base_output = attention_output + mlp_residual
        return attention_output, base_output

    def test_post_matches_complete_block_then_graph_equation(self):
        attention, mlp, graph = self.transforms()
        attention_output, base_output = self.base_values()

        actual = apply_placement(
            self.hidden,
            placement=POST,
            attention=attention,
            mlp=mlp,
            graph=graph,
        )

        expected = 1.5 * base_output + self.graph_bias
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(graph.inputs[0], base_output)
        torch.testing.assert_close(mlp.inputs[0], attention_output)

    def test_parblk_matches_full_block_plus_graph_residual_equation(self):
        attention, mlp, graph = self.transforms()
        attention_output, base_output = self.base_values()

        actual = apply_placement(
            self.hidden,
            placement=PARBLK,
            attention=attention,
            mlp=mlp,
            graph=graph,
        )

        graph_output = 1.5 * self.hidden + self.graph_bias
        expected = base_output + (graph_output - self.hidden)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(graph.inputs[0], self.hidden)
        torch.testing.assert_close(mlp.inputs[0], attention_output)

    def test_parattn_merges_graph_before_mlp_equation(self):
        attention, mlp, graph = self.transforms()
        attention_output, _ = self.base_values()

        actual = apply_placement(
            self.hidden,
            placement=PARATTN,
            attention=attention,
            mlp=mlp,
            graph=graph,
        )

        graph_output = 1.5 * self.hidden + self.graph_bias
        merged = attention_output + (graph_output - self.hidden)
        expected = merged + (-0.25 * merged + self.mlp_bias)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(graph.inputs[0], self.hidden)
        torch.testing.assert_close(mlp.inputs[0], merged)

    def test_parmlp_uses_shared_attention_output_equation(self):
        attention, mlp, graph = self.transforms()
        attention_output, base_output = self.base_values()

        actual = apply_placement(
            self.hidden,
            placement=PARMLP,
            attention=attention,
            mlp=mlp,
            graph=graph,
        )

        graph_output = 1.5 * attention_output + self.graph_bias
        expected = base_output + (graph_output - attention_output)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(graph.inputs[0], attention_output)
        torch.testing.assert_close(mlp.inputs[0], attention_output)

    def test_identity_graph_recovers_base_block_for_every_placement(self):
        attention_output, base_output = self.base_values()
        del attention_output

        for placement in (POST, PARBLK, PARATTN, PARMLP):
            with self.subTest(placement=placement):
                attention = RecordingTransform(0.5, self.attention_bias)
                mlp = RecordingTransform(-0.25, self.mlp_bias)
                graph_inputs = []

                def identity_graph(hidden):
                    graph_inputs.append(hidden.detach().clone())
                    return hidden

                actual = apply_placement(
                    self.hidden,
                    placement=placement,
                    attention=attention,
                    mlp=mlp,
                    graph=identity_graph,
                )
                torch.testing.assert_close(
                    actual, base_output, atol=0, rtol=0
                )
                self.assertEqual(len(graph_inputs), 1)

    def test_rejects_unknown_placement_before_calling_transforms(self):
        calls = []

        def transform(hidden):
            calls.append(hidden)
            return hidden

        with self.assertRaisesRegex(
            PlacementContractError, "unsupported placement"
        ):
            apply_placement(
                self.hidden,
                placement="UNKNOWN",
                attention=transform,
                mlp=transform,
                graph=transform,
            )
        self.assertEqual(calls, [])

    def test_rejects_graph_output_with_changed_hidden_shape(self):
        attention, mlp, _ = self.transforms()

        with self.assertRaisesRegex(
            PlacementContractError, "graph changed hidden shape"
        ):
            apply_placement(
                self.hidden,
                placement=POST,
                attention=attention,
                mlp=mlp,
                graph=lambda hidden: hidden[..., :2],
            )


if __name__ == "__main__":
    unittest.main()
