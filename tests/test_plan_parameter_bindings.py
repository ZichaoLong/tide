from __future__ import annotations

import dataclasses
import unittest

from tide.builders import build_diamond, build_single_layer, build_singleton
from tide.plan import (
    OperationParameterBinding,
    Plan,
    PlanValidationError,
)


def _binding(
    owner_kind: str,
    owner_id: str,
    operation: str,
    parameter_set_id: str,
) -> OperationParameterBinding:
    return OperationParameterBinding(
        owner_kind=owner_kind,
        owner_id=owner_id,
        operation=operation,
        parameter_set_id=parameter_set_id,
    )


class PlanParameterBindingTestCase(unittest.TestCase):
    def assert_invalid(
        self,
        plan: Plan,
        text: str,
        failure_codes: tuple[str, ...],
    ) -> None:
        with self.assertRaises(PlanValidationError) as raised:
            plan.validate()
        self.assertIn(text, str(raised.exception))
        self.assertEqual(raised.exception.failure_codes, failure_codes)

    def test_schema_v1_rejects_new_bindings_and_keeps_legacy_node_gate(self) -> None:
        base = build_singleton()
        use_binding = _binding(
            "node", base.nodes[0].node_id, "input_norm", "params.shared"
        )
        self.assert_invalid(
            dataclasses.replace(base, parameter_bindings=(use_binding,)),
            "parameter_bindings must be empty in Plan schema version '1'",
            ("plan.schema",),
        )

        legacy_node = dataclasses.replace(
            base.nodes[0], parameter_group="legacy.group"
        )
        self.assert_invalid(
            dataclasses.replace(
                base,
                schema_version="2",
                nodes=(legacy_node,),
            ),
            "parameter_group must be null in Plan schema version '2'",
            ("plan.schema",),
        )

    def test_v2_bindings_are_canonical_and_change_the_logical_hash(self) -> None:
        base = build_single_layer(receiver_count=2, k=1)
        left, right = (node.node_id for node in base.nodes)
        bindings = (
            _binding("node", right, "input_norm", "params.shared-norm"),
            _binding("node", left, "input_norm", "params.shared-norm"),
        )
        plan = dataclasses.replace(
            base, schema_version="2", parameter_bindings=bindings
        ).validate()
        reordered = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=tuple(reversed(bindings)),
        ).validate()

        self.assertEqual(plan.parameter_bindings, tuple(reversed(bindings)))
        self.assertEqual(plan.canonical_bytes(), reordered.canonical_bytes())
        self.assertEqual(plan.canonical_hash(), reordered.canonical_hash())
        self.assertEqual(
            [
                item["owner_id"]
                for item in plan.canonical_dict()["parameter_bindings"]
            ],
            [left, right],
        )

        independent = dataclasses.replace(
            plan,
            parameter_bindings=(
                _binding("node", left, "input_norm", "params.left-norm"),
                _binding("node", right, "input_norm", "params.right-norm"),
            ),
        ).validate()
        self.assertNotEqual(plan.canonical_hash(), independent.canonical_hash())

    def test_explicit_singleton_binding_is_valid_and_others_remain_unlisted(self) -> None:
        base = build_singleton()
        binding = _binding(
            "node", base.nodes[0].node_id, "input_norm", "params.norm"
        )
        plan = dataclasses.replace(
            base, schema_version="2", parameter_bindings=(binding,)
        ).validate()
        self.assertEqual(plan.parameter_bindings, (binding,))
        self.assertEqual(
            plan.canonical_dict()["parameter_bindings"],
            [
                {
                    "owner_kind": "node",
                    "owner_id": base.nodes[0].node_id,
                    "operation": "input_norm",
                    "parameter_set_id": "params.norm",
                }
            ],
        )

    def test_compatible_operations_can_share_one_parameter_set(self) -> None:
        base = build_single_layer(receiver_count=2, k=1)
        left, right = (node.node_id for node in base.nodes)
        plan = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding("node", left, "input_norm", "params.norm"),
                _binding("node", right, "ffn_norm", "params.norm"),
            ),
        )
        self.assertIs(plan.validate(), plan)

    def test_same_formula_with_different_slot_arity_is_rejected(self) -> None:
        base = build_diamond(d_model=4, branch_k=2)
        aggregate = {
            "type": "edge_softmax",
            "formula_id": "TEST-AGG-EDGE-SOFTMAX-V1",
        }
        nodes = tuple(
            dataclasses.replace(node, aggregate=aggregate)
            if node.node_id in {"node.branch.a", "node.out"}
            else node
            for node in base.nodes
        )
        plan = dataclasses.replace(
            base,
            nodes=nodes,
            schema_version="2",
            parameter_bindings=(
                _binding(
                    "node", "node.branch.a", "aggregate", "params.aggregate"
                ),
                _binding("node", "node.out", "aggregate", "params.aggregate"),
            ),
        )
        self.assert_invalid(
            plan,
            "incompatible canonical parameter slots",
            ("plan.formula",),
        )

    def test_same_formula_with_different_slot_shapes_is_rejected(self) -> None:
        base = build_single_layer(receiver_count=2, k=1, d_model=4)
        left, right = (node.node_id for node in base.nodes)
        nodes = tuple(
            dataclasses.replace(
                node,
                node_compute={
                    "type": "double_residual_swiglu",
                    "formula_id": "TEST-NODE-SWIGLU-V1",
                    "hidden_dim": 8 if node.node_id == left else 12,
                },
            )
            for node in base.nodes
        )
        plan = dataclasses.replace(
            base,
            nodes=nodes,
            schema_version="2",
            parameter_bindings=(
                _binding("node", left, "node_compute", "params.compute"),
                _binding("node", right, "node_compute", "params.compute"),
            ),
        )
        self.assert_invalid(
            plan,
            "incompatible canonical parameter slots",
            ("plan.formula",),
        )

    def test_region_and_graph_operation_owners_are_supported(self) -> None:
        base = build_single_layer(receiver_count=2, k=1)
        region = dataclasses.replace(
            base.regions[0],
            score={
                "type": "linear",
                "formula_id": "TEST-SCORE-LINEAR-V1",
            },
        )
        plan = dataclasses.replace(
            base,
            schema_version="2",
            regions=(region,),
            output_aggregate={
                "type": "node_softmax",
                "formula_id": "TEST-AGG-TERMINAL-SOFTMAX-V1",
            },
            parameter_bindings=(
                _binding(
                    "region", region.region_id, "score", "params.selector"
                ),
                _binding(
                    "graph", "graph", "output_aggregate", "params.output"
                ),
            ),
        )
        self.assertIs(plan.validate(), plan)

    def test_duplicate_unknown_and_mismatched_uses_are_rejected(self) -> None:
        base = build_single_layer(receiver_count=2, k=1)
        left, right = (node.node_id for node in base.nodes)
        duplicate = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding("node", left, "input_norm", "params.a"),
                _binding("node", left, "input_norm", "params.b"),
            ),
        )
        self.assert_invalid(
            duplicate,
            "each operation use may have at most one explicit parameter binding",
            ("plan.schema",),
        )

        unknown = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding("node", "node.missing", "input_norm", "params.a"),
            ),
        )
        self.assert_invalid(
            unknown, "parameter binding names unknown node", ("plan.topology",)
        )

        mismatched_node = dataclasses.replace(
            base.node_by_id(right),
            ffn_norm={
                "type": "rmsnorm",
                "formula_id": "TEST-RMSNORM-V1",
            },
        )
        mismatched = dataclasses.replace(
            base,
            schema_version="2",
            nodes=tuple(
                mismatched_node if node.node_id == right else node
                for node in base.nodes
            ),
            parameter_bindings=(
                _binding("node", left, "input_norm", "params.norm"),
                _binding("node", right, "ffn_norm", "params.norm"),
            ),
        )
        self.assert_invalid(
            mismatched, "mixes formula_id", ("plan.formula",)
        )

    def test_bindings_require_a_registered_parameterized_operation(self) -> None:
        base = build_singleton()
        node_id = base.nodes[0].node_id
        parameterless = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding(
                    "node", node_id, "node_compute", "params.identity"
                ),
            ),
        )
        self.assert_invalid(
            parameterless,
            "at least one parameter role",
            ("plan.formula",),
        )

        wrong_owner_operation = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding("region", base.regions[0].region_id, "input_norm", "p"),
            ),
        )
        self.assert_invalid(
            wrong_owner_operation,
            "operation 'input_norm' is not valid for owner_kind 'region'",
            ("plan.schema",),
        )

        wrong_graph_owner = dataclasses.replace(
            base,
            schema_version="2",
            parameter_bindings=(
                _binding("graph", "other", "output_aggregate", "p"),
            ),
        )
        self.assert_invalid(
            wrong_graph_owner,
            "graph owner_id must be exactly 'graph'",
            ("plan.schema",),
        )


if __name__ == "__main__":
    unittest.main()
