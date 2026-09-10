from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import torch

from tide.builders import build_single_layer
from tide.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from tide.engine import SettleGraph, StateStore
from tide.packed import PackedSettleGraph
from tide.plan import OperationParameterBinding, Plan, bind_dtypes
from tide.specialized import SINGLE_LAYER_V1, SpecializedExecutor


_SEQUENCE_IDS = ("sharing.sequence.a", "sharing.sequence.b")
_EXECUTION_MASK = torch.ones((2, 3), dtype=torch.bool)
_TOKEN_POSITIONS = torch.arange(3, dtype=torch.int64).repeat(2, 1)


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


def _affine_plan(*, binding_operation: Optional[str]) -> Plan:
    plan = build_single_layer(receiver_count=2, k=2, d_model=3)
    nodes = tuple(
        dataclasses.replace(
            node,
            node_compute={
                "type": "affine_residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
                "bias": True,
                "output_shape": [3],
            },
        )
        for node in plan.nodes
    )
    bindings = (
        tuple(
            _binding(
                "node",
                node.node_id,
                binding_operation,
                f"parameters.{binding_operation}",
            )
            for node in nodes
        )
        if binding_operation is not None
        else ()
    )
    return dataclasses.replace(
        plan,
        nodes=nodes,
        schema_version="2",
        parameter_bindings=bindings,
    ).validate()


def _stateful_update_plan() -> Plan:
    plan = build_single_layer(receiver_count=2, k=2, d_model=3)
    nodes = tuple(
        dataclasses.replace(
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
            selector_read_shape=(2,),
            selector_read={
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 2,
                "output_shape": [2],
            },
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
                "output_shape": [3],
            },
            node_compute={
                "type": "double_residual_swiglu",
                "formula_id": "TEST-NODE-SWIGLU-V1",
                "hidden_dim": 5,
                "bias": True,
                "output_shape": [3],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0], profile="BO", selector_timing="post"
    )
    return dataclasses.replace(
        plan,
        nodes=nodes,
        regions=(region,),
        schema_version="2",
        parameter_bindings=tuple(
            _binding("node", node.node_id, "update", "parameters.update")
            for node in nodes
        ),
    ).validate()


def _prefill(model_or_executor: object, hidden: torch.Tensor):
    return model_or_executor.prefill(
        hidden,
        execution_mask=_EXECUTION_MASK,
        sequence_ids=_SEQUENCE_IDS,
        token_positions=_TOKEN_POSITIONS,
        detach_at_end=False,
    )


class ParameterSharingRuntimeTests(unittest.TestCase):
    def test_parameter_conversion_reapplies_declared_bindings(self) -> None:
        plan = _affine_plan(binding_operation="input_norm")
        left_id, right_id = (node.node_id for node in plan.nodes)
        previous = torch.__future__.get_overwrite_module_params_on_conversion()
        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            model = SettleGraph(plan)
            before_conversion = model.receiver(left_id).input_norm.weight

            model.to(dtype=torch.float64)

            left = model.receiver(left_id).input_norm.weight
            right = model.receiver(right_id).input_norm.weight
            self.assertIsNot(left, before_conversion)
            self.assertIs(left, right)
            optimizer_parameters = list(
                torch.optim.AdamW(model.parameters(), lr=1e-3).param_groups[0]["params"]
            )
            self.assertEqual(
                len(optimizer_parameters),
                len({id(parameter) for parameter in optimizer_parameters}),
            )
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(previous)

    def test_meta_conversions_reapply_declared_bindings(self) -> None:
        plan = _affine_plan(binding_operation="input_norm")
        left_id, right_id = (node.node_id for node in plan.nodes)
        previous = torch.__future__.get_overwrite_module_params_on_conversion()
        try:
            torch.__future__.set_overwrite_module_params_on_conversion(False)
            model = SettleGraph(plan)
            self.assertIs(model.to(device="meta"), model)
            self.assertIs(
                model.receiver(left_id).input_norm.weight,
                model.receiver(right_id).input_norm.weight,
            )

            torch.__future__.set_overwrite_module_params_on_conversion(True)
            empty_model = SettleGraph(plan)
            self.assertIs(empty_model.to_empty(device="meta"), empty_model)
            self.assertIs(
                empty_model.receiver(left_id).input_norm.weight,
                empty_model.receiver(right_id).input_norm.weight,
            )
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(previous)

    def test_only_the_declared_operation_is_tied_and_optimizer_is_unique(self) -> None:
        plan = _affine_plan(binding_operation="input_norm")
        left_id, right_id = (node.node_id for node in plan.nodes)
        model = SettleGraph(plan).to(dtype=torch.float64)
        left = model.receiver(left_id)
        right = model.receiver(right_id)

        self.assertIs(left.input_norm.weight, right.input_norm.weight)
        self.assertIsNot(left.ffn_norm.weight, right.ffn_norm.weight)
        self.assertIsNot(left.down_proj.weight, right.down_proj.weight)
        duplicate_parameters = list(model.named_parameters(remove_duplicate=False))
        self.assertEqual(
            sum(parameter is left.input_norm.weight for _, parameter in duplicate_parameters),
            2,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(optimizer_parameters), len({id(p) for p in optimizer_parameters}))
        self.assertEqual(
            sum(parameter is left.input_norm.weight for parameter in optimizer_parameters),
            1,
        )

    def test_multi_parameter_node_compute_binding_ties_every_role(self) -> None:
        plan = _affine_plan(binding_operation="node_compute")
        left_id, right_id = (node.node_id for node in plan.nodes)
        model = SettleGraph(plan)
        left = model.receiver(left_id)
        right = model.receiver(right_id)

        self.assertIs(left.down_proj.weight, right.down_proj.weight)
        self.assertIs(left.down_proj.bias, right.down_proj.bias)
        self.assertIsNot(left.input_norm.weight, right.input_norm.weight)
        self.assertIsNot(left.ffn_norm.weight, right.ffn_norm.weight)

    def test_shared_gradient_equals_the_sum_of_untied_gradients(self) -> None:
        shared_plan = _affine_plan(binding_operation="input_norm")
        untied_plan = _affine_plan(binding_operation=None)
        left_id, right_id = (node.node_id for node in shared_plan.nodes)
        torch.manual_seed(101)
        shared = SettleGraph(shared_plan).to(dtype=torch.float64)
        untied = SettleGraph(untied_plan).to(dtype=torch.float64)
        untied.load_state_dict(shared.state_dict(), strict=True)

        hidden = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) / 7.0
        shared_result = _prefill(shared, hidden.clone().requires_grad_())
        untied_result = _prefill(untied, hidden.clone().requires_grad_())
        torch.testing.assert_close(
            shared_result.output, untied_result.output, atol=0, rtol=0
        )
        shared_result.output.square().sum().backward()
        untied_result.output.square().sum().backward()

        shared_gradient = shared.receiver(left_id).input_norm.weight.grad
        left_gradient = untied.receiver(left_id).input_norm.weight.grad
        right_gradient = untied.receiver(right_id).input_norm.weight.grad
        self.assertIsNotNone(shared_gradient)
        self.assertIsNotNone(left_gradient)
        self.assertIsNotNone(right_gradient)
        torch.testing.assert_close(
            shared_gradient,
            left_gradient + right_gradient,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_eager_packed_and_specialized_reuse_the_same_parameter_owner(self) -> None:
        plan = _affine_plan(binding_operation="input_norm")
        left_id, right_id = (node.node_id for node in plan.nodes)
        torch.manual_seed(202)
        models = [SettleGraph(plan).to(dtype=torch.float64) for _ in range(3)]
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict(), strict=True)
        executors = (
            models[0],
            PackedSettleGraph(models[1]),
            SpecializedExecutor(models[2], SINGLE_LAYER_V1),
        )
        hidden = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) / 11.0
        results = []
        gradients = []
        for model, executor in zip(models, executors):
            self.assertIs(
                model.receiver(left_id).input_norm.weight,
                model.receiver(right_id).input_norm.weight,
            )
            result = _prefill(executor, hidden.clone().requires_grad_())
            result.output.square().sum().backward()
            results.append(result)
            gradients.append(model.receiver(left_id).input_norm.weight.grad)

        for result in results[1:]:
            torch.testing.assert_close(result.output, results[0].output, atol=0, rtol=0)
            torch.testing.assert_close(
                result.balance_loss, results[0].balance_loss, atol=0, rtol=0
            )
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient, gradients[0], atol=1e-12, rtol=1e-12)

    def test_stateful_update_parameters_are_shared_but_state_is_not(self) -> None:
        plan = _stateful_update_plan()
        left_id, right_id = (node.node_id for node in plan.nodes)
        torch.manual_seed(303)
        eager_model = SettleGraph(plan).to(dtype=torch.float64)
        packed_model = SettleGraph(plan).to(dtype=torch.float64)
        packed_model.load_state_dict(eager_model.state_dict(), strict=True)
        for model in (eager_model, packed_model):
            self.assertIs(
                model.receiver(left_id).ema_observe.weight,
                model.receiver(right_id).ema_observe.weight,
            )
            self.assertIs(
                model.receiver(left_id).ema_observe.bias,
                model.receiver(right_id).ema_observe.bias,
            )

        initial = StateStore(
            values={
                (_SEQUENCE_IDS[0], left_id): torch.zeros(2, dtype=torch.float64),
                (_SEQUENCE_IDS[0], right_id): torch.ones(2, dtype=torch.float64),
            },
            next_position={_SEQUENCE_IDS[0]: 0},
        )
        hidden = torch.arange(3, dtype=torch.float64).reshape(1, 1, 3) / 5.0
        call = {
            "execution_mask": torch.ones((1, 1), dtype=torch.bool),
            "sequence_ids": (_SEQUENCE_IDS[0],),
            "token_positions": torch.zeros((1, 1), dtype=torch.int64),
            "state": initial,
            "detach_at_end": False,
        }
        eager = eager_model.prefill(hidden, **call)
        packed = PackedSettleGraph(packed_model).prefill(hidden, **call)
        torch.testing.assert_close(eager.output, packed.output, atol=0, rtol=0)
        for node_id in (left_id, right_id):
            torch.testing.assert_close(
                eager.state.values[(_SEQUENCE_IDS[0], node_id)],
                packed.state.values[(_SEQUENCE_IDS[0], node_id)],
                atol=0,
                rtol=0,
            )
        left_state = eager.state.values[(_SEQUENCE_IDS[0], left_id)]
        right_state = eager.state.values[(_SEQUENCE_IDS[0], right_id)]
        self.assertIsNot(left_state, right_state)
        self.assertNotEqual(left_state.data_ptr(), right_state.data_ptr())
        self.assertFalse(torch.equal(left_state, right_state))

    def test_checkpoint_round_trip_restores_binding_and_rejects_conflicts(self) -> None:
        plan = _affine_plan(binding_operation="input_norm")
        typed_plan = bind_dtypes(
            plan,
            hidden="float64",
            parameter="float64",
            state="float64",
            readout="float64",
        )
        left_id, right_id = (node.node_id for node in plan.nodes)
        torch.manual_seed(404)
        source = SettleGraph(typed_plan).to(dtype=torch.float64)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        hidden = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) / 13.0
        result = _prefill(source, hidden.requires_grad_())
        result.output.square().sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
            save_checkpoint(
                path,
                model=source,
                typed_plan=typed_plan,
                state=result.state,
                optimizer=optimizer,
            )
            target = SettleGraph(typed_plan).to(dtype=torch.float64)
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
            load_checkpoint(
                path,
                model=target,
                typed_plan=typed_plan,
                mode="resume",
                optimizer=target_optimizer,
            )
            self.assertIs(
                target.receiver(left_id).input_norm.weight,
                target.receiver(right_id).input_norm.weight,
            )
            for name, value in source.state_dict().items():
                torch.testing.assert_close(
                    target.state_dict()[name], value, atol=0, rtol=0
                )

            alias_names = [
                name
                for name, parameter in source.named_parameters(remove_duplicate=False)
                if parameter is source.receiver(left_id).input_norm.weight
            ]
            self.assertEqual(len(alias_names), 2)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["model_state"][alias_names[1]] = (
                payload["model_state"][alias_names[1]].clone() + 1.0
            )
            conflicting_path = Path(directory) / "conflicting.pt"
            torch.save(payload, conflicting_path)
            untouched = SettleGraph(typed_plan).to(dtype=torch.float64)
            with self.assertRaisesRegex(
                CheckpointError, "aliased model parameter entries carry inconsistent"
            ):
                load_checkpoint(
                    conflicting_path,
                    model=untouched,
                    typed_plan=typed_plan,
                    mode="init-from",
                )


if __name__ == "__main__":
    unittest.main()
