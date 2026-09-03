"""Live-backend SettleGraph tests selected by TEST_BACKEND.

The module is inert during ordinary unit-test discovery.  A validation process
that explicitly supplies ``TEST_BACKEND`` must resolve that backend or fail;
it never skips an unavailable requested accelerator and never falls back to
CPU.  The process creates no CPU autograd workload before backend resolution.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

import torch

from tide.builders import build_single_layer, build_singleton
from tide.checkpoint import load_checkpoint, save_checkpoint
from tide.engine import SettleGraph
from tide.equivalence import (
    CPU_FLOAT64_TOLERANCE,
    SAME_BACKEND_FLOAT32_TOLERANCE,
    compare_nested,
    validate_trace_invariants,
)
from tide.ops import safe_module_key
from tide.plan import bind_dtypes
from tide.runtime import RuntimeRequest, resolve_runtime


_TEST_BACKEND = os.environ.get("TEST_BACKEND", "").strip().lower()
_TEST_DTYPE = os.environ.get("TEST_DTYPE", "float32").strip().lower()
_TEST_DEVICE_INDEX = int(os.environ.get("TEST_DEVICE_INDEX", "0"))


def _replace_plan(plan, *, nodes=None, regions=None):
    return dataclasses.replace(
        plan,
        nodes=tuple(plan.nodes if nodes is None else nodes),
        regions=tuple(plan.regions if regions is None else regions),
    ).validate()


def _trainable_plan():
    plan = build_single_layer(receiver_count=2, k=1, d_model=2)
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
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
                "output_shape": [1],
            },
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
                "output_shape": [2],
            },
            node_compute={
                "type": "affine_residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
                "bias": True,
                "output_shape": [2],
            },
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 0.7,
                "output_shape": [2],
            },
        )
        for node in plan.nodes
    )
    region = dataclasses.replace(
        plan.regions[0],
        profile="BO",
        selector_timing="post",
        score={
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
        },
    )
    return _replace_plan(plan, nodes=nodes, regions=(region,))


def _state_plan(kind: str):
    plan = build_singleton(d_model=2)
    if kind == "gdn":
        state_shape = (2, 3)
        update = {
            "type": "gdn",
            "formula_id": "state.gdn.v1",
            "key_dim": 2,
            "value_dim": 3,
            "norm_eps": 1e-6,
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
            "norm_eps": 1e-6,
            "state_shape": [3, 2, 3],
        }
        selector_read = {
            "type": "content_state_summary_linear",
            "formula_id": "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
            "out_dim": 1,
            "output_shape": [1],
        }
    else:  # pragma: no cover - helper guard
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
    )
    region = dataclasses.replace(
        plan.regions[0], profile="BO", selector_timing="post"
    )
    return _replace_plan(plan, nodes=(node,), regions=(region,))


@unittest.skipUnless(
    bool(_TEST_BACKEND),
    "set TEST_BACKEND to require live SettleGraph backend semantics",
)
class LiveBackendSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _TEST_DTYPE not in {"float64", "float32"}:
            raise RuntimeError(
                "the live SettleGraph qualification subset currently supports "
                "only float64 and float32"
            )
        cls.runtime = resolve_runtime(
            RuntimeRequest(
                _TEST_BACKEND,
                _TEST_DEVICE_INDEX,
                _TEST_DTYPE,
                1729,
            )
        )
        cls.tolerance = (
            CPU_FLOAT64_TOLERANCE
            if _TEST_DTYPE == "float64"
            else SAME_BACKEND_FLOAT32_TOLERANCE
        )

    @classmethod
    def _typed(cls, plan):
        return bind_dtypes(
            plan,
            hidden=_TEST_DTYPE,
            parameter=_TEST_DTYPE,
            state=_TEST_DTYPE,
            readout=_TEST_DTYPE,
        )

    @classmethod
    def _model(cls):
        typed = cls._typed(_trainable_plan())
        model = SettleGraph(typed).to(
            device=cls.runtime.device, dtype=cls.runtime.dtype
        )
        selector = model.selector("region.0")
        with torch.no_grad():
            for index, node_id in enumerate(model.plan.regions[0].node_ids):
                linear = selector.linears[safe_module_key(node_id)]
                linear.weight.zero_()
                linear.bias.fill_(2.0 if index == 0 else -2.0)
        return typed, model

    @classmethod
    def _inputs(cls, *, requires_grad: bool = False):
        hidden = torch.tensor(
            [
                [[0.25, -0.5], [1.0, 0.75], [-0.25, 1.5]],
                [[-0.75, 0.5], [0.125, -1.0], [9.0, 8.0]],
            ],
            device=cls.runtime.device,
            dtype=cls.runtime.dtype,
            requires_grad=requires_grad,
        )
        execution = torch.tensor(
            [[True, True, True], [True, True, False]],
            device=cls.runtime.device,
            dtype=torch.bool,
        )
        positions = torch.tensor(
            [[0, 1, 2], [0, 1, 99]],
            device=cls.runtime.device,
            dtype=torch.int64,
        )
        return hidden, execution, positions

    def test_forward_state_trace_and_backward_match_both_schedules(self) -> None:
        _, model = self._model()
        hidden, execution, positions = self._inputs(requires_grad=True)
        token_result = model.prefill(
            hidden,
            execution,
            ["long", "short"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        cotangent = torch.tensor(
            [
                [[0.5, -0.25], [0.75, 0.125], [-0.5, 0.25]],
                [[0.25, 0.5], [-0.75, 1.0], [0.0, 0.0]],
            ],
            device=self.runtime.device,
            dtype=self.runtime.dtype,
        )
        token_loss = (token_result.output * cotangent).sum()
        token_loss = token_loss + 0.2 * token_result.balance_loss
        token_loss.backward()
        token_input_gradient = hidden.grad.detach().clone()
        token_parameter_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }

        model.zero_grad(set_to_none=True)
        region_hidden, _, _ = self._inputs(requires_grad=True)
        region_result = model.prefill_region_major(
            region_hidden,
            execution,
            ["long", "short"],
            positions,
            detach_at_end=False,
            record_trace=True,
        )
        region_loss = (region_result.output * cotangent).sum()
        region_loss = region_loss + 0.2 * region_result.balance_loss
        region_loss.backward()
        region_parameter_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        self.runtime.synchronize()

        self.assertEqual(region_result.output.device.type, self.runtime.info.backend)
        assert token_result.trace is not None
        assert region_result.trace is not None
        executed = (
            ("long", 0),
            ("long", 1),
            ("long", 2),
            ("short", 0),
            ("short", 1),
        )
        validate_trace_invariants(
            model.plan,
            token_result.trace,
            executed,
            tolerance=self.tolerance,
        )
        validate_trace_invariants(
            model.plan,
            region_result.trace,
            executed,
            tolerance=self.tolerance,
        )
        compare_nested(
            token_result,
            region_result,
            tolerance=self.tolerance,
        ).require_pass()
        compare_nested(
            token_input_gradient,
            region_hidden.grad,
            tolerance=self.tolerance,
        ).require_pass()
        compare_nested(
            token_parameter_gradients,
            region_parameter_gradients,
            tolerance=self.tolerance,
        ).require_pass()

    def test_gdn_and_attention_forward_backward(self) -> None:
        hidden, execution, positions = self._inputs(requires_grad=False)
        for kind in ("gdn", "attention_window"):
            typed = self._typed(_state_plan(kind))
            model = SettleGraph(typed).to(
                device=self.runtime.device, dtype=self.runtime.dtype
            )
            source = hidden[:1].detach().clone().requires_grad_(True)
            result = model.prefill_region_major(
                source,
                execution[:1],
                [kind],
                positions[:1],
                detach_at_end=False,
                record_trace=True,
            )
            (result.output.square().mean() + result.balance_loss).backward()
            self.runtime.synchronize()
            with self.subTest(kind=kind):
                self.assertEqual(result.output.device.type, self.runtime.info.backend)
                self.assertIsNotNone(source.grad)
                self.assertTrue(bool(torch.isfinite(source.grad).all().item()))
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in model.parameters()
                    )
                )
                assert result.trace is not None
                validate_trace_invariants(
                    model.plan,
                    result.trace,
                    ((kind, 0), (kind, 1), (kind, 2)),
                    tolerance=self.tolerance,
                )

    def test_optimizer_and_checkpoint_resume(self) -> None:
        typed, model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        hidden, execution, positions = self._inputs(requires_grad=False)
        first = model.prefill_region_major(
            hidden[:1, :1],
            execution[:1, :1],
            ["sequence"],
            positions[:1, :1],
            detach_at_end=False,
        )
        optimizer.zero_grad(set_to_none=True)
        first.output.square().sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portable.pt"
            artifact = save_checkpoint(
                path,
                model=model,
                typed_plan=typed,
                state=first.state,
                optimizer=optimizer,
                progress={"global_step": 1, "token_count": 1},
                training_state={"statistics_window_boundary": True},
            )
            restored = SettleGraph(typed).to(
                device=self.runtime.device, dtype=self.runtime.dtype
            )
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
            loaded = load_checkpoint(
                path,
                model=restored,
                typed_plan=typed,
                mode="resume",
                optimizer=restored_optimizer,
                expected_sha256=artifact.sha256,
            )
            next_hidden = hidden[:1, 1:2]
            next_execution = execution[:1, 1:2]
            next_positions = positions[:1, 1:2]
            expected = model.prefill_region_major(
                next_hidden,
                next_execution,
                ["sequence"],
                next_positions,
                state=first.state,
                detach_at_end=False,
            )
            actual = restored.prefill_region_major(
                next_hidden,
                next_execution,
                ["sequence"],
                next_positions,
                state=loaded.state,
                detach_at_end=False,
            )
            self.runtime.synchronize()
            self.assertEqual(loaded.progress["global_step"], 1)
            self.assertTrue(restored_optimizer.state_dict()["state"])
            compare_nested(
                expected,
                actual,
                tolerance=self.tolerance,
            ).require_pass()


if __name__ == "__main__":
    unittest.main()
