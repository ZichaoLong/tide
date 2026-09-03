from __future__ import annotations

import copy
import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from tide.builders import build_chain, build_singleton
from tide.checkpoint import (
    SCHEMA_VERSION,
    CheckpointError,
    deserialize_state_store,
    load_checkpoint,
    save_checkpoint,
    serialize_state_store,
)
from tide.engine import SettleGraph, StateStore
from tide.ops import AttentionState
from tide.plan import bind_dtypes


def _typed_float64(plan):
    return bind_dtypes(
        plan,
        hidden="float64",
        parameter="float64",
        state="float64",
        readout="float64",
    )


def _stateful_singleton():
    plan = build_singleton(d_model=2)
    node = dataclasses.replace(
        plan.nodes[0],
        state_shape=(2,),
        state_owner=plan.nodes[0].node_id,
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
            "type": "affine_residual",
            "formula_id": "TEST-NODE-AFFINE-V1",
            "bias": True,
            "output_shape": [2],
        },
    )
    region = dataclasses.replace(
        plan.regions[0], profile="BO", selector_timing="content"
    )
    return dataclasses.replace(plan, nodes=(node,), regions=(region,)).validate()


def _attention_singleton():
    plan = _stateful_singleton()
    node = dataclasses.replace(
        plan.nodes[0],
        state_shape=(3, 2, 3),
        update={
            "type": "attention_window",
            "formula_id": "state.attention-window.v1",
            "key_dim": 2,
            "value_dim": 3,
            "window": 3,
            "norm_eps": 1e-12,
            "state_shape": [3, 2, 3],
        },
        ffn_read={
            "type": "state_default",
            "formula_id": "read.ffn.attention-window.v1",
            "output_shape": [2],
        },
    )
    return dataclasses.replace(plan, nodes=(node,)).validate()


def _assert_state_close(test, left, right):
    test.assertEqual(left.next_position, right.next_position)
    test.assertEqual(set(left.values), set(right.values))
    for key in left.values:
        left_value = left.values[key]
        right_value = right.values[key]
        if isinstance(left_value, AttentionState):
            test.assertIsInstance(right_value, AttentionState)
            torch.testing.assert_close(
                left_value.positions, right_value.positions, atol=0, rtol=0
            )
            torch.testing.assert_close(
                left_value.keys, right_value.keys, atol=0, rtol=0
            )
            torch.testing.assert_close(
                left_value.values, right_value.values, atol=0, rtol=0
            )
        elif left_value is None:
            test.assertIsNone(right_value)
        else:
            torch.testing.assert_close(left_value, right_value, atol=0, rtol=0)
    test.assertEqual(set(left.selector_history), set(right.selector_history))
    for key in left.selector_history:
        torch.testing.assert_close(
            left.selector_history[key],
            right.selector_history[key],
            atol=0,
            rtol=0,
        )


def _assert_nested_exact(test, left, right, path="root"):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertEqual(left.device, right.device, path)
        test.assertEqual(left.dtype, right.dtype, path)
        test.assertEqual(left.shape, right.shape, path)
        test.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict):
        test.assertIsInstance(right, dict, path)
        test.assertEqual(set(left), set(right), path)
        for key in left:
            _assert_nested_exact(test, left[key], right[key], f"{path}[{key!r}]")
    elif isinstance(left, (list, tuple)):
        test.assertIs(type(left), type(right), path)
        test.assertEqual(len(left), len(right), path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_exact(
                test, left_item, right_item, f"{path}[{index}]"
            )
    else:
        test.assertEqual(left, right, path)


class _FailingLoadSettleGraph(SettleGraph):
    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs)
        raise RuntimeError("injected model commit failure")


class CheckpointTests(unittest.TestCase):
    def _trained_fixture(self):
        typed = _typed_float64(_stateful_singleton())
        model = SettleGraph(typed).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = model.interpret_token(
            torch.tensor([[1.0, -2.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
        )
        optimizer.zero_grad(set_to_none=True)
        result.output.square().sum().backward()
        optimizer.step()
        return typed, model, optimizer, result.state

    def _assert_resume_failure_is_atomic(
        self,
        path,
        typed,
        error_pattern,
        *,
        model_type=SettleGraph,
    ):
        target = model_type(typed).double()
        optimizer = torch.optim.Adam(target.parameters(), lr=0.25)
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in target.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model_before = {
            key: value.detach().clone()
            for key, value in target.state_dict().items()
        }
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        rng_before = torch.get_rng_state().clone()

        with self.assertRaisesRegex(CheckpointError, error_pattern):
            load_checkpoint(
                path,
                model=target,
                typed_plan=typed,
                mode="resume",
                optimizer=optimizer,
            )

        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, model_before[key], atol=0, rtol=0)
        _assert_nested_exact(
            self,
            optimizer.state_dict(),
            optimizer_before,
            "optimizer",
        )
        torch.testing.assert_close(
            torch.get_rng_state(), rng_before, atol=0, rtol=0
        )

    def test_malformed_root_rng_and_adam_tensor_fail_atomically(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.pt"
            save_checkpoint(
                valid_path,
                model=model,
                typed_plan=typed,
                state=state,
                optimizer=optimizer,
            )

            cases = []

            root_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            root_payload["unexpected"] = "not part of checkpoint v1"
            cases.append(
                (
                    "root-extra-key.pt",
                    root_payload,
                    "checkpoint root has an unexpected key set",
                )
            )

            rng_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            rng_payload["rng_state"] = {
                "torch_cpu": torch.zeros((1,), dtype=torch.uint8)
            }
            cases.append(
                (
                    "rng-shape-one.pt",
                    rng_payload,
                    "checkpoint CPU RNG state cannot be restored",
                )
            )

            optimizer_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            optimizer_torch_state = optimizer_payload["optimizer_state"][
                "torch_state_dict"
            ]
            first_state = next(
                iter(optimizer_torch_state["state"].values())
            )
            first_state["exp_avg"] = torch.zeros(
                (999,), dtype=first_state["exp_avg"].dtype
            )
            cases.append(
                (
                    "adam-exp-avg-shape.pt",
                    optimizer_payload,
                    r"optimizer_state\.state\[\d+\]\['exp_avg'\].*shape",
                )
            )

            for filename, key, value, error_pattern in (
                (
                    "adam-lr-nan.pt",
                    "lr",
                    float("nan"),
                    r"param_groups\[0\]\['lr'\].*finite",
                ),
                (
                    "adam-betas-invalid.pt",
                    "betas",
                    (2.0, 2.0),
                    r"betas.*less than 1",
                ),
                (
                    "adam-eps-negative.pt",
                    "eps",
                    -1.0,
                    r"param_groups\[0\]\['eps'\].*value > 0",
                ),
            ):
                payload = torch.load(
                    valid_path, map_location="cpu", weights_only=True
                )
                payload["optimizer_state"]["torch_state_dict"][
                    "param_groups"
                ][0][key] = value
                cases.append((filename, payload, error_pattern))

            unsafe_metadata_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            unsafe_metadata_payload["training_state"] = {
                "nested": {1: b"weights-only accepts this, project v1 does not"}
            }
            cases.append(
                (
                    "unsafe-training-metadata.pt",
                    unsafe_metadata_payload,
                    "training_state.*mapping keys must be strings",
                )
            )

            for filename, payload, error_pattern in cases:
                with self.subTest(filename=filename):
                    corrupt_path = root / filename
                    torch.save(payload, corrupt_path)
                    self._assert_resume_failure_is_atomic(
                        corrupt_path, typed, error_pattern
                    )

    def test_optimizer_mapping_state_manifest_and_storage_are_strict(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.pt"
            save_checkpoint(
                valid_path,
                model=model,
                typed_plan=typed,
                state=state,
                optimizer=optimizer,
            )

            missing_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            missing_state = missing_payload["optimizer_state"][
                "torch_state_dict"
            ]["state"]
            del missing_state[next(iter(missing_state))]
            missing_path = root / "missing-optimizer-state.pt"
            torch.save(missing_payload, missing_path)
            self._assert_resume_failure_is_atomic(
                missing_path,
                typed,
                "initialized-state parameter manifest does not match",
            )

            alias_payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            alias_state = next(
                iter(
                    alias_payload["optimizer_state"]["torch_state_dict"][
                        "state"
                    ].values()
                )
            )
            alias_state["exp_avg_sq"] = alias_state["exp_avg"]
            alias_path = root / "aliased-optimizer-state.pt"
            torch.save(alias_payload, alias_path)
            self._assert_resume_failure_is_atomic(
                alias_path, typed, "mutable optimizer storage is shared"
            )

            target = SettleGraph(typed).double()
            clone_optimizer = torch.optim.Adam(
                [
                    torch.nn.Parameter(parameter.detach().clone())
                    for parameter in target.parameters()
                ],
                lr=1e-3,
            )
            with self.assertRaisesRegex(
                CheckpointError, "is not owned by the checkpoint model"
            ):
                load_checkpoint(
                    valid_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                    optimizer=clone_optimizer,
                )

            with self.assertRaisesRegex(
                CheckpointError, "target optimizer type does not match"
            ):
                load_checkpoint(
                    valid_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                    optimizer=torch.optim.AdamW(target.parameters(), lr=1e-3),
                )

    def test_save_rejects_non_weights_only_metadata_before_creating_parent(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(
                CheckpointError, "unsupported weights-only type object"
            ):
                save_checkpoint(
                    parent / "checkpoint.pt",
                    model=model,
                    typed_plan=typed,
                    state=state,
                    optimizer=optimizer,
                    training_state={"opaque": object()},
                )
            self.assertFalse(parent.exists())

            foreign_optimizer = torch.optim.Adam(
                [
                    torch.nn.Parameter(parameter.detach().clone())
                    for parameter in model.parameters()
                ],
                lr=1e-3,
            )
            with self.assertRaisesRegex(
                CheckpointError, "is not owned by the checkpoint model"
            ):
                save_checkpoint(
                    parent / "foreign-optimizer.pt",
                    model=model,
                    typed_plan=typed,
                    state=state,
                    optimizer=foreign_optimizer,
                )
            self.assertFalse(parent.exists())

    def test_commit_failure_rolls_back_model_optimizer_and_rng(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                typed_plan=typed,
                state=state,
                optimizer=optimizer,
            )
            self._assert_resume_failure_is_atomic(
                path,
                typed,
                "checkpoint commit failed.*injected model commit failure",
                model_type=_FailingLoadSettleGraph,
            )

    def test_resume_round_trip_restores_model_optimizer_state_and_continuation(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            artifact = save_checkpoint(
                path,
                model=model,
                typed_plan=typed,
                state=state,
                optimizer=optimizer,
                progress={"global_step": 1, "token_count": 1},
                training_state={"balance_window_index": 0},
            )
            restored_model = SettleGraph(typed).double()
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=1e-3
            )
            loaded = load_checkpoint(
                path,
                model=restored_model,
                typed_plan=typed,
                mode="resume",
                optimizer=restored_optimizer,
                expected_sha256=artifact.sha256,
            )
            self.assertEqual(loaded.progress["global_step"], 1)
            _assert_state_close(self, loaded.state, state)
            self.assertTrue(restored_optimizer.state_dict()["state"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    restored_model.state_dict()[key], value, atol=0, rtol=0
                )

            next_hidden = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
            original_next = model.interpret_token(
                next_hidden,
                torch.tensor([True]),
                ["s"],
                torch.tensor([1]),
                state=state,
            )
            restored_next = restored_model.interpret_token(
                next_hidden,
                torch.tensor([True]),
                ["s"],
                torch.tensor([1]),
                state=loaded.state,
            )
            torch.testing.assert_close(
                restored_next.output, original_next.output, atol=0, rtol=0
            )
            _assert_state_close(self, restored_next.state, original_next.state)

    def test_init_from_loads_weights_but_not_runtime_or_optimizer_state(self):
        typed, model, optimizer, state = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                typed_plan=typed,
                state=state,
                optimizer=optimizer,
                progress={"global_step": 9},
            )
            target = SettleGraph(typed).double()
            target_optimizer = torch.optim.Adam(target.parameters(), lr=0.25)
            loaded = load_checkpoint(
                path,
                model=target,
                typed_plan=typed,
                mode="init-from",
                optimizer=target_optimizer,
            )
            self.assertEqual(loaded.state, StateStore())
            self.assertEqual(loaded.progress, {})
            self.assertEqual(target_optimizer.state_dict()["state"], {})
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    target.state_dict()[key], value, atol=0, rtol=0
                )

    def test_attention_state_round_trip_and_continuation(self):
        typed = _typed_float64(_attention_singleton())
        model = SettleGraph(typed).double()
        first = model.prefill(
            torch.tensor(
                [[[1.0, -0.5], [0.25, 2.0]]], dtype=torch.float64
            ),
            torch.ones((1, 2), dtype=torch.bool),
            ["s"],
            torch.tensor([[0, 1]]),
            detach_at_end=False,
        )
        state = first.state.values[("s", typed.logical_plan.nodes[0].node_id)]
        self.assertIsInstance(state, AttentionState)
        self.assertEqual(state.positions.tolist(), [0, 1])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.pt"
            save_checkpoint(path, model=model, typed_plan=typed, state=first.state)
            restored_model = SettleGraph(typed).double()
            loaded = load_checkpoint(
                path,
                model=restored_model,
                typed_plan=typed,
                mode="resume",
            )
            _assert_state_close(self, loaded.state, first.state)

            hidden = torch.tensor([[0.75, -1.25]], dtype=torch.float64)
            expected = model.interpret_token(
                hidden,
                torch.tensor([True]),
                ["s"],
                torch.tensor([2]),
                state=first.state,
            )
            actual = restored_model.interpret_token(
                hidden,
                torch.tensor([True]),
                ["s"],
                torch.tensor([2]),
                state=loaded.state,
            )
            torch.testing.assert_close(actual.output, expected.output, atol=0, rtol=0)
            _assert_state_close(self, actual.state, expected.state)

    def test_serialized_cpu_state_alias_is_rejected_before_device_transfer(self):
        typed = _typed_float64(_stateful_singleton())
        model = SettleGraph(typed).double()
        node_id = typed.logical_plan.nodes[0].node_id
        state = StateStore(
            values={
                ("a", node_id): torch.zeros(2, dtype=torch.float64),
                ("b", node_id): torch.ones(2, dtype=torch.float64),
            },
            next_position={"a": 1, "b": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.pt"
            corrupt_path = Path(directory) / "aliased-state.pt"
            save_checkpoint(
                valid_path, model=model, typed_plan=typed, state=state
            )
            payload = torch.load(
                valid_path, map_location="cpu", weights_only=True
            )
            backing = torch.zeros(4, dtype=torch.float64)
            receiver_values = payload["sequence_state"]["receiver_values"]
            receiver_values[0]["payload"]["value"] = backing[:2]
            receiver_values[1]["payload"]["value"] = backing[2:]
            torch.save(payload, corrupt_path)

            target = SettleGraph(typed).double().to("meta")
            with self.assertRaisesRegex(
                CheckpointError, "mutable state storage is shared"
            ):
                load_checkpoint(
                    corrupt_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                )

    def test_deserialize_rejects_state_history_and_attention_dtype_casts(self):
        base = {
            "receiver_values": [],
            "selector_history": [],
            "next_position": {"s": 1},
        }
        cases = {
            "receiver": {
                **base,
                "receiver_values": [
                    {
                        "sequence_id": "s",
                        "node_id": "n",
                        "payload": {
                            "kind": "tensor",
                            "value": torch.zeros(2, dtype=torch.float32),
                        },
                    }
                ],
            },
            "history": {
                **base,
                "selector_history": [
                    {
                        "sequence_id": "s",
                        "owner_id": "n",
                        "value": torch.zeros((), dtype=torch.float32),
                    }
                ],
            },
            "attention_positions": {
                **base,
                "receiver_values": [
                    {
                        "sequence_id": "s",
                        "node_id": "n",
                        "payload": {
                            "kind": "attention_window",
                            "positions": torch.tensor([0.0]),
                            "keys": torch.zeros((1, 2), dtype=torch.float64),
                            "values": torch.zeros((1, 3), dtype=torch.float64),
                        },
                    }
                ],
            },
            "attention_values": {
                **base,
                "receiver_values": [
                    {
                        "sequence_id": "s",
                        "node_id": "n",
                        "payload": {
                            "kind": "attention_window",
                            "positions": torch.tensor([0], dtype=torch.int64),
                            "keys": torch.zeros((1, 2), dtype=torch.float64),
                            "values": torch.zeros((1, 3), dtype=torch.float32),
                        },
                    }
                ],
            },
        }
        for name, record in cases.items():
            with self.subTest(name=name), self.assertRaises(CheckpointError):
                deserialize_state_store(
                    record, device="cpu", dtype=torch.float64
                )

    def test_checkpoint_state_ids_use_the_strict_stable_string_contract(self):
        invalid_ids = (7, "e\u0301", " leading", "trailing ", "bad\x00id", "\ud800")
        for invalid_id in invalid_ids:
            records = {
                "position": {
                    "receiver_values": [],
                    "selector_history": [],
                    "next_position": {invalid_id: 0},
                },
                "receiver-sequence": {
                    "receiver_values": [
                        {
                            "sequence_id": invalid_id,
                            "node_id": "node.0",
                            "payload": {"kind": "none"},
                        }
                    ],
                    "selector_history": [],
                    "next_position": {},
                },
                "receiver-node": {
                    "receiver_values": [
                        {
                            "sequence_id": "sequence",
                            "node_id": invalid_id,
                            "payload": {"kind": "none"},
                        }
                    ],
                    "selector_history": [],
                    "next_position": {},
                },
                "history-owner": {
                    "receiver_values": [],
                    "selector_history": [
                        {
                            "sequence_id": "sequence",
                            "owner_id": invalid_id,
                            "value": torch.zeros((), dtype=torch.float64),
                        }
                    ],
                    "next_position": {},
                },
            }
            for name, record in records.items():
                with self.subTest(name=name, value=repr(invalid_id)):
                    with self.assertRaises(CheckpointError):
                        deserialize_state_store(
                            record, device="cpu", dtype=torch.float64
                        )

            stores = (
                StateStore(next_position={invalid_id: 0}),
                StateStore(values={("sequence", invalid_id): None}),
                StateStore(
                    selector_history={
                        ("sequence", invalid_id): torch.zeros(
                            (), dtype=torch.float64
                        )
                    }
                ),
            )
            for index, state in enumerate(stores):
                with self.subTest(
                    name="serialize", index=index, value=repr(invalid_id)
                ):
                    with self.assertRaises(CheckpointError):
                        serialize_state_store(state)

    def test_model_plan_binding_and_parameter_dtype_are_not_relabelled(self):
        typed, model, optimizer, state = self._trained_fixture()
        del optimizer
        wrong_logical = _typed_float64(build_chain(length=2, d_model=2))
        wrong_binding = bind_dtypes(
            typed.logical_plan,
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(CheckpointError, "logical Plan"):
                save_checkpoint(
                    root / "wrong-logical.pt",
                    model=model,
                    typed_plan=wrong_logical,
                    state=state,
                )
            with self.assertRaisesRegex(CheckpointError, "typed Plan"):
                save_checkpoint(
                    root / "wrong-binding.pt",
                    model=model,
                    typed_plan=wrong_binding,
                    state=state,
                )
            with self.assertRaisesRegex(CheckpointError, "parameter.*dtype"):
                save_checkpoint(
                    root / "wrong-parameter-dtype.pt",
                    model=SettleGraph(typed),
                    typed_plan=typed,
                    state=StateStore(),
                )
            with self.assertRaisesRegex(
                CheckpointError, "constructed from a TypedPlan"
            ):
                save_checkpoint(
                    root / "untyped-model.pt",
                    model=SettleGraph(typed.logical_plan).double(),
                    typed_plan=typed,
                    state=StateStore(),
                )
            self.assertEqual(list(root.iterdir()), [])

            valid_path = root / "valid.pt"
            save_checkpoint(valid_path, model=model, typed_plan=typed, state=state)
            with self.assertRaisesRegex(CheckpointError, "logical Plan"):
                load_checkpoint(
                    valid_path,
                    model=SettleGraph(wrong_logical).double(),
                    typed_plan=typed,
                    mode="resume",
                )
            with self.assertRaisesRegex(CheckpointError, "parameter.*dtype"):
                load_checkpoint(
                    valid_path,
                    model=SettleGraph(typed).float(),
                    typed_plan=typed,
                    mode="resume",
                )

    def test_malformed_state_dtype_fails_before_model_mutation(self):
        typed, model, optimizer, state = self._trained_fixture()
        del optimizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            corrupt_path = Path(directory) / "corrupt-state.pt"
            save_checkpoint(path, model=model, typed_plan=typed, state=state)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            sequence_state = dict(payload["sequence_state"])
            receiver_values = list(sequence_state["receiver_values"])
            item = dict(receiver_values[0])
            state_payload = dict(item["payload"])
            state_payload["value"] = state_payload["value"].float()
            item["payload"] = state_payload
            receiver_values[0] = item
            sequence_state["receiver_values"] = receiver_values
            payload["sequence_state"] = sequence_state
            torch.save(payload, corrupt_path)

            for mode in ("resume", "init-from"):
                with self.subTest(mode=mode):
                    target = SettleGraph(typed).double()
                    before = {
                        key: value.detach().clone()
                        for key, value in target.state_dict().items()
                    }
                    with self.assertRaisesRegex(
                        CheckpointError, "binding state dtype"
                    ):
                        load_checkpoint(
                            corrupt_path,
                            model=target,
                            typed_plan=typed,
                            mode=mode,
                        )
                    for key, value in target.state_dict().items():
                        torch.testing.assert_close(
                            value, before[key], atol=0, rtol=0
                        )

    def test_attention_positions_precede_next_position_but_may_be_older(self):
        typed = _typed_float64(_attention_singleton())
        model = SettleGraph(typed).double()
        result = model.interpret_token(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([True]),
            ["s"],
            torch.tensor([0]),
        )
        older = StateStore(
            values=result.state.values,
            next_position={"s": 3},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "older-attention.pt"
            save_checkpoint(path, model=model, typed_plan=typed, state=older)
            loaded = load_checkpoint(
                path,
                model=SettleGraph(typed).double(),
                typed_plan=typed,
                mode="resume",
            )
            _assert_state_close(self, loaded.state, older)

            invalid = StateStore(
                values=result.state.values,
                next_position={"s": 0},
            )
            with self.assertRaisesRegex(CheckpointError, "must precede"):
                save_checkpoint(
                    Path(directory) / "future-attention.pt",
                    model=model,
                    typed_plan=typed,
                    state=invalid,
                )

            corrupt_path = Path(directory) / "future-attention-load.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            sequence_state = dict(payload["sequence_state"])
            sequence_state["next_position"] = {"s": 0}
            payload["sequence_state"] = sequence_state
            torch.save(payload, corrupt_path)
            target = SettleGraph(typed).double()
            before = {
                key: value.detach().clone()
                for key, value in target.state_dict().items()
            }
            with self.assertRaisesRegex(CheckpointError, "must precede"):
                load_checkpoint(
                    corrupt_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                )
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_v1_state_uses_next_position_and_strictly_rejects_old_field(self):
        state = StateStore(next_position={"s": 3})
        record = serialize_state_store(state)
        self.assertEqual(SCHEMA_VERSION, "tide.settlegraph.checkpoint.v1")
        self.assertEqual(record["next_position"], {"s": 3})
        self.assertNotIn("last_position", record)
        restored = deserialize_state_store(
            record, device="cpu", dtype=torch.float64
        )
        self.assertEqual(restored.next_position, {"s": 3})

        legacy_record = dict(record)
        legacy_record["last_position"] = legacy_record.pop("next_position")
        with self.assertRaisesRegex(CheckpointError, "unexpected key set"):
            deserialize_state_store(
                legacy_record, device="cpu", dtype=torch.float64
            )

    def test_old_position_field_is_rejected_before_model_mutation(self):
        typed, model, optimizer, state = self._trained_fixture()
        del optimizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            legacy_path = Path(directory) / "checkpoint-old-position.pt"
            save_checkpoint(path, model=model, typed_plan=typed, state=state)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            sequence_state = dict(payload["sequence_state"])
            sequence_state["last_position"] = sequence_state.pop(
                "next_position"
            )
            payload["sequence_state"] = sequence_state
            torch.save(payload, legacy_path)

            target = SettleGraph(typed).double()
            before = {
                key: value.detach().clone()
                for key, value in target.state_dict().items()
            }
            with self.assertRaisesRegex(CheckpointError, "unexpected key set"):
                load_checkpoint(
                    legacy_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                )
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_hash_or_plan_mismatch_fails_before_model_mutation(self):
        typed, model, optimizer, state = self._trained_fixture()
        del optimizer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model=model, typed_plan=typed, state=state)
            target = SettleGraph(typed).double()
            before = {
                key: value.detach().clone()
                for key, value in target.state_dict().items()
            }
            with self.assertRaisesRegex(CheckpointError, "SHA-256"):
                load_checkpoint(
                    path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                    expected_sha256="0" * 64,
                )
            corrupt_path = Path(directory) / "wrong-plan-hash.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["logical_plan_hash"] = "0" * 64
            torch.save(payload, corrupt_path)
            with self.assertRaisesRegex(CheckpointError, "logical Plan hash"):
                load_checkpoint(
                    corrupt_path,
                    model=target,
                    typed_plan=typed,
                    mode="resume",
                )
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
