from __future__ import annotations

import concurrent.futures
import copy
import dataclasses
import hashlib
import os
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from tide.builders import build_single_layer, build_singleton, build_small_hb
from tide.engine import SettleGraph, StateStore
from tide.fixtures import (
    FIXTURE_SCHEMA_VERSION,
    FixtureError,
    _decode_logical_plan,
    _decode_typed_plan,
    _record_hash,
    _seal_payload,
    eager_parameter_tensors,
    load_fixture_bundle,
    save_fixture_bundle,
    save_negative_fixture_bundle,
)
from tide.failures import ExecutionFailed, FailureEnvelope, capture_execution
from tide.parameter_manifest import build_eager_parameter_manifest
from tide.parameter_manifest import load_eager_parameter_tensors
from tide.plan import bind_dtypes


def _typed(plan, dtype: str = "float64"):
    return bind_dtypes(
        plan,
        hidden=dtype,
        parameter=dtype,
        state=dtype,
        readout=dtype,
    )


def _set_gradient_contract(parts) -> None:
    paths = {
        "inputs.hidden",
        *(f"parameters.{key}" for key in parts["parameters"]),
        *(
            f"learnable_initial_state.{key}"
            for key in parts["learnable_initial_state"]
        ),
    }
    previous = parts["gradient"].get("path_assertions", {})
    parts["gradient"]["required_keys"] = sorted(paths)
    parts["gradient"]["path_assertions"] = {
        path: previous.get(
            path, "connected" if path == "inputs.hidden" else "disconnected"
        )
        for path in sorted(paths)
    }


def _fixture_parts(dtype: str = "float64"):
    typed = _typed(build_singleton(d_model=2), dtype)
    model = SettleGraph(typed).to(dtype=getattr(torch, dtype))
    parameter_schema = build_eager_parameter_manifest(model)
    hidden = torch.tensor(
        [
            [[1.0, -2.0], [0.25, 0.5], [8.0, 9.0]],
            [[-1.0, 3.0], [7.0, 6.0], [0.75, -0.25]],
        ],
        dtype=getattr(torch, dtype),
    )
    execution = torch.tensor(
        [[True, True, False], [True, False, True]], dtype=torch.bool
    )
    inputs = {
        "hidden": hidden,
        "sequence_ids": ["seq.a", "seq.b"],
        "token_positions": torch.tensor(
            [[0, 1, 999], [0, 999, 1]], dtype=torch.int64
        ),
        "execution_mask": execution,
        "lm_target_mask": torch.tensor(
            [[False, True, False], [False, False, True]], dtype=torch.bool
        ),
        "routing_stats_mask": execution.clone(),
    }
    control = {
        "requested_k": {},
        "reset_sequence_ids": [],
        "chunk_boundaries": [1, 2],
        "detach_boundaries": [2],
        "random_keys": {"seed": 1729, "version": "tide.random-key.v1"},
    }
    parts = {
        "fixture_id": f"fixture.singleton.{dtype}",
        "typed_plan": typed,
        "source": {
            "kind": "generated",
            "identifier": "tests.test_fixtures._fixture_parts",
            "command": "python -m unittest tests.test_fixtures",
        },
        "inputs": inputs,
        "parameters": eager_parameter_tensors(model, parameter_schema),
        "learnable_initial_state": {},
        "initial_state": StateStore(),
        "control": control,
        "expected": {
            "outcome": "success",
            "golden": {"output": hidden.clone()},
        },
        "gradient": {
            "objective": "output-cotangent-v1",
            "output_cotangent": torch.full_like(hidden, 0.25),
            "alpha_lm": 0.0,
            "alpha_balance": 0.0,
            "required_keys": ["inputs.hidden"],
            "path_assertions": {"inputs.hidden": "connected"},
        },
        "routing_classification": "all-active",
        "parameter_schema": parameter_schema,
    }
    _set_gradient_contract(parts)
    return parts


def _rewrite_payload(path: Path, mutate) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def _refresh_content_hash(payload) -> None:
    payload["content_hash"] = _record_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _failure(phase: str, *codes: str):
    return {
        "outcome": "failure",
        "error": FailureEnvelope.create(phase, codes).to_dict(),
    }


class FixtureBundleTests(unittest.TestCase):
    def test_content_hash_domain_separates_bytes_from_user_mappings(self) -> None:
        parts = _fixture_parts()
        parts["control"]["random_keys"]["opaque"] = b"x"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            save_fixture_bundle(path, **parts)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["control"]["random_keys"]["opaque"] = {
                "$bytes_hex": "78"
            }
            torch.save(payload, path)

            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.integrity")

    def test_surrogate_metadata_has_stable_schema_failure(self) -> None:
        parts = _fixture_parts()
        parts["control"]["random_keys"]["label"] = "\ud800"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(Path(directory) / "fixture.pt", **parts)
            self.assertEqual(raised.exception.code, "artifact.schema")

            path = Path(directory) / "malformed.pt"
            save_fixture_bundle(path, **_fixture_parts())
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["control"]["random_keys"]["label"] = "\ud800"
            torch.save(payload, path)
            with self.assertRaises(FixtureError) as loaded:
                load_fixture_bundle(path)
            self.assertEqual(loaded.exception.code, "artifact.schema")

    def test_gradient_contract_requires_every_differentiable_leaf(self) -> None:
        mutations = (
            lambda gradient: gradient.update(
                required_keys=[], path_assertions={}
            ),
            lambda gradient: gradient.update(
                required_keys=["inputs.hidden", "parameters.not-real"],
                path_assertions={
                    "inputs.hidden": "connected",
                    "parameters.not-real": "disconnected",
                },
            ),
            lambda gradient: gradient["path_assertions"].__setitem__(
                "inputs.hidden", "absent"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(mutations):
                parts = _fixture_parts()
                mutate(parts["gradient"])
                with self.subTest(index=index), self.assertRaises(
                    FixtureError
                ) as raised:
                    save_fixture_bundle(
                        Path(directory) / f"bad-gradient-{index}.pt", **parts
                    )
                self.assertEqual(raised.exception.code, "artifact.schema")

    def test_tensor_manifest_authenticates_backing_storage_holes(self) -> None:
        parts = _fixture_parts()
        original = parts["inputs"]["hidden"]
        backing = torch.zeros(31, dtype=original.dtype)
        hidden = torch.as_strided(
            backing,
            size=tuple(original.shape),
            stride=(20, 4, 1),
            storage_offset=1,
        )
        hidden.copy_(original)
        parts["inputs"]["hidden"] = hidden
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            save_fixture_bundle(path, **parts)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            stored = payload["inputs"]["hidden"]
            storage = stored.untyped_storage()
            byte_view = torch.empty(0, dtype=torch.uint8)
            byte_view.set_(storage, 0, (storage.nbytes(),), (1,))
            byte_view[0] = 123
            torch.save(payload, path)

            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.integrity")

    def test_round_trip_preserves_cpu_values_and_separate_identities(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            artifact = save_fixture_bundle(path, **parts)
            repeated = save_fixture_bundle(Path(directory) / "fixture-copy.pt", **parts)
            loaded = load_fixture_bundle(path, expected_sha256=artifact.sha256)

            self.assertEqual(FIXTURE_SCHEMA_VERSION, "tide.settlegraph.fixture.v1")
            self.assertEqual(loaded.fixture_id, parts["fixture_id"])
            self.assertEqual(
                loaded.typed_plan.logical_hash(),
                parts["typed_plan"].logical_hash(),
            )
            self.assertEqual(
                loaded.typed_plan.typed_hash(), parts["typed_plan"].typed_hash()
            )
            self.assertEqual(loaded.artifact, artifact)
            self.assertNotEqual(artifact.sha256, artifact.content_hash)
            self.assertNotEqual(artifact.content_hash, artifact.tensor_artifact_hash)
            self.assertEqual(artifact.content_hash, repeated.content_hash)
            self.assertEqual(
                artifact.tensor_artifact_hash, repeated.tensor_artifact_hash
            )
            torch.testing.assert_close(
                loaded.inputs["hidden"], parts["inputs"]["hidden"], atol=0, rtol=0
            )
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in loaded.parameters.values()))
            self.assertEqual(
                {entry["path"] for entry in loaded.tensor_manifest},
                {
                    "/expected/golden/output",
                    "/gradient/output_cotangent",
                    "/inputs/execution_mask",
                    "/inputs/hidden",
                    "/inputs/lm_target_mask",
                    "/inputs/routing_stats_mask",
                    "/inputs/token_positions",
                    *{f"/parameters/{key.replace('~', '~0').replace('/', '~1')}" for key in loaded.parameters},
                },
            )

            restored = SettleGraph(loaded.typed_plan).double()
            restored_manifest = build_eager_parameter_manifest(restored)
            self.assertEqual(
                restored_manifest.canonical_dict(), dict(loaded.parameter_schema)
            )
            load_eager_parameter_tensors(
                restored, restored_manifest, loaded.parameters
            )
            result = restored.prefill(
                loaded.inputs["hidden"],
                loaded.inputs["execution_mask"],
                loaded.inputs["sequence_ids"],
                loaded.inputs["token_positions"],
                state=loaded.initial_state,
                requested_k=loaded.control["requested_k"] or None,
                lm_target_mask=loaded.inputs["lm_target_mask"],
                routing_stats_mask=loaded.inputs["routing_stats_mask"],
                reset_sequence_ids=loaded.control["reset_sequence_ids"],
            )
            torch.testing.assert_close(
                result.output,
                loaded.expected["golden"]["output"],
                atol=0,
                rtol=0,
            )

    def test_file_hash_is_checked_before_safe_decode(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            artifact = save_fixture_bundle(path, **parts)
            with path.open("ab") as stream:
                stream.write(b"corrupt-tail")
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path, expected_sha256=artifact.sha256)
            self.assertEqual(raised.exception.envelope.codes, ("artifact.integrity",))
            captured = capture_execution(
                lambda: load_fixture_bundle(path, expected_sha256=artifact.sha256)
            )
            self.assertIsInstance(captured, ExecutionFailed)
            assert isinstance(captured, ExecutionFailed)
            self.assertEqual(captured.envelope, raised.exception.envelope)

    def test_content_and_tensor_manifest_tampering_are_separate_gates(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content_path = root / "content.pt"
            save_fixture_bundle(content_path, **parts)
            _rewrite_payload(
                content_path,
                lambda payload: payload["inputs"]["hidden"].add_(1.0),
            )
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(content_path)
            self.assertEqual(raised.exception.code, "artifact.integrity")

            manifest_path = root / "manifest.pt"
            save_fixture_bundle(manifest_path, **parts)

            def mutate(payload):
                payload["inputs"]["hidden"].add_(1.0)
                _refresh_content_hash(payload)

            _rewrite_payload(manifest_path, mutate)
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(manifest_path)
            self.assertEqual(raised.exception.code, "artifact.integrity")

            typed_path = root / "manifest-bool.pt"
            save_fixture_bundle(typed_path, **parts)

            def replace_integer_with_bool(payload):
                payload["tensor_manifest"]["tensors"][0][
                    "storage_offset"
                ] = False
                payload["tensor_artifact_hash"] = _record_hash(
                    payload["tensor_manifest"]
                )
                _refresh_content_hash(payload)

            _rewrite_payload(typed_path, replace_integer_with_bool)
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(typed_path)
            self.assertEqual(raised.exception.code, "artifact.schema")

    def test_noncanonical_plan_bytes_fail_even_with_refreshed_hashes(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            save_fixture_bundle(path, **parts)

            def mutate(payload):
                payload["logical_plan_bytes"] += b"\n"
                payload["logical_plan_hash"] = hashlib.sha256(
                    payload["logical_plan_bytes"]
                ).hexdigest()
                _refresh_content_hash(payload)

            _rewrite_payload(path, mutate)
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.integrity")
            self.assertIn("not canonical", str(raised.exception))

    def test_raw_logical_plan_json_types_have_exact_schema_envelopes(self) -> None:
        plan = build_small_hb()
        canonical = plan.canonical_dict()
        cases = (
            ("topology-kind-scalar", ("topology_kind",), 1),
            ("dtype-roles-scalar", ("dtype_roles",), 1),
            ("output-aggregate-scalar", ("output_aggregate",), 1),
            ("entry-string", ("entry_node_ids",), plan.entry_node_ids[0]),
            (
                "terminal-string",
                ("terminal_node_ids",),
                plan.terminal_node_ids[0],
            ),
            ("hidden-shape-scalar", ("nodes", 0, "hidden_shape"), 1),
            ("node-config-scalar", ("nodes", 0, "emit"), 1),
            ("operation-type-object", ("nodes", 0, "update", "type"), {}),
            (
                "formula-id-array",
                ("nodes", 0, "update", "formula_id"),
                [],
            ),
            (
                "region-members-string",
                ("regions", 0, "node_ids"),
                plan.regions[0].node_ids[0],
            ),
            (
                "control-dependencies-string",
                ("regions", 1, "control_dependencies"),
                plan.regions[0].region_id,
            ),
            ("profile-scalar", ("regions", 0, "profile"), 1),
            (
                "selector-timing-scalar",
                ("regions", 0, "selector_timing"),
                1,
            ),
            ("k-max-string", ("regions", 0, "k_max"), "one"),
            ("hb-line-string", ("regions", 0, "line"), "zero"),
            ("hb-phase-scalar", ("regions", 0, "phase"), 1),
        )
        expected = FailureEnvelope.create("plan", "plan.schema")
        for case_name, path, value in cases:
            with self.subTest(case=case_name):
                record = copy.deepcopy(canonical)
                target = record
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                with self.assertRaises(FixtureError) as raised:
                    _decode_logical_plan(record, plan_id=f"raw.{case_name}")
                self.assertEqual(raised.exception.envelope, expected)

    def test_raw_logical_plan_values_retain_exact_semantic_envelopes(self) -> None:
        plan = build_small_hb()
        canonical = plan.canonical_dict()
        cases = (
            (
                "registered-formula-dispatch",
                ("output_aggregate",),
                {
                    "type": "node_softmax",
                    "formula_id": "agg.mean.v1",
                    "output_shape": [plan.d_model],
                },
                FailureEnvelope.create("plan", "plan.formula"),
            ),
            (
                "profile-value",
                ("regions", 0, "profile"),
                "invalid",
                FailureEnvelope.create("plan", "plan.formula"),
            ),
            (
                "k-max-value",
                ("regions", 0, "k_max"),
                0,
                FailureEnvelope.create("plan", "plan.formula"),
            ),
            (
                "hb-negative-line",
                ("regions", 0, "line"),
                -1,
                FailureEnvelope.create("plan", "plan.topology"),
            ),
            (
                "hb-empty-phase",
                ("regions", 0, "phase"),
                "",
                FailureEnvelope.create("plan", "plan.topology"),
            ),
        )
        for case_name, path, value, expected in cases:
            with self.subTest(case=case_name):
                record = copy.deepcopy(canonical)
                target = record
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                with self.assertRaises(FixtureError) as raised:
                    _decode_logical_plan(record, plan_id=f"raw.{case_name}")
                self.assertEqual(raised.exception.envelope, expected)

    def test_raw_typed_plan_binding_types_have_exact_envelopes(self) -> None:
        logical = build_singleton(d_model=2)
        typed = _typed(logical, "float32")
        canonical = typed.canonical_dict()
        cases = (
            ("binding-scalar", ("binding",), 1),
            ("dtype-roles-scalar", ("binding", "dtype_roles"), 1),
            (
                "dtype-value-object",
                ("binding", "dtype_roles", "hidden"),
                {},
            ),
            (
                "dtype-value-array",
                ("binding", "dtype_roles", "hidden"),
                [],
            ),
        )
        expected = FailureEnvelope.create("binding", "binding.invalid")
        for case_name, path, value in cases:
            with self.subTest(case=case_name):
                record = copy.deepcopy(canonical)
                target = record
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = value
                with self.assertRaises(FixtureError) as raised:
                    _decode_typed_plan(
                        logical,
                        record,
                        expected_logical_hash=logical.canonical_hash(),
                    )
                self.assertEqual(raised.exception.envelope, expected)

    def test_fake_negative_is_rejected_and_schema_envelope_is_strict(self) -> None:
        parts = _fixture_parts()
        parts["expected"] = _failure("input", "input.mask")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            with self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(path, **parts)
            self.assertEqual(raised.exception.code, "artifact.schema")
            self.assertFalse(path.exists())

            parts = _fixture_parts()
            artifact = save_fixture_bundle(path, **parts)
            def mutate(payload):
                payload["schema_version"] = "tide.settlegraph.fixture.v2"
                _refresh_content_hash(payload)

            _rewrite_payload(path, mutate)
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.schema")

    def test_mask_position_requested_k_and_parameter_schema_are_preflighted(self) -> None:
        cases = []
        mask = _fixture_parts()
        mask["inputs"]["lm_target_mask"][0, 2] = True
        cases.append((mask, "input.mask"))

        position = _fixture_parts()
        position["inputs"]["token_positions"][0, 1] = 2
        cases.append((position, "input.position"))

        missing_parameter = _fixture_parts()
        missing_parameter["parameters"] = dict(missing_parameter["parameters"])
        missing_parameter["parameters"].pop(next(iter(missing_parameter["parameters"])))
        cases.append((missing_parameter, "artifact.schema"))

        open_plan = build_single_layer(receiver_count=2, k=1, d_model=2)
        region = dataclasses.replace(
            open_plan.regions[0],
            k_requested={
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "requested_k",
                "minimum": 1,
                "maximum": 1,
            },
        )
        open_plan = dataclasses.replace(open_plan, regions=(region,)).validate()
        missing_k = _fixture_parts()
        typed = _typed(open_plan)
        model = SettleGraph(typed).double()
        manifest = build_eager_parameter_manifest(model)
        missing_k["typed_plan"] = typed
        missing_k["parameters"] = eager_parameter_tensors(model, manifest)
        missing_k["parameter_schema"] = manifest
        cases.append((missing_k, "input.schema"))

        with tempfile.TemporaryDirectory() as directory:
            for index, (parts, code) in enumerate(cases):
                with self.subTest(index=index, code=code), self.assertRaises(
                    FixtureError
                ) as raised:
                    save_fixture_bundle(Path(directory) / f"bad-{index}.pt", **parts)
                self.assertEqual(raised.exception.code, code)

    def test_named_negative_bundles_reach_their_declared_loader_failure(self) -> None:
        cases = []
        plan_case = _fixture_parts()
        plan_case["expected"] = _failure("plan", "plan.topology")
        cases.append(
            (
                "plan.topology.repeat-region-member",
                plan_case,
                FailureEnvelope.create("plan", "plan.topology"),
            )
        )
        mask_case = _fixture_parts()
        mask_case["expected"] = _failure("input", "input.mask")
        cases.append(
            (
                "input.mask.lm-outside-execution",
                mask_case,
                FailureEnvelope.create("input", "input.mask"),
            )
        )

        base = build_single_layer(receiver_count=2, k=2, d_model=2)
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
            )
            for node in base.nodes
        )
        plan = dataclasses.replace(
            base,
            nodes=nodes,
            regions=(
                dataclasses.replace(
                    base.regions[0], profile="BO", selector_timing="content"
                ),
            ),
        ).validate()
        typed = _typed(plan)
        model = SettleGraph(typed).double()
        manifest = build_eager_parameter_manifest(model)
        state_case = _fixture_parts()
        state_case.update(
            typed_plan=typed,
            parameters=eager_parameter_tensors(model, manifest),
            parameter_schema=manifest,
            initial_state=StateStore(
                values={
                    ("seq.a", nodes[0].node_id): torch.zeros(2, dtype=torch.float64),
                    ("seq.a", nodes[1].node_id): torch.ones(2, dtype=torch.float64),
                },
                next_position={"seq.a": 0},
            ),
            expected=_failure("state", "state.owner_alias"),
        )
        _set_gradient_contract(state_case)
        cases.append(
            (
                "state.owner-alias.nonoverlap-view",
                state_case,
                FailureEnvelope.create("state", "state.owner_alias"),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            for index, (mutation_kind, parts, expected) in enumerate(cases):
                with self.subTest(mutation_kind=mutation_kind):
                    path = Path(directory) / f"negative-{index}.pt"
                    artifact = save_negative_fixture_bundle(
                        path, mutation_kind=mutation_kind, **parts
                    )
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    self.assertEqual(
                        payload["source"]["mutation_kind"], mutation_kind
                    )
                    self.assertEqual(
                        FailureEnvelope.from_dict(payload["expected"]["error"]),
                        expected,
                    )
                    with self.assertRaises(FixtureError) as raised:
                        load_fixture_bundle(path, expected_sha256=artifact.sha256)
                    self.assertEqual(raised.exception.envelope, expected)
                    if mutation_kind == "state.owner-alias.nonoverlap-view":
                        defused = Path(directory) / "negative-state-defused.pt"
                        state_entries = payload["initial_state"]["receiver_values"]
                        for item in state_entries:
                            state_payload = item["payload"]
                            if state_payload["kind"] == "tensor":
                                state_payload["value"] = state_payload["value"].clone()
                        torch.save(payload, defused)
                        with self.assertRaises(FixtureError) as defused_error:
                            load_fixture_bundle(defused)
                        self.assertEqual(
                            defused_error.exception.code, "artifact.integrity"
                        )

                        _seal_payload(payload)
                        resigned = Path(directory) / "negative-state-resigned.pt"
                        torch.save(payload, resigned)
                        with self.assertRaises(FixtureError) as resigned_error:
                            load_fixture_bundle(resigned)
                        self.assertEqual(
                            resigned_error.exception.code, "artifact.schema"
                        )

    def test_negative_mutation_requires_a_valid_unmutated_base(self) -> None:
        parts = _fixture_parts()
        parts["expected"] = _failure("plan", "plan.topology")
        parts["inputs"]["execution_mask"][0, 2] = False
        parts["inputs"]["lm_target_mask"][0, 2] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-base.pt"
            with self.assertRaises(FixtureError) as raised:
                save_negative_fixture_bundle(
                    path,
                    mutation_kind="plan.topology.repeat-region-member",
                    **parts,
                )
            self.assertEqual(raised.exception.code, "input.mask")
            self.assertFalse(path.exists())

    def test_publish_is_no_replace_and_load_io_is_enveloped(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.pt"
            save_fixture_bundle(path, **parts)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_fixture_bundle(path, **parts)
            self.assertEqual(path.read_bytes(), original)

            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(Path(directory) / "missing.pt")
            self.assertEqual(raised.exception.code, "artifact.integrity")

    def test_publish_cleanup_cannot_replace_primary_link_failure(self) -> None:
        parts = _fixture_parts()
        primary = OSError("synthetic link failure")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.pt"
            with (
                mock.patch("tide.fixtures.os.link", side_effect=primary),
                mock.patch(
                    "tide.fixtures.os.unlink",
                    side_effect=OSError("synthetic cleanup failure"),
                ),
                self.assertRaises(OSError) as raised,
            ):
                save_fixture_bundle(destination, **parts)
            self.assertIs(raised.exception, primary)
            self.assertFalse(destination.exists())

    def test_stream_close_cannot_replace_primary_serialization_failure(self) -> None:
        parts = _fixture_parts()
        primary = OSError("synthetic serialization failure")
        real_fdopen = os.fdopen

        class HostileCloseStream:
            def __init__(self, stream):
                self.stream = stream

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def close(self):
                self.stream.close()
                raise OSError("synthetic close failure")

        def hostile_fdopen(*args, **kwargs):
            return HostileCloseStream(real_fdopen(*args, **kwargs))

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.pt"
            with (
                mock.patch("tide.fixtures.os.fdopen", side_effect=hostile_fdopen),
                mock.patch("tide.fixtures.torch.save", side_effect=primary),
                self.assertRaises(OSError) as raised,
            ):
                save_fixture_bundle(destination, **parts)
            self.assertIs(raised.exception, primary)
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_successful_publish_commits_after_staging_unlink(self) -> None:
        parts = _fixture_parts()
        events = []
        real_fsync = os.fsync
        real_unlink = os.unlink

        def observed_fsync(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                events.append("directory-fsync")
            return real_fsync(descriptor)

        def observed_unlink(path, *args, **kwargs):
            result = real_unlink(path, *args, **kwargs)
            if str(path).endswith(".tmp"):
                events.append("staging-unlink")
            return result

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "tide.fixtures.os.fsync", side_effect=observed_fsync
        ), mock.patch(
            "tide.fixtures.os.unlink", side_effect=observed_unlink
        ):
            destination = Path(directory) / "fixture.pt"
            save_fixture_bundle(destination, **parts)
            self.assertEqual(
                events[-2:], ["staging-unlink", "directory-fsync"]
            )
            self.assertEqual(tuple(Path(directory).iterdir()), (destination,))

    def test_publish_does_not_delete_a_recreated_staging_name(self) -> None:
        parts = _fixture_parts()
        recreated = None
        real_unlink = os.unlink

        def unlink_and_recreate(path, *args, **kwargs):
            nonlocal recreated
            result = real_unlink(path, *args, **kwargs)
            if str(path).endswith(".tmp") and recreated is None:
                recreated = Path(path)
                recreated.write_bytes(b"foreign")
            return result

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.pt"
            with mock.patch(
                "tide.fixtures.os.unlink", side_effect=unlink_and_recreate
            ):
                save_fixture_bundle(destination, **parts)
            self.assertTrue(destination.exists())
            self.assertIsNotNone(recreated)
            assert recreated is not None
            self.assertEqual(recreated.read_bytes(), b"foreign")
            recreated.unlink()

    def test_precommit_staging_unlink_failure_rolls_back_destination(self) -> None:
        parts = _fixture_parts()
        real_unlink = os.unlink
        failed = False

        def fail_first_staging_unlink(path, *args, **kwargs):
            nonlocal failed
            if not failed and str(path).endswith(".tmp"):
                failed = True
                raise OSError("synthetic staging unlink failure")
            return real_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.pt"
            with mock.patch(
                "tide.fixtures.os.unlink",
                side_effect=fail_first_staging_unlink,
            ), self.assertRaisesRegex(OSError, "synthetic staging unlink"):
                save_fixture_bundle(destination, **parts)
            self.assertTrue(failed)
            self.assertFalse(destination.exists())
            self.assertEqual(tuple(Path(directory).iterdir()), ())

    def test_post_link_failures_remove_the_authenticated_destination(self) -> None:
        parts = _fixture_parts()
        real_fsync = os.fsync
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fsync-failure.pt"
            failed_directory_fsync = False

            def fail_first_directory_fsync(descriptor):
                nonlocal failed_directory_fsync
                if (
                    not failed_directory_fsync
                    and stat.S_ISDIR(os.fstat(descriptor).st_mode)
                ):
                    failed_directory_fsync = True
                    raise OSError("synthetic directory fsync failure")
                return real_fsync(descriptor)

            with mock.patch(
                "tide.fixtures.os.fsync",
                side_effect=fail_first_directory_fsync,
            ), self.assertRaisesRegex(OSError, "synthetic directory fsync"):
                save_fixture_bundle(destination, **parts)
            self.assertTrue(failed_directory_fsync)
            self.assertFalse(destination.exists())

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "lstat-failure.pt"
            real_lstat = os.lstat
            failed_destination_lstat = False

            def fail_first_destination_lstat(path, *args, **kwargs):
                nonlocal failed_destination_lstat
                if (
                    not failed_destination_lstat
                    and os.fspath(path) == os.fspath(destination)
                ):
                    failed_destination_lstat = True
                    raise OSError("synthetic destination lstat failure")
                return real_lstat(path, *args, **kwargs)

            with mock.patch(
                "tide.fixtures.os.lstat",
                side_effect=fail_first_destination_lstat,
            ), self.assertRaisesRegex(OSError, "synthetic destination lstat"):
                save_fixture_bundle(destination, **parts)
            self.assertTrue(failed_destination_lstat)
            self.assertFalse(destination.exists())

    def test_concurrent_publish_has_one_winner_and_no_partial_file(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "fixture.pt"
            real_link = os.link
            barrier = threading.Barrier(2)

            def synchronized_link(source, target, *args, **kwargs):
                barrier.wait(timeout=10)
                return real_link(source, target, *args, **kwargs)

            def publish():
                try:
                    return save_fixture_bundle(destination, **parts)
                except BaseException as exc:
                    return exc

            with mock.patch(
                "tide.fixtures.os.link", side_effect=synchronized_link
            ), concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = tuple(pool.map(lambda _: publish(), range(2)))

            self.assertEqual(
                sum(not isinstance(item, BaseException) for item in outcomes), 1
            )
            self.assertEqual(
                sum(isinstance(item, FileExistsError) for item in outcomes), 1
            )
            loaded = load_fixture_bundle(destination)
            self.assertEqual(loaded.fixture_id, parts["fixture_id"])
            self.assertEqual(tuple(root.iterdir()), (destination,))

    def test_staged_path_symlink_swap_cannot_overwrite_or_publish(self) -> None:
        parts = _fixture_parts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "fixture.pt"
            victim = root / "victim.bin"
            victim.write_bytes(b"do-not-overwrite")
            real_mkstemp = tempfile.mkstemp

            def swapped_mkstemp(*args, **kwargs):
                descriptor, temporary = real_mkstemp(*args, **kwargs)
                Path(temporary).unlink()
                Path(temporary).symlink_to(victim)
                return descriptor, temporary

            with mock.patch(
                "tide.fixtures.tempfile.mkstemp",
                side_effect=swapped_mkstemp,
            ), self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(destination, **parts)
            self.assertEqual(raised.exception.code, "artifact.integrity")
            self.assertEqual(victim.read_bytes(), b"do-not-overwrite")
            self.assertFalse(destination.exists())

    def test_open_descriptor_rejects_a_staging_swap_during_link(self) -> None:
        parts = _fixture_parts()
        real_mkstemp = tempfile.mkstemp
        real_link = os.link
        temporary = None

        def observed_mkstemp(*args, **kwargs):
            nonlocal temporary
            descriptor, name = real_mkstemp(*args, **kwargs)
            temporary = Path(name)
            return descriptor, name

        def swap_name_then_link(source, target, *args, **kwargs):
            assert temporary is not None
            temporary.unlink()
            temporary.write_bytes(b"foreign")
            return real_link(source, target, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fixture.pt"
            with (
                mock.patch(
                    "tide.fixtures.tempfile.mkstemp",
                    side_effect=observed_mkstemp,
                ),
                mock.patch(
                    "tide.fixtures.os.link",
                    side_effect=swap_name_then_link,
                ),
                self.assertRaises(OSError),
            ):
                save_fixture_bundle(destination, **parts)
            self.assertFalse(destination.exists())
            self.assertIsNotNone(temporary)
            assert temporary is not None
            self.assertEqual(temporary.read_bytes(), b"foreign")
            temporary.unlink()

    def test_noncontiguous_hidden_layout_is_preserved_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for storage_offset in (0, 1):
                with self.subTest(storage_offset=storage_offset):
                    parts = _fixture_parts()
                    backing = torch.arange(24, dtype=torch.float64).reshape(
                        2, 3, 4
                    )
                    hidden = backing[
                        ..., storage_offset : storage_offset + 4 : 2
                    ]
                    self.assertEqual(hidden.shape, (2, 3, 2))
                    self.assertFalse(hidden.is_contiguous())
                    parts["inputs"]["hidden"] = hidden
                    parts["expected"]["golden"]["output"] = hidden
                    parts["gradient"]["output_cotangent"] = (
                        hidden.mul(0).add(0.25)
                    )
                    path = Path(directory) / f"noncontiguous-{storage_offset}.pt"
                    save_fixture_bundle(path, **parts)
                    loaded = load_fixture_bundle(path)
                    loaded_hidden = loaded.inputs["hidden"]
                    self.assertFalse(loaded_hidden.is_contiguous())
                    self.assertEqual(loaded_hidden.stride(), hidden.stride())
                    self.assertEqual(
                        loaded_hidden.storage_offset(), storage_offset
                    )
                    torch.testing.assert_close(
                        loaded_hidden, hidden, atol=0, rtol=0
                    )
                    entry = next(
                        item
                        for item in loaded.tensor_manifest
                        if item["path"] == "/inputs/hidden"
                    )
                    self.assertEqual(entry["stride"], list(hidden.stride()))
                    self.assertEqual(
                        entry["storage_offset"], storage_offset
                    )

    def test_overlapping_tensor_layouts_are_rejected_before_publish(self) -> None:
        layouts = (
            torch.zeros((1, 3, 2), dtype=torch.float64).expand(2, 3, 2),
            torch.as_strided(
                torch.arange(8, dtype=torch.float64),
                size=(2, 3, 2),
                stride=(1, 1, 1),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, hidden in enumerate(layouts):
                with self.subTest(index=index):
                    parts = _fixture_parts()
                    parts["inputs"]["hidden"] = hidden
                    path = Path(directory) / f"overlap-{index}.pt"
                    with self.assertRaises(FixtureError) as raised:
                        save_fixture_bundle(path, **parts)
                    self.assertEqual(raised.exception.code, "artifact.schema")
                    self.assertFalse(path.exists())

    def test_unresigned_schema_tamper_fails_integrity_before_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            save_fixture_bundle(path, **_fixture_parts())
            _rewrite_payload(
                path,
                lambda payload: payload.__setitem__(
                    "schema_version", "tide.settlegraph.fixture.v2"
                ),
            )
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.integrity")

    def test_parameter_storage_alias_and_gradient_schema_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aliased_parts = _fixture_parts()
            aliased_parameters = dict(aliased_parts["parameters"])
            keys = sorted(aliased_parameters)
            backing = torch.arange(4, dtype=torch.float64)
            aliased_parameters[keys[0]] = backing[:2]
            aliased_parameters[keys[1]] = backing[2:]
            aliased_parts["parameters"] = aliased_parameters
            direct_path = Path(directory) / "direct-alias.pt"
            with self.assertRaises(FixtureError) as direct_error:
                save_fixture_bundle(direct_path, **aliased_parts)
            self.assertEqual(direct_error.exception.code, "artifact.schema")
            self.assertFalse(direct_path.exists())

            path = Path(directory) / "alias.pt"
            save_fixture_bundle(path, **_fixture_parts())

            def alias_parameters(payload):
                keys = sorted(payload["parameters"])
                first = payload["parameters"][keys[0]]
                second = payload["parameters"][keys[1]]
                backing = torch.empty(
                    first.numel() + second.numel(), dtype=first.dtype
                )
                backing[: first.numel()].copy_(first.reshape(-1))
                backing[first.numel() :].copy_(second.reshape(-1))
                payload["parameters"][keys[0]] = backing[: first.numel()].view(
                    first.shape
                )
                payload["parameters"][keys[1]] = backing[first.numel() :].view(
                    second.shape
                )
                _seal_payload(payload)

            _rewrite_payload(path, alias_parameters)
            with self.assertRaises(FixtureError) as raised:
                load_fixture_bundle(path)
            self.assertEqual(raised.exception.code, "artifact.schema")

    def test_raw_enum_like_fields_reject_unhashable_values_as_schema(self) -> None:
        mutations = {
            "expected-outcome": lambda parts: parts.__setitem__(
                "expected", {"outcome": []}
            ),
            "path-assertion": lambda parts: parts["gradient"][
                "path_assertions"
            ].__setitem__("inputs.hidden", []),
            "routing-classification": lambda parts: parts.__setitem__(
                "routing_classification", []
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    parts = _fixture_parts()
                    mutate(parts)
                    path = Path(directory) / f"{name}.pt"
                    with self.assertRaises(FixtureError) as raised:
                        save_fixture_bundle(path, **parts)
                    self.assertEqual(raised.exception.code, "artifact.schema")
                    self.assertFalse(path.exists())

            malformed = _fixture_parts()
            malformed["gradient"] = dict(malformed["gradient"])
            malformed["gradient"].pop("path_assertions")
            with self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(
                    Path(directory) / "bad-gradient.pt", **malformed
                )
            self.assertEqual(raised.exception.code, "artifact.schema")

            unsupported = _fixture_parts()
            unsupported["learnable_initial_state"] = {
                "unowned": torch.zeros(2, dtype=torch.float64)
            }
            with self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(
                    Path(directory) / "learnable-state.pt", **unsupported
                )
            self.assertEqual(raised.exception.code, "artifact.schema")

    def test_state_alias_and_absolute_private_paths_are_rejected_before_publish(self) -> None:
        base = build_single_layer(receiver_count=2, k=2, d_model=2)
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
            )
            for node in base.nodes
        )
        region = dataclasses.replace(
            base.regions[0], profile="BO", selector_timing="content"
        )
        plan = dataclasses.replace(base, nodes=nodes, regions=(region,)).validate()
        typed = _typed(plan)
        model = SettleGraph(typed).double()
        manifest = build_eager_parameter_manifest(model)
        backing = torch.zeros(4, dtype=torch.float64)
        parts = _fixture_parts()
        parts.update(
            typed_plan=typed,
            parameters=eager_parameter_tensors(model, manifest),
            parameter_schema=manifest,
            initial_state=StateStore(
                values={
                    ("seq.a", nodes[0].node_id): backing[:2],
                    ("seq.a", nodes[1].node_id): backing[2:],
                },
                next_position={"seq.a": 0},
            ),
        )
        _set_gradient_contract(parts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sentinel.pt"
            path.write_bytes(b"preserve-me")
            with self.assertRaises(FixtureError) as raised:
                save_fixture_bundle(path, **parts)
            self.assertEqual(raised.exception.code, "state.owner_alias")
            self.assertEqual(path.read_bytes(), b"preserve-me")

            for name, source in (
                (
                    "identifier",
                    {
                        "kind": "manual",
                        "identifier": "/private/data/corpus.json",
                        "command": "python generate.py",
                    },
                ),
                (
                    "command",
                    {
                        "kind": "manual",
                        "identifier": "portable-source",
                        "command": "python /private/data/generate.py",
                    },
                ),
            ):
                absolute = _fixture_parts()
                absolute["source"] = source
                with self.subTest(source_field=name), self.assertRaises(
                    FixtureError
                ) as raised:
                    save_fixture_bundle(
                        Path(directory) / f"private-{name}.pt", **absolute
                    )
                self.assertEqual(raised.exception.code, "artifact.schema")

            for name, identifier, command in (
                (
                    "windows-identifier",
                    r"C:\private\data\corpus.json",
                    "python generate.py",
                ),
                (
                    "unc-identifier",
                    r"\\server\private\corpus.json",
                    "python generate.py",
                ),
                (
                    "windows-command",
                    "portable-source",
                    r"python C:\private\data\generate.py",
                ),
                (
                    "unc-command",
                    "portable-source",
                    r"python \\server\private\generate.py",
                ),
                (
                    "windows-rooted-command",
                    "portable-source",
                    r"python \private\data\generate.py",
                ),
                (
                    "redirected-posix-command",
                    "portable-source",
                    "python generate.py >/private/data/output.json",
                ),
                (
                    "redirected-windows-command",
                    "portable-source",
                    r"python generate.py 2>C:\private\data\output.txt",
                ),
            ):
                absolute = _fixture_parts()
                absolute["source"] = {
                    "kind": "manual",
                    "identifier": identifier,
                    "command": command,
                }
                with self.subTest(source_field=name), self.assertRaises(
                    FixtureError
                ) as raised:
                    save_fixture_bundle(
                        Path(directory) / f"private-{name}.pt", **absolute
                    )
                self.assertEqual(raised.exception.code, "artifact.schema")


if __name__ == "__main__":
    unittest.main()
