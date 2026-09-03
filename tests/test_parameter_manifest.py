from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tide.builders import (
    build_diamond,
    build_multi_entry_terminal,
    build_single_layer,
    build_singleton,
)
from tide.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from tide.engine import SettleGraph, StateStore
from tide.failures import FailureEnvelope, failure_envelope_from_exception
from tide.parameter_manifest import (
    EAGER_EXECUTOR_ID,
    EAGER_PARAMETER_BINDING_VERSION,
    PARAMETER_SCHEMA_CANONICALIZER_ID,
    PARAMETER_SCHEMA_VERSION,
    LogicalParameterKey,
    ParameterManifestEntry,
    ParameterManifestError,
    ParameterSchemaManifest,
    build_eager_parameter_manifest,
    build_parameter_schema_manifest,
    export_eager_parameter_tensors,
    load_eager_parameter_tensors,
    logical_parameter_tensor_key,
)
from tide.plan import bind_dtypes


def _replace_plan(plan, *, nodes=None, regions=None, **changes):
    return dataclasses.replace(
        plan,
        nodes=tuple(plan.nodes if nodes is None else nodes),
        regions=tuple(plan.regions if regions is None else regions),
        **changes,
    ).validate()


def _stateful_plan(kind: str, *, score_type: str = "mlp"):
    plan = build_singleton(d_model=3)
    node = plan.nodes[0]
    if kind == "ema":
        state_shape = (2,)
        update = {
            "type": "ema",
            "formula_id": "state.ema.v1",
            "state_dim": 2,
            "decay": 0.25,
            "state_shape": [2],
        }
        selector_read = {
            "type": "content_state_linear",
            "formula_id": "TEST-READ-PROJ-V1",
            "out_dim": 2,
            "output_shape": [2],
        }
        ffn_formula = "read.ffn.ema.v1"
        compute = {
            "type": "affine_residual",
            "formula_id": "TEST-NODE-AFFINE-V1",
            "bias": True,
            "output_shape": [3],
        }
    elif kind == "gdn":
        state_shape = (2, 4)
        update = {
            "type": "gdn",
            "formula_id": "state.gdn.v1",
            "key_dim": 2,
            "value_dim": 4,
            "norm_eps": 1e-12,
            "state_shape": [2, 4],
        }
        selector_read = {
            "type": "content_state_linear",
            "formula_id": "TEST-READ-PROJ-V1",
            "out_dim": 2,
            "output_shape": [2],
        }
        ffn_formula = "read.ffn.gdn.v1"
        compute = {
            "type": "double_residual_swiglu",
            "formula_id": "TEST-NODE-SWIGLU-V1",
            "hidden_dim": 5,
            "bias": True,
            "output_shape": [3],
        }
    elif kind == "attention_window":
        state_shape = (3, 2, 4)
        update = {
            "type": "attention_window",
            "formula_id": "state.attention-window.v1",
            "key_dim": 2,
            "value_dim": 4,
            "window": 3,
            "norm_eps": 1e-12,
            "state_shape": [3, 2, 4],
        }
        selector_read = {
            "type": "content_state_summary_linear",
            "formula_id": "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
            "out_dim": 2,
            "output_shape": [2],
        }
        ffn_formula = "read.ffn.attention-window.v1"
        compute = {
            "type": "double_residual_swiglu",
            "formula_id": "TEST-NODE-SWIGLU-V1",
            "hidden_dim": 5,
            "bias": True,
            "output_shape": [3],
        }
    else:  # pragma: no cover - helper guard
        raise AssertionError(kind)
    node = dataclasses.replace(
        node,
        state_shape=state_shape,
        state_owner=node.node_id,
        update=update,
        selector_read_shape=(2,),
        selector_read=selector_read,
        ffn_read={
            "type": "state_default",
            "formula_id": ffn_formula,
            "output_shape": [3],
        },
        node_compute=compute,
    )
    if score_type == "linear":
        score = {
            "type": "linear",
            "formula_id": "TEST-SCORE-LINEAR-V1",
            "bias": True,
        }
    else:
        score = {
            "type": "mlp",
            "formula_id": "TEST-SCORE-MLP-V1",
            "hidden_dim": 4,
            "bias": True,
        }
    region = dataclasses.replace(
        plan.regions[0],
        profile="BO",
        selector_timing="post",
        score=score,
    )
    return _replace_plan(plan, nodes=(node,), regions=(region,))


class ParameterSchemaManifestTests(unittest.TestCase):
    def test_manifest_rejects_non_eager_executor_identity(self):
        manifest = build_eager_parameter_manifest(
            SettleGraph(build_singleton(d_model=2))
        )
        with self.assertRaisesRegex(
            ParameterManifestError, "unsupported eager executor"
        ):
            dataclasses.replace(manifest, executor_id="another.executor.v1")
        self.assertEqual(manifest.executor_id, EAGER_EXECUTOR_ID)

    def test_state_dict_hook_cannot_substitute_same_shape_parameter(self):
        graph = SettleGraph(build_singleton(d_model=2))
        names = [name for name, _ in graph.named_parameters()]
        self.assertEqual(len(names), 2)

        def substitute_parameter(module, state_dict, prefix, local_metadata):
            del module, prefix, local_metadata
            state_dict[names[0]] = state_dict[names[1]]

        graph.register_state_dict_post_hook(substitute_parameter)
        with self.assertRaisesRegex(
            ParameterManifestError, "not bound to the named parameter storage"
        ):
            build_eager_parameter_manifest(graph)

    def test_stateless_parameterless_operations_add_no_phantom_entries(self):
        plan = build_singleton()
        manifest = build_eager_parameter_manifest(SettleGraph(plan))
        self.assertEqual(
            manifest.canonical_dict(),
            build_parameter_schema_manifest(plan).canonical_dict(),
        )
        self.assertEqual(PARAMETER_SCHEMA_VERSION, "tide.parameter-schema.v1")
        self.assertEqual(
            PARAMETER_SCHEMA_CANONICALIZER_ID,
            "tide-parameter-schema-json-v1",
        )
        self.assertEqual(
            EAGER_PARAMETER_BINDING_VERSION,
            "tide.eager-parameter-binding.v1",
        )
        self.assertEqual(
            [(entry.logical_key.field, entry.logical_key.parameter_role) for entry in manifest.entries],
            [("ffn_norm", "w"), ("input_norm", "w")],
        )
        parameter_formulas = {entry.formula_id for entry in manifest.entries}
        self.assertTrue(
            {
                "update.none.v1",
                "read.selector.content.v1",
                "read.ffn.zero.v1",
                "node.identity.v1",
                "emit.hard.v1",
                "score.fixed-by-node.v1",
                "agg.mean.v1",
            }.isdisjoint(parameter_formulas)
        )

        # Byte-level regression golden for a stable standard builder.
        self.assertEqual(
            manifest.canonical_hash(),
            "4a1118136967525c4e50ac1f97efb24d53d72cb5728b5929029da6219afffbd0",
        )
        self.assertEqual(
            manifest.eager_binding_hash(),
            "ca71278c6c287dd0c0a37298ad496cbbfe10cb294d0656abb4f17f9e5eb98c68",
        )

    def test_logical_hash_excludes_eager_locator_and_entry_input_order(self):
        original = build_eager_parameter_manifest(SettleGraph(build_singleton()))
        changed_entries = list(reversed(original.entries))
        changed_entries[0] = dataclasses.replace(
            changed_entries[0], state_dict_locator="another.executor.parameter.0"
        )
        changed = ParameterSchemaManifest(
            logical_plan_hash=original.logical_plan_hash,
            entries=tuple(changed_entries),
        )
        self.assertEqual(original.canonical_bytes(), changed.canonical_bytes())
        self.assertEqual(original.canonical_hash(), changed.canonical_hash())
        self.assertNotEqual(
            original.eager_binding_hash(), changed.eager_binding_hash()
        )
        self.assertNotIn(b"receivers.", original.canonical_bytes())
        self.assertIn(b"receivers.", original.eager_binding_bytes())

    def test_ema_read_projection_affine_and_linear_score_roles(self):
        manifest = build_eager_parameter_manifest(
            SettleGraph(_stateful_plan("ema", score_type="linear"))
        )
        by_field_role = {
            (entry.logical_key.field, entry.logical_key.parameter_role): entry
            for entry in manifest.entries
        }
        expected_shapes = {
            ("update", "W_obs"): (2, 3),
            ("update", "b_obs"): (2,),
            ("selector_read", "W_sel"): (2, 5),
            ("selector_read", "b_sel"): (2,),
            ("ffn_read", "W_out"): (3, 2),
            ("node_compute", "W_node"): (3, 3),
            ("node_compute", "b_node"): (3,),
            ("score", "w_score"): (2,),
            ("score", "b_score"): (),
        }
        for key, shape in expected_shapes.items():
            with self.subTest(key=key):
                self.assertEqual(by_field_role[key].shape, shape)
                self.assertEqual(by_field_role[key].dtype_role, "parameter")
                self.assertIsNone(by_field_role[key].parameter_group)
        self.assertEqual(
            by_field_role[("score", "w_score")].state_dict_shape, (1, 2)
        )
        self.assertEqual(
            by_field_role[("score", "b_score")].state_dict_shape, (1,)
        )
        self.assertEqual(
            by_field_role[("score", "w_score")].logical_to_state_dict,
            "reshape-row-major",
        )
        self.assertEqual(
            by_field_role[("score", "b_score")].logical_to_state_dict,
            "reshape-row-major",
        )

    def test_gdn_and_mlp_roles_cover_every_formula_parameter(self):
        model = SettleGraph(_stateful_plan("gdn"))
        manifest = build_eager_parameter_manifest(model)
        roles = {
            (entry.logical_key.field, entry.logical_key.parameter_role, entry.shape)
            for entry in manifest.entries
        }
        expected = {
            ("update", "W_k", (2, 3)),
            ("update", "W_nu", (4, 3)),
            ("update", "w_eta", (3,)),
            ("update", "b_eta", ()),
            ("update", "w_gamma", (3,)),
            ("update", "b_gamma", ()),
            ("update", "beta", ()),
            ("ffn_read", "W_q", (2, 3)),
            ("ffn_read", "W_out", (3, 4)),
            ("node_compute", "W_g", (5, 3)),
            ("node_compute", "b_g", (5,)),
            ("node_compute", "W_u", (5, 3)),
            ("node_compute", "b_u", (5,)),
            ("node_compute", "W_o", (3, 5)),
            ("node_compute", "b_o", (3,)),
            ("score", "W_1", (4, 2)),
            ("score", "b_1", (4,)),
            ("score", "w_2", (4,)),
            ("score", "b_2", ()),
        }
        self.assertTrue(expected.issubset(roles))

        by_role = {
            (entry.logical_key.field, entry.logical_key.parameter_role): entry
            for entry in manifest.entries
        }
        expected_eager_shapes = {
            ("update", "w_eta"): (1, 3),
            ("update", "b_eta"): (1,),
            ("update", "w_gamma"): (1, 3),
            ("update", "b_gamma"): (1,),
            ("score", "w_2"): (1, 4),
            ("score", "b_2"): (1,),
        }
        for key, state_dict_shape in expected_eager_shapes.items():
            with self.subTest(binding=key):
                self.assertEqual(by_role[key].state_dict_shape, state_dict_shape)
                self.assertEqual(
                    by_role[key].logical_to_state_dict, "reshape-row-major"
                )
        exported = export_eager_parameter_tensors(model, manifest)
        for entry in manifest.entries:
            tensor_key = logical_parameter_tensor_key(
                entry.logical_key.canonical_dict()
            )
            with self.subTest(exported=entry.logical_key.parameter_role):
                self.assertEqual(tuple(exported[tensor_key].shape), entry.shape)

    def test_attention_roles_and_formula_ids_are_not_conflated(self):
        manifest = build_eager_parameter_manifest(
            SettleGraph(_stateful_plan("attention_window"))
        )
        selected = {
            (entry.logical_key.field, entry.logical_key.parameter_role): (
                entry.formula_id,
                entry.shape,
            )
            for entry in manifest.entries
            if entry.logical_key.field in {"update", "ffn_read"}
        }
        self.assertEqual(
            selected,
            {
                ("update", "W_k"): ("state.attention-window.v1", (2, 3)),
                ("update", "W_nu"): ("state.attention-window.v1", (4, 3)),
                ("ffn_read", "W_q"): ("read.ffn.attention-window.v1", (2, 3)),
                ("ffn_read", "W_out"): ("read.ffn.attention-window.v1", (3, 4)),
            },
        )

    def test_edge_and_terminal_parameter_owners_use_stable_semantic_ids(self):
        for aggregate_type, formula_id, roles in (
            ("edge_softmax", "TEST-AGG-EDGE-SOFTMAX-V1", {"eta"}),
            (
                "edge_linear_mean",
                "TEST-AGG-EDGE-AFFINE-MEAN-V1",
                {"W", "b"},
            ),
        ):
            base = build_diamond(d_model=2, branch_k=2)
            nodes = tuple(
                dataclasses.replace(
                    node,
                    aggregate={
                        "type": aggregate_type,
                        "formula_id": formula_id,
                        **({"bias": True} if aggregate_type == "edge_linear_mean" else {}),
                        "output_shape": [2],
                    },
                )
                if node.node_id == "node.out"
                else node
                for node in base.nodes
            )
            manifest = build_eager_parameter_manifest(
                SettleGraph(_replace_plan(base, nodes=nodes))
            )
            aggregate_entries = [
                entry
                for entry in manifest.entries
                if entry.logical_key.field == "aggregate"
            ]
            with self.subTest(aggregate_type=aggregate_type):
                self.assertEqual(
                    {entry.logical_key.parameter_role for entry in aggregate_entries},
                    roles,
                )
                self.assertEqual(
                    {entry.logical_key.edge_id for entry in aggregate_entries},
                    {"edge.a-out", "edge.b-out"},
                )
                self.assertTrue(
                    all(entry.logical_key.node_id == "node.out" for entry in aggregate_entries)
                )
                self.assertTrue(all(entry.formula_id == formula_id for entry in aggregate_entries))

        output_plan = _replace_plan(
            build_multi_entry_terminal(d_model=2),
            output_aggregate={
                "type": "node_softmax",
                "formula_id": "TEST-AGG-TERMINAL-SOFTMAX-V1",
                "output_shape": [2],
            },
        )
        output_entries = [
            entry
            for entry in build_eager_parameter_manifest(SettleGraph(output_plan)).entries
            if entry.logical_key.field == "output_aggregate"
        ]
        self.assertEqual(
            {entry.logical_key.terminal_node_id for entry in output_entries},
            {"node.out.a", "node.out.b"},
        )
        self.assertTrue(
            all(entry.logical_key.parameter_role == "eta_out" for entry in output_entries)
        )

    def test_no_parameter_score_read_and_emit_variants_remain_empty(self):
        base = build_single_layer(receiver_count=2, k=1, d_model=2)
        score_configs = (
            {"type": "constant", "formula_id": "score.constant.v1", "value": 2.0},
            {
                "type": "fixed",
                "formula_id": "TEST-SCORE-CONST-V1",
                "values_by_node": {node.node_id: 1.0 for node in base.nodes},
            },
            {"type": "read_sum", "formula_id": "score.read-sum.v1"},
        )
        for score in score_configs:
            region = dataclasses.replace(base.regions[0], score=score)
            manifest = build_eager_parameter_manifest(
                SettleGraph(_replace_plan(base, regions=(region,)))
            )
            with self.subTest(score=score["type"]):
                self.assertFalse(
                    any(entry.logical_key.field == "score" for entry in manifest.entries)
                )

        singleton = build_singleton(d_model=2)
        node = dataclasses.replace(
            singleton.nodes[0],
            selector_read_shape=(1,),
            selector_read={
                "type": "content_norm",
                "formula_id": "read.selector.content-rms.v1",
                "out_dim": 1,
                "output_shape": [1],
            },
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 0.5,
                "output_shape": [2],
            },
        )
        manifest = build_eager_parameter_manifest(
            SettleGraph(_replace_plan(singleton, nodes=(node,)))
        )
        self.assertEqual(
            {entry.logical_key.field for entry in manifest.entries},
            {"input_norm", "ffn_norm"},
        )

    def test_typed_dtype_shape_extra_parameter_and_alias_are_rejected(self):
        plan = build_singleton(d_model=2)
        typed = bind_dtypes(
            plan,
            hidden="float64",
            parameter="float64",
            state="float64",
            readout="float64",
        )
        with self.assertRaisesRegex(ParameterManifestError, "expected torch.float64"):
            build_eager_parameter_manifest(SettleGraph(typed))
        build_eager_parameter_manifest(SettleGraph(typed).double())

        wrong_shape = SettleGraph(plan)
        wrong_shape.receiver("node.0").input_norm.weight = torch.nn.Parameter(
            torch.ones(3)
        )
        with self.assertRaisesRegex(ParameterManifestError, "expects state_dict shape"):
            build_eager_parameter_manifest(wrong_shape)

        extra = SettleGraph(plan)
        extra.register_parameter("undeclared_parameter", torch.nn.Parameter(torch.zeros(())))
        with self.assertRaisesRegex(ParameterManifestError, "unexpected=.*undeclared_parameter"):
            build_eager_parameter_manifest(extra)

        aliased = SettleGraph(plan)
        aliased.receiver("node.0").ffn_norm.weight = aliased.receiver(
            "node.0"
        ).input_norm.weight
        with self.assertRaisesRegex(ParameterManifestError, "parameter aliases"):
            build_eager_parameter_manifest(aliased)

        internally_overlapping = SettleGraph(plan)
        internally_overlapping.receiver(
            "node.0"
        ).input_norm.weight = torch.nn.Parameter(torch.ones(1).expand(2))
        with self.assertRaisesRegex(
            ParameterManifestError, "internally overlapping"
        ):
            build_eager_parameter_manifest(internally_overlapping)

        graph = SettleGraph(plan)
        manifest = build_eager_parameter_manifest(graph)
        tampered_entries = list(manifest.entries)
        tampered_entries[0] = dataclasses.replace(
            tampered_entries[0], formula_id="another.formula.v1"
        )
        tampered = ParameterSchemaManifest(
            logical_plan_hash=manifest.logical_plan_hash,
            entries=tuple(tampered_entries),
        )
        with self.assertRaisesRegex(
            ParameterManifestError, "do not match the Plan formula schema"
        ):
            tampered.validate_model(graph)

    def test_eager_binding_is_exact_for_each_logical_key(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        entries = list(manifest.entries)

        swapped = list(entries)
        swapped[0] = dataclasses.replace(
            swapped[0], state_dict_locator=entries[1].state_dict_locator
        )
        swapped[1] = dataclasses.replace(
            swapped[1], state_dict_locator=entries[0].state_dict_locator
        )
        wrong_storage_shape = list(entries)
        wrong_storage_shape[0] = dataclasses.replace(
            wrong_storage_shape[0],
            state_dict_shape=(1, 2),
            logical_to_state_dict="reshape-row-major",
        )
        wrong_transform = list(entries)
        wrong_transform[0] = dataclasses.replace(
            wrong_transform[0], logical_to_state_dict="reshape-row-major"
        )

        for name, changed in (
            ("locator", swapped),
            ("storage-shape", wrong_storage_shape),
            ("transform", wrong_transform),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ParameterManifestError, "eager locator/shape/transform bindings"
            ):
                ParameterSchemaManifest(
                    logical_plan_hash=manifest.logical_plan_hash,
                    entries=tuple(changed),
                ).validate_model(graph)

        with self.assertRaisesRegex(
            ParameterManifestError, "logical_to_state_dict"
        ):
            dataclasses.replace(entries[0], logical_to_state_dict=[])

    def test_distinct_parameter_views_cannot_share_backing_storage(self):
        plan = build_singleton(d_model=2)
        graph = SettleGraph(plan)
        backing = torch.arange(4.0)
        graph.receiver("node.0").input_norm.weight = torch.nn.Parameter(
            backing[:2]
        )
        graph.receiver("node.0").ffn_norm.weight = torch.nn.Parameter(
            backing[2:]
        )
        self.assertIsNot(
            graph.receiver("node.0").input_norm.weight,
            graph.receiver("node.0").ffn_norm.weight,
        )
        with self.assertRaisesRegex(
            ParameterManifestError, "parameter backing-storage aliases"
        ):
            build_eager_parameter_manifest(graph)

    def test_overlapping_external_parameter_storages_are_rejected(self):
        plan = build_singleton(d_model=2)
        graph = SettleGraph(plan)
        backing = bytearray(24)
        first = torch.frombuffer(backing, dtype=torch.float32, count=2, offset=0)
        second = torch.frombuffer(backing, dtype=torch.float32, count=2, offset=4)
        self.assertNotEqual(
            first.untyped_storage()._cdata, second.untyped_storage()._cdata
        )
        graph.receiver("node.0").input_norm.weight = torch.nn.Parameter(first)
        graph.receiver("node.0").ffn_norm.weight = torch.nn.Parameter(second)
        with self.assertRaisesRegex(
            ParameterManifestError, "parameter backing-storage aliases"
        ):
            build_eager_parameter_manifest(graph)

    def test_unclosed_parameter_group_and_invalid_owner_are_rejected(self):
        key = LogicalParameterKey(
            field="input_norm",
            parameter_role="w",
            region_id="region.0",
            node_id="node.0",
        )
        with self.assertRaisesRegex(ParameterManifestError, "parameter groups are not closed"):
            ParameterManifestEntry(
                logical_key=key,
                formula_id="norm.rms.v1",
                shape=(2,),
                dtype_role="parameter",
                parameter_group="shared.0",
                state_dict_locator="receiver.norm.weight",
            )
        with self.assertRaisesRegex(ParameterManifestError, "invalid owner fields"):
            LogicalParameterKey(
                field="aggregate",
                parameter_role="eta",
                node_id="node.0",
                region_id="region.0",
            )

    def test_zero_ffn_read_does_not_materialize_unowned_query_parameters(self):
        for kind, forbidden in (
            ("gdn", {"gdn_query.weight", "gdn_out.weight"}),
            ("attention_window", {"attn_query.weight", "attn_out.weight"}),
        ):
            plan = _stateful_plan(kind)
            node = dataclasses.replace(
                plan.nodes[0],
                ffn_read={
                    "type": "zero",
                    "formula_id": "read.ffn.zero.v1",
                    "output_shape": [3],
                },
            )
            plan = _replace_plan(plan, nodes=(node,))
            graph = SettleGraph(plan)
            manifest = build_eager_parameter_manifest(graph)
            locators = {entry.state_dict_locator for entry in manifest.entries}
            with self.subTest(kind=kind):
                self.assertTrue(
                    all(not any(name.endswith(suffix) for suffix in forbidden) for name in locators)
                )
                self.assertFalse(
                    any(entry.logical_key.field == "ffn_read" for entry in manifest.entries)
                )

    def test_zero_ffn_read_rejects_legacy_ghost_checkpoint_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind, stem in (("gdn", "gdn"), ("attention_window", "attn")):
                plan = _stateful_plan(kind)
                node = dataclasses.replace(
                    plan.nodes[0],
                    ffn_read={
                        "type": "zero",
                        "formula_id": "read.ffn.zero.v1",
                        "output_shape": [3],
                    },
                )
                typed = bind_dtypes(
                    _replace_plan(plan, nodes=(node,)),
                    hidden="float32",
                    parameter="float32",
                    state="float32",
                    readout="float32",
                )
                source = SettleGraph(typed)
                valid_path = Path(directory) / f"{kind}-valid.pt"
                legacy_path = Path(directory) / f"{kind}-legacy-ghost.pt"
                save_checkpoint(
                    valid_path,
                    model=source,
                    typed_plan=typed,
                    state=StateStore(),
                )
                payload = torch.load(
                    valid_path, map_location="cpu", weights_only=True
                )
                key_locator = next(
                    key
                    for key in payload["model_state"]
                    if key.endswith(f"{stem}_key.weight")
                )
                prefix = key_locator.removesuffix(f"{stem}_key.weight")
                payload["model_state"][f"{prefix}{stem}_query.weight"] = (
                    torch.zeros_like(payload["model_state"][key_locator])
                )
                payload["model_state"][f"{prefix}{stem}_out.weight"] = torch.zeros(
                    (3, 4), dtype=torch.float32
                )
                torch.save(payload, legacy_path)

                target = SettleGraph(typed)
                before = {
                    key: value.detach().clone()
                    for key, value in target.state_dict().items()
                }
                with self.subTest(kind=kind), self.assertRaisesRegex(
                    CheckpointError, "model key mismatch.*extra="
                ) as raised:
                    load_checkpoint(
                        legacy_path,
                        model=target,
                        typed_plan=typed,
                        mode="resume",
                    )
                self.assertEqual(
                    failure_envelope_from_exception(
                        raised.exception, codes="checkpoint.compatibility"
                    ),
                    FailureEnvelope.create(
                        "checkpoint", "checkpoint.compatibility"
                    ),
                )
                for key, value in target.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_portable_logical_tensor_mapping_loads_without_eager_keys(self):
        typed = bind_dtypes(
            build_singleton(d_model=2),
            hidden="float64",
            parameter="float64",
            state="float64",
            readout="float64",
        )
        source = SettleGraph(typed).double()
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), 1):
                parameter.fill_(index / 7.0)
        manifest = build_eager_parameter_manifest(source)
        portable = {
            key: value.detach().cpu().clone()
            for key, value in export_eager_parameter_tensors(source, manifest).items()
        }
        self.assertTrue(all(key.startswith("tide.logical-parameter.v1:") for key in portable))
        self.assertTrue(all("receivers." not in key for key in portable))

        target = SettleGraph(typed).double()
        target_manifest = build_eager_parameter_manifest(target)
        load_eager_parameter_tensors(target, target_manifest, portable)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name], atol=0, rtol=0)

        before = {
            name: value.detach().clone() for name, value in target.state_dict().items()
        }
        malformed = dict(portable)
        first_key = next(iter(malformed))
        malformed[first_key] = malformed[first_key].float()
        with self.assertRaises(ParameterManifestError):
            load_eager_parameter_tensors(target, target_manifest, malformed)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_logical_tensor_load_stages_all_sources_before_target_copy(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        by_field = {
            entry.logical_key.field: logical_parameter_tensor_key(
                entry.logical_key.canonical_dict()
            )
            for entry in manifest.entries
        }
        input_key = by_field["input_norm"]
        ffn_key = by_field["ffn_norm"]
        with torch.no_grad():
            graph.receiver("node.0").input_norm.weight.copy_(
                torch.tensor([1.0, 2.0])
            )
            graph.receiver("node.0").ffn_norm.weight.copy_(
                torch.tensor([3.0, 4.0])
            )
        live_sources = dict(export_eager_parameter_tensors(graph, manifest))
        live_sources[input_key], live_sources[ffn_key] = (
            live_sources[ffn_key],
            live_sources[input_key],
        )

        load_eager_parameter_tensors(graph, manifest, live_sources)

        torch.testing.assert_close(
            graph.receiver("node.0").input_norm.weight,
            torch.tensor([3.0, 4.0]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            graph.receiver("node.0").ffn_norm.weight,
            torch.tensor([1.0, 2.0]),
            atol=0,
            rtol=0,
        )

    def test_logical_tensor_load_rejects_undeclared_source_storage_alias(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = dict(export_eager_parameter_tensors(graph, manifest))
        keys = sorted(portable)
        self.assertEqual(len(keys), 2)
        backing = torch.arange(4.0)
        portable[keys[0]] = backing[:2]
        portable[keys[1]] = backing[2:]

        with self.assertRaisesRegex(
            ParameterManifestError,
            "backing storage must be independent",
        ):
            load_eager_parameter_tensors(graph, manifest, portable)

    def test_logical_tensor_load_rejects_internal_overlap_and_nonstring_keys(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = {
            key: value.detach().clone()
            for key, value in export_eager_parameter_tensors(
                graph, manifest
            ).items()
        }
        first_key = next(iter(portable))
        overlapping = dict(portable)
        overlapping[first_key] = torch.ones(1).expand(2)
        with self.assertRaisesRegex(
            ParameterManifestError, "internally overlapping"
        ):
            load_eager_parameter_tensors(graph, manifest, overlapping)

        mixed_keys = dict(portable)
        mixed_keys[1] = mixed_keys.pop(first_key)
        with self.assertRaisesRegex(
            ParameterManifestError, "keys must all be strings"
        ):
            load_eager_parameter_tensors(graph, manifest, mixed_keys)

    def test_device_staging_finishes_before_the_first_target_copy(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = {
            key: value.detach().clone().add_(1.0)
            for key, value in export_eager_parameter_tensors(
                graph, manifest
            ).items()
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in graph.named_parameters()
        }
        versions = {
            name: parameter._version
            for name, parameter in graph.named_parameters()
        }
        original_to = torch.Tensor.to
        to_calls = 0

        def fail_second_to(source, *args, **kwargs):
            nonlocal to_calls
            to_calls += 1
            if to_calls == 2:
                raise RuntimeError("synthetic second-stage failure")
            return original_to(source, *args, **kwargs)

        with mock.patch.object(
            torch.Tensor, "to", new=fail_second_to
        ), self.assertRaisesRegex(
            ParameterManifestError, "cannot stage logical parameter"
        ):
            load_eager_parameter_tensors(graph, manifest, portable)

        self.assertEqual(to_calls, 2)
        for name, parameter in graph.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            self.assertEqual(parameter._version, versions[name])

    def test_failed_copy_restores_values_but_invalidates_active_graph(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = {
            key: value.detach().clone().add_(1.0)
            for key, value in export_eager_parameter_tensors(
                graph, manifest
            ).items()
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in graph.named_parameters()
        }
        versions = {
            name: parameter._version
            for name, parameter in graph.named_parameters()
        }
        objective = sum(
            parameter.square().sum() for parameter in graph.parameters()
        )
        original_copy = torch.Tensor.copy_
        copy_calls = 0

        def fail_second_copy(target, source, *args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 2:
                raise RuntimeError("synthetic second-copy failure")
            return original_copy(target, source, *args, **kwargs)

        with mock.patch.object(
            torch.Tensor, "copy_", new=fail_second_copy
        ), self.assertRaisesRegex(RuntimeError, "synthetic second-copy failure"):
            load_eager_parameter_tensors(graph, manifest, portable)

        for name, parameter in graph.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            self.assertGreater(parameter._version, versions[name])
        with self.assertRaises(RuntimeError):
            objective.backward()

    def test_rollback_failure_is_attached_without_hiding_primary_copy_error(self):
        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = {
            key: value.detach().clone().add_(1.0)
            for key, value in export_eager_parameter_tensors(
                graph, manifest
            ).items()
        }
        original_copy = torch.Tensor.copy_
        copy_calls = 0

        def fail_primary_and_first_rollback(target, source, *args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 2:
                raise RuntimeError("synthetic primary copy failure")
            if copy_calls == 3:
                raise RuntimeError("synthetic rollback failure")
            return original_copy(target, source, *args, **kwargs)

        with mock.patch.object(
            torch.Tensor,
            "copy_",
            new=fail_primary_and_first_rollback,
        ), self.assertRaisesRegex(
            RuntimeError, "synthetic primary copy failure"
        ) as raised:
            load_eager_parameter_tensors(graph, manifest, portable)

        rollback_failures = getattr(
            raised.exception, "tide_parameter_rollback_failures", ()
        )
        self.assertEqual(len(rollback_failures), 1)
        self.assertIsInstance(rollback_failures[0][1], RuntimeError)

    def test_hostile_primary_exception_survives_rollback_diagnostics(self):
        class LockedError(RuntimeError):
            def add_note(self, note):
                raise RuntimeError("notes locked")

            def __setattr__(self, name, value):
                raise AttributeError("attributes locked")

        graph = SettleGraph(build_singleton(d_model=2))
        manifest = build_eager_parameter_manifest(graph)
        portable = {
            key: value.detach().clone().add_(1.0)
            for key, value in export_eager_parameter_tensors(
                graph, manifest
            ).items()
        }
        primary = LockedError("hostile primary copy failure")
        original_copy = torch.Tensor.copy_
        copy_calls = 0

        def fail_primary_and_first_rollback(target, source, *args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            if copy_calls == 2:
                raise primary
            if copy_calls == 3:
                raise RuntimeError("synthetic rollback failure")
            return original_copy(target, source, *args, **kwargs)

        with mock.patch.object(
            torch.Tensor,
            "copy_",
            new=fail_primary_and_first_rollback,
        ), self.assertRaises(LockedError) as raised:
            load_eager_parameter_tensors(graph, manifest, portable)

        self.assertIs(raised.exception, primary)
        self.assertGreaterEqual(copy_calls, 4)


if __name__ == "__main__":
    unittest.main()
