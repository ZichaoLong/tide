from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from typing import Optional, Tuple

from tide.builders import (
    build_chain,
    build_diamond,
    build_hb_lattice,
    build_mixed_regions,
    build_multi_entry_terminal,
    build_single_layer,
    build_singleton,
    build_small_hb,
    build_unequal_path,
)
from tide.plan import (
    ConcreteBinding,
    EdgeSpec,
    NodeSpec,
    PLAN_CANONICALIZER_ID,
    Plan,
    PlanValidationError,
    RegionSpec,
    bind_dtypes,
    validate_stable_id,
)


class PlanTestCase(unittest.TestCase):
    def assert_invalid(
        self,
        plan: Plan,
        text: str,
        *,
        failure_codes: Optional[Tuple[str, ...]] = None,
    ) -> PlanValidationError:
        with self.assertRaises(PlanValidationError) as raised:
            plan.validate()
        self.assertIn(text, str(raised.exception))
        self.assertTrue(raised.exception.failure_codes)
        if failure_codes is not None:
            self.assertEqual(raised.exception.failure_codes, failure_codes)
        return raised.exception

    def test_standard_builders_are_expanded_and_valid(self) -> None:
        fixtures = [
            (build_singleton(), (1, 0, 1)),
            (build_single_layer(receiver_count=8, k=2), (8, 0, 1)),
            (build_chain(length=4), (4, 3, 4)),
            (build_diamond(), (4, 4, 3)),
            (build_unequal_path(), (5, 5, 4)),
            (build_multi_entry_terminal(), (4, 4, 2)),
            (build_mixed_regions(), (5, 6, 4)),
            (build_small_hb(), (14, 22, 8)),
        ]
        for plan, expected_counts in fixtures:
            with self.subTest(plan=plan.plan_id):
                self.assertIs(plan.validate(), plan)
                self.assertEqual(
                    (len(plan.nodes), len(plan.edges), len(plan.regions)),
                    expected_counts,
                )
                self.assertEqual(len(plan.canonical_hash()), 64)
                self.assertTrue(plan.entry_node_ids)
                self.assertTrue(plan.terminal_node_ids)

    def test_declaration_reordering_does_not_change_hash(self) -> None:
        original = build_small_hb()
        reordered_regions = tuple(
            dataclasses.replace(
                region,
                node_ids=tuple(reversed(region.node_ids)),
                control_dependencies=tuple(
                    reversed(region.control_dependencies)
                ),
            )
            for region in reversed(original.regions)
        )
        reordered = dataclasses.replace(
            original,
            nodes=tuple(reversed(original.nodes)),
            edges=tuple(reversed(original.edges)),
            regions=reordered_regions,
            entry_node_ids=tuple(reversed(original.entry_node_ids)),
            terminal_node_ids=tuple(reversed(original.terminal_node_ids)),
        )
        self.assertEqual(original.canonical_json(), reordered.canonical_json())
        self.assertEqual(original.canonical_hash(), reordered.canonical_hash())

    def test_semantic_change_changes_hash(self) -> None:
        top1 = build_diamond(branch_k=1)
        top2 = build_diamond(branch_k=2)
        self.assertNotEqual(top1.canonical_hash(), top2.canonical_hash())

        changed_node = dataclasses.replace(
            top1.nodes[0],
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 1.0,
                "output_shape": [top1.d_model],
            },
        )
        changed = dataclasses.replace(
            top1,
            nodes=(changed_node,) + top1.nodes[1:],
        )
        self.assertNotEqual(top1.canonical_hash(), changed.canonical_hash())

    def test_normalization_config_is_part_of_logical_and_typed_hashes(self) -> None:
        original = build_singleton()
        changed_node = dataclasses.replace(
            original.nodes[0],
            input_norm={
                "type": "rmsnorm",
                "formula_id": "norm.rms.v1",
                "eps": 1e-5,
            },
        )
        changed = dataclasses.replace(
            original, nodes=(changed_node,)
        ).validate()
        self.assertNotEqual(original.logical_hash(), changed.logical_hash())

        changed_ffn_node = dataclasses.replace(
            original.nodes[0],
            ffn_norm={
                "type": "rmsnorm",
                "formula_id": "norm.rms.v1",
                "eps": 2e-5,
            },
        )
        changed_ffn = dataclasses.replace(
            original, nodes=(changed_ffn_node,)
        ).validate()
        self.assertNotEqual(original.logical_hash(), changed_ffn.logical_hash())

        original_typed = bind_dtypes(
            original,
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        changed_typed = bind_dtypes(
            changed,
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        self.assertNotEqual(original_typed.typed_hash(), changed_typed.typed_hash())

    def test_hb_multi_digit_coordinates_remain_unique(self) -> None:
        plan = build_hb_lattice(
            branch_factor=11,
            expansion_depth=2,
            platform_lines=0,
            region_size=3,
        )
        self.assertIs(plan.validate(), plan)
        node_ids = [node.node_id for node in plan.nodes]
        edge_ids = [edge.edge_id for edge in plan.edges]
        endpoints = [(edge.source, edge.target) for edge in plan.edges]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(len(endpoints), len(set(endpoints)))
        self.assertTrue(any(".0010" in node_id for node_id in node_ids))

    def test_renaming_and_builder_provenance_do_not_change_logical_hash(self) -> None:
        plan = build_chain()
        renamed = dataclasses.replace(
            plan,
            plan_id="human-readable-alias",
            builder={"name": "another-builder", "version": "99"},
        )
        self.assertEqual(plan.canonical_hash(), renamed.canonical_hash())
        self.assertNotEqual(plan.to_record_dict()["plan_id"], renamed.to_record_dict()["plan_id"])

    def test_logical_and_concrete_dtype_bindings_have_distinct_hashes(self) -> None:
        logical = build_singleton()
        typed = bind_dtypes(
            logical,
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        typed_again = bind_dtypes(
            build_singleton(),
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        self.assertEqual(logical.canonical_hash(), typed.logical_hash())
        self.assertNotEqual(logical.canonical_hash(), typed.canonical_hash())
        self.assertEqual(typed.canonical_hash(), typed_again.canonical_hash())
        self.assertEqual(typed.binding.dtype_roles["hidden"], "float32")
        with self.assertRaises(PlanValidationError) as raised:
            ConcreteBinding(
                {
                    "hidden": "runtime",
                    "parameter": "float32",
                    "state": "float32",
                    "readout": "float32",
                }
            ).validate_for(logical)
        self.assertEqual(
            raised.exception.failure_codes, ("binding.invalid",)
        )

    def test_canonical_json_is_json_safe_and_configs_are_copied(self) -> None:
        mutable_emit = {
            "type": "custom",
            "formula": "g = h",
            "nested": {"values": [1, 2]},
        }
        node = dataclasses.replace(build_singleton().nodes[0], emit=mutable_emit)
        plan = dataclasses.replace(build_singleton(), nodes=(node,)).validate()
        before = plan.canonical_hash()
        mutable_emit["nested"]["values"].append(3)
        self.assertEqual(before, plan.canonical_hash())
        decoded = json.loads(plan.canonical_json())
        self.assertEqual(decoded["nodes"][0]["emit"]["nested"]["values"], [1, 2])
        with self.assertRaises(TypeError):
            plan.nodes[0].emit["type"] = "softp"  # type: ignore[index]

    def test_canonicalizer_ascii_bytes_and_hash_are_stable(self) -> None:
        self.assertEqual(PLAN_CANONICALIZER_ID, "tide-plan-json-v1")
        logical = build_singleton()
        expected_logical_hash = (
            "2b3ad87e06cb01076b1da1450f41e775a"
            "732740a322aa5d63de422297db67757"
        )
        self.assertEqual(logical.canonical_hash(), expected_logical_hash)
        self.assertEqual(
            hashlib.sha256(logical.canonical_bytes()).hexdigest(),
            expected_logical_hash,
        )

        typed = bind_dtypes(
            logical,
            hidden="float32",
            parameter="float32",
            state="float32",
            readout="float32",
        )
        expected_typed_bytes = (
            b'{"binding":{"dtype_roles":{"hidden":"float32",'
            b'"parameter":"float32","readout":"float32",'
            b'"state":"float32"}},"logical_plan_hash":'
            b'"2b3ad87e06cb01076b1da1450f41e775a732740a322aa5d63de422297db67757",'
            b'"schema_version":"1"}'
        )
        self.assertEqual(typed.canonical_bytes(), expected_typed_bytes)
        self.assertEqual(
            typed.canonical_hash(),
            "73387a3891a20d05edffe5a13ed67dfa96ad44afc5ca37c0dd6075f36ff3432c",
        )

    def test_plan_schema_version_is_an_exact_gate(self) -> None:
        incompatible = dataclasses.replace(build_singleton(), schema_version="2")
        self.assert_invalid(incompatible, "schema_version must be exactly '1'")
        with self.assertRaises(PlanValidationError):
            incompatible.canonical_hash()

    def test_unicode_scalar_order_and_utf8_bytes_are_canonical(self) -> None:
        base = build_single_layer(receiver_count=3, k=1, d_model=2)
        scalar_order = (
            "node.\N{LATIN SMALL LETTER E WITH ACUTE}",
            "node.\ue000",
            "node.\U00010000",
        )
        declared_order = tuple(reversed(scalar_order))
        renamed_nodes = tuple(
            dataclasses.replace(node, node_id=node_id)
            for node, node_id in zip(base.nodes, declared_order)
        )
        score_values = {
            declared_order[0]: 1e-6,
            declared_order[1]: -0.0,
            declared_order[2]: 1,
        }
        renamed_region = dataclasses.replace(
            base.regions[0],
            node_ids=declared_order,
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": score_values,
            },
        )
        plan = dataclasses.replace(
            base,
            nodes=tuple(reversed(renamed_nodes)),
            regions=(renamed_region,),
            entry_node_ids=declared_order,
            terminal_node_ids=declared_order,
        ).validate()

        self.assertEqual(tuple(node.node_id for node in plan.nodes), scalar_order)
        self.assertEqual(plan.regions[0].node_ids, scalar_order)
        canonical_bytes = plan.canonical_bytes()
        self.assertIn(scalar_order[0].encode("utf-8"), canonical_bytes)
        self.assertIn(scalar_order[2].encode("utf-8"), canonical_bytes)
        self.assertNotIn(b"\\u00e9", canonical_bytes)
        self.assertNotIn(b"\\ud800", canonical_bytes.lower())
        self.assertIn(b'"eps":1e-06', canonical_bytes)
        self.assertIn(
            f'"{scalar_order[1]}":0.0'.encode("utf-8"), canonical_bytes
        )
        self.assertIn(b'"node.\xc3\xa9":1.0', canonical_bytes)
        decoded = json.loads(canonical_bytes)
        self.assertEqual(
            tuple(decoded["regions"][0]["score"]["values_by_node"]),
            scalar_order,
        )

        reordered_score = dict(reversed(tuple(score_values.items())))
        reordered_region = dataclasses.replace(
            plan.regions[0],
            node_ids=tuple(reversed(plan.regions[0].node_ids)),
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": reordered_score,
            },
        )
        reordered = dataclasses.replace(
            plan,
            nodes=tuple(reversed(plan.nodes)),
            regions=(reordered_region,),
            entry_node_ids=tuple(reversed(plan.entry_node_ids)),
            terminal_node_ids=tuple(reversed(plan.terminal_node_ids)),
        )
        self.assertEqual(reordered.canonical_bytes(), canonical_bytes)
        self.assertEqual(
            plan.canonical_hash(),
            "50633109caeeb1122ae6a0e9bdf8ccd0e5fa3a0660d995c8e83515dca6455904",
        )

    def test_stable_ids_reject_non_nfc_whitespace_and_surrogates(self) -> None:
        self.assertEqual(
            validate_stable_id("sequence.\U0001f600", kind="sequence"),
            "sequence.\U0001f600",
        )
        invalid_cases = (
            ("node.e\N{COMBINING ACUTE ACCENT}", "NFC"),
            ("\u2000node", "surrounding Unicode whitespace"),
            ("\x1cnode", "surrounding Unicode whitespace"),
            ("node.\ud800", "Unicode scalar values"),
            ("node.\ud83d\ude00", "Unicode scalar values"),
        )
        for value, message in invalid_cases:
            with self.subTest(value=ascii(value)):
                with self.assertRaisesRegex(ValueError, message):
                    validate_stable_id(value, kind="node")

        base = build_singleton()
        invalid_node = dataclasses.replace(
            base.nodes[0], node_id="node.\ud800"
        )
        invalid_plan = dataclasses.replace(base, nodes=(invalid_node,))
        with self.assertRaisesRegex(PlanValidationError, "Unicode scalar values"):
            invalid_plan.canonical_hash()

        hb = build_small_hb()
        invalid_phase = dataclasses.replace(
            hb.regions[0], phase="phase.\udfff"
        )
        invalid_hb = dataclasses.replace(
            hb, regions=(invalid_phase,) + hb.regions[1:]
        )
        with self.assertRaisesRegex(PlanValidationError, "Unicode scalar values"):
            invalid_hb.canonical_hash()

    def test_operation_config_json_errors_are_schema_gated(self) -> None:
        base = build_singleton()
        base_node = base.nodes[0]
        invalid_configs = (
            {"type": "custom", "formula": "x + \ud800"},
            {
                "type": "custom",
                "formula": "x",
                "nested": {"bad.\udfff": 1},
            },
            {
                "type": "custom",
                "formula": "x",
                "nested": ["valid", "bad.\ud800"],
            },
            {"type": "custom", "formula": "x", "value": float("nan")},
            {"type": "custom", "formula": "x", "value": float("inf")},
            {"type": "custom", "formula": "x", "value": (1, 2)},
        )
        for config in invalid_configs:
            with self.subTest(config=ascii(config)):
                node = dataclasses.replace(base_node, emit=config)
                self.assert_invalid(
                    dataclasses.replace(base, nodes=(node,)),
                    "JSON-safe object",
                    failure_codes=("plan.schema",),
                )

    def test_known_formula_configs_are_canonicalized_before_hashing(self) -> None:
        base = build_singleton(d_model=3)

        omitted_identity = dataclasses.replace(
            base.nodes[0],
            node_compute={
                "type": "IDENTITY",
                "formula_id": "node.identity.v1",
            },
        )
        explicit_identity = dataclasses.replace(
            base.nodes[0],
            node_compute={
                "type": "identity",
                "formula_id": "node.identity.v1",
                "output_shape": [3],
            },
        )
        omitted_plan = dataclasses.replace(
            base, nodes=(omitted_identity,)
        ).validate()
        explicit_plan = dataclasses.replace(
            base, nodes=(explicit_identity,)
        ).validate()
        self.assertEqual(omitted_plan.canonical_json(), explicit_plan.canonical_json())
        self.assertEqual(omitted_plan.canonical_hash(), explicit_plan.canonical_hash())

        omitted_affine = dataclasses.replace(
            base.nodes[0],
            node_compute={
                "type": "affine-residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
            },
        )
        explicit_affine = dataclasses.replace(
            base.nodes[0],
            node_compute={
                "type": "affine_residual",
                "formula_id": "TEST-NODE-AFFINE-V1",
                "bias": True,
                "output_shape": [3],
            },
        )
        omitted_plan = dataclasses.replace(
            base, nodes=(omitted_affine,)
        ).validate()
        explicit_plan = dataclasses.replace(
            base, nodes=(explicit_affine,)
        ).validate()
        self.assertEqual(omitted_plan.canonical_hash(), explicit_plan.canonical_hash())

        omitted_hst = dataclasses.replace(
            base.nodes[0],
            emit={"type": "hst", "formula_id": "emit.hst.v1"},
        )
        explicit_hst = dataclasses.replace(
            base.nodes[0],
            emit={
                "type": "hst",
                "formula_id": "emit.hst.v1",
                "zeta": 1,
                "output_shape": [3],
            },
        )
        self.assertEqual(
            dataclasses.replace(base, nodes=(omitted_hst,)).canonical_hash(),
            dataclasses.replace(base, nodes=(explicit_hst,)).canonical_hash(),
        )

        omitted_score = dataclasses.replace(
            base.regions[0],
            score={"type": "constant", "formula_id": "score.constant.v1"},
        )
        explicit_score = dataclasses.replace(
            base.regions[0],
            score={
                "type": "constant",
                "formula_id": "score.constant.v1",
                "value": -0.0,
            },
        )
        self.assertEqual(
            dataclasses.replace(base, regions=(omitted_score,)).canonical_hash(),
            dataclasses.replace(base, regions=(explicit_score,)).canonical_hash(),
        )

        integer_fixed_score = dataclasses.replace(
            base.regions[0],
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {base.nodes[0].node_id: 0},
            },
        )
        negative_zero_fixed_score = dataclasses.replace(
            base.regions[0],
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {base.nodes[0].node_id: -0.0},
            },
        )
        self.assertEqual(
            dataclasses.replace(
                base, regions=(integer_fixed_score,)
            ).canonical_hash(),
            dataclasses.replace(
                base, regions=(negative_zero_fixed_score,)
            ).canonical_hash(),
        )

    def test_formula_integer_normalization_is_lossless(self) -> None:
        base = build_singleton(d_model=2)

        def constant_score_plan(value: object) -> Plan:
            region = dataclasses.replace(
                base.regions[0],
                score={
                    "type": "constant",
                    "formula_id": "score.constant.v1",
                    "value": value,
                },
            )
            return dataclasses.replace(base, regions=(region,))

        safe_boundary = (1 << 53) - 1
        safe_integer = constant_score_plan(safe_boundary).validate()
        safe_float = constant_score_plan(float(safe_boundary)).validate()
        self.assertEqual(safe_integer.canonical_hash(), safe_float.canonical_hash())
        self.assertEqual(
            safe_integer.regions[0].score["value"], float(safe_boundary)
        )

        for unsafe_integer in (1 << 53, (1 << 53) + 1, -(1 << 53)):
            with self.subTest(unsafe_integer=unsafe_integer):
                invalid = constant_score_plan(unsafe_integer)
                self.assert_invalid(invalid, "JSON-safe range")
                with self.assertRaises(PlanValidationError):
                    invalid.canonical_hash()

        unsafe_fixed_region = dataclasses.replace(
            base.regions[0],
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {base.nodes[0].node_id: (1 << 53) + 1},
            },
        )
        self.assert_invalid(
            dataclasses.replace(base, regions=(unsafe_fixed_region,)),
            "JSON-safe range",
        )

    def test_known_formula_schema_rejects_unknown_and_fixed_switches(self) -> None:
        base = build_singleton(d_model=2)
        node_cases = (
            (
                "aggregate",
                {
                    "type": "edge_linear_mean",
                    "formula_id": "TEST-AGG-EDGE-AFFINE-MEAN-V1",
                    "bias": False,
                },
                "requires bias=True",
            ),
            (
                "node_compute",
                {
                    "type": "affine_residual",
                    "formula_id": "TEST-NODE-AFFINE-V1",
                    "bias": False,
                },
                "requires bias=True",
            ),
            (
                "node_compute",
                {
                    "type": "double_residual_swiglu",
                    "formula_id": "TEST-NODE-SWIGLU-V1",
                    "bias": False,
                },
                "requires bias=True",
            ),
            (
                "emit",
                {
                    "type": "hard",
                    "formula_id": "emit.hard.v1",
                    "ignored_key": "must-not-be-ignored",
                },
                "unknown config keys",
            ),
        )
        for field, config, message in node_cases:
            with self.subTest(field=field, config=config):
                node = dataclasses.replace(base.nodes[0], **{field: config})
                self.assert_invalid(
                    dataclasses.replace(base, nodes=(node,)), message
                )

        score_cases = (
            ("linear", {"bias": False}, "requires bias=True"),
            ("mlp", {"bias": False}, "requires bias=True"),
            (
                "linear",
                {"shared_parameters": True},
                "requires shared_parameters=False",
            ),
            ("mlp", {"context_dim": 1}, "requires context_dim=0"),
        )
        for score_type, overrides, message in score_cases:
            with self.subTest(score_type=score_type, overrides=overrides):
                formula = (
                    "TEST-SCORE-LINEAR-V1"
                    if score_type == "linear"
                    else "TEST-SCORE-MLP-V1"
                )
                region = dataclasses.replace(
                    base.regions[0],
                    score={
                        "type": score_type,
                        "formula_id": formula,
                        **overrides,
                    },
                )
                self.assert_invalid(
                    dataclasses.replace(base, regions=(region,)), message
                )

        stateful_node = dataclasses.replace(
            base.nodes[0],
            state_shape=(2,),
            state_owner=base.nodes[0].node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "learnable_decay": True,
            },
        )
        stateful_region = dataclasses.replace(base.regions[0], profile="BO")
        self.assert_invalid(
            dataclasses.replace(
                base, nodes=(stateful_node,), regions=(stateful_region,)
            ),
            "requires learnable_decay=False",
        )

    def test_reference_selector_and_update_shapes_are_cross_checked(self) -> None:
        base = build_singleton(d_model=2)
        multidimensional_read = dataclasses.replace(
            base.nodes[0],
            selector_read_shape=(1, 1),
            selector_read={
                "type": "content_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
            },
        )
        self.assert_invalid(
            dataclasses.replace(base, nodes=(multidimensional_read,)),
            "requires selector_read_shape (1,)",
        )

        mismatched_ema = dataclasses.replace(
            base.nodes[0],
            state_shape=(2,),
            state_owner=base.nodes[0].node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 3,
            },
        )
        stateful_region = dataclasses.replace(base.regions[0], profile="BO")
        self.assert_invalid(
            dataclasses.replace(
                base, nodes=(mismatched_ema,), regions=(stateful_region,)
            ),
            "dimensions require state_shape (3,)",
        )

        valid_ema = dataclasses.replace(
            base.nodes[0],
            state_shape=(2,),
            state_owner=base.nodes[0].node_id,
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_dim": 2,
            },
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
            },
        )
        self.assert_invalid(
            dataclasses.replace(
                base, nodes=(valid_ema,), regions=(stateful_region,)
            ),
            "incompatible with 'content' selector timing",
        )

        content_only_projection = dataclasses.replace(
            valid_ema,
            selector_read={
                "type": "content_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
            },
        )
        pre_region = dataclasses.replace(
            stateful_region, selector_timing="pre"
        )
        self.assert_invalid(
            dataclasses.replace(
                base,
                nodes=(content_only_projection,),
                regions=(pre_region,),
            ),
            "incompatible with 'pre' selector timing",
        )

        for read_type, formula_id in (
            ("content", "read.selector.content.v1"),
            ("content_norm", "read.selector.content-rms.v1"),
        ):
            with self.subTest(read_type=read_type, timing="pre"):
                read_node = dataclasses.replace(
                    valid_ema,
                    selector_read_shape=(2,) if read_type == "content" else (1,),
                    selector_read={"type": read_type, "formula_id": formula_id},
                )
                self.assert_invalid(
                    dataclasses.replace(
                        base, nodes=(read_node,), regions=(pre_region,)
                    ),
                    "incompatible with 'pre' selector timing",
                )

        state_summary_at_content = dataclasses.replace(
            base.nodes[0],
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_summary_linear",
                "formula_id": "TEST-READ-STATE-RMS-SUMMARY-PROJ-V1",
                "out_dim": 1,
            },
        )
        self.assert_invalid(
            dataclasses.replace(base, nodes=(state_summary_at_content,)),
            "incompatible with 'content' selector timing",
        )

    def test_state_default_read_formula_must_match_update_family(self) -> None:
        base = build_singleton(d_model=2)
        mismatched = dataclasses.replace(
            base.nodes[0],
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.ema.v1",
            },
        )
        self.assert_invalid(
            dataclasses.replace(base, nodes=(mismatched,)),
            "requires formula_id 'read.ffn.zero.v1'",
        )

        matched = dataclasses.replace(
            base.nodes[0],
            ffn_read={
                "type": "state_default",
                "formula_id": "read.ffn.zero.v1",
            },
        )
        matched_plan = dataclasses.replace(base, nodes=(matched,))
        self.assertIs(matched_plan.validate(), matched_plan)


    def test_helpers_use_stable_id_and_edge_order(self) -> None:
        plan = build_diamond()
        self.assertEqual(plan.node_by_id("node.root").node_id, "node.root")
        self.assertEqual(
            plan.region_by_id("region.branches").node_ids,
            ("node.branch.a", "node.branch.b"),
        )
        self.assertEqual(
            [edge.edge_id for edge in plan.incoming_edges["node.out"]],
            ["edge.a-out", "edge.b-out"],
        )
        self.assertEqual(
            tuple(region.region_id for region in plan.topological_regions),
            ("region.root", "region.branches", "region.out"),
        )
        with self.assertRaises(KeyError):
            plan.node_by_id("missing")

    def test_duplicate_ids_and_parallel_edges_are_rejected(self) -> None:
        plan = build_chain(length=2)
        duplicate_node = dataclasses.replace(
            plan, nodes=plan.nodes + (plan.nodes[0],)
        )
        self.assert_invalid(duplicate_node, "duplicate node IDs")

        duplicate_edge_id = dataclasses.replace(
            plan,
            edges=plan.edges
            + (
                EdgeSpec(
                    plan.edges[0].edge_id,
                    plan.edges[0].source,
                    plan.edges[0].target,
                ),
            ),
        )
        self.assert_invalid(duplicate_edge_id, "duplicate edge IDs")
        self.assert_invalid(duplicate_edge_id, "duplicate parallel receiver edges")

        parallel = dataclasses.replace(
            plan,
            edges=plan.edges
            + (EdgeSpec("edge.parallel", "node.0000", "node.0001"),),
        )
        self.assert_invalid(parallel, "duplicate parallel receiver edges")

        non_string_id = dataclasses.replace(
            plan,
            edges=(
                EdgeSpec(
                    7,  # type: ignore[arg-type]
                    plan.edges[0].source,
                    plan.edges[0].target,
                ),
            ),
        )
        self.assert_invalid(non_string_id, "edge ID must be a nonempty string")

    def test_cycle_and_region_internal_edge_are_rejected(self) -> None:
        chain = build_chain(length=3)
        cycle = dataclasses.replace(
            chain,
            edges=chain.edges
            + (EdgeSpec("edge.back", "node.0002", "node.0000"),),
        )
        self.assert_invalid(cycle, "receiver graph is cyclic")

        layer = build_single_layer(receiver_count=2)
        internal = dataclasses.replace(
            layer,
            edges=(EdgeSpec("edge.internal", "node.0000", "node.0001"),),
        )
        self.assert_invalid(internal, "connects nodes inside region")

    def test_region_partition_and_dependency_cycle_are_rejected(self) -> None:
        plan = build_chain(length=2)
        missing_member = dataclasses.replace(
            plan,
            regions=(
                dataclasses.replace(plan.regions[0], node_ids=("node.0001",)),
                plan.regions[1],
            ),
        )
        self.assert_invalid(missing_member, "nodes missing from region partition")

        cyclic_first = dataclasses.replace(
            plan.regions[0], control_dependencies=("region.0001",)
        )
        region_cycle = dataclasses.replace(
            plan, regions=(cyclic_first, plan.regions[1])
        )
        self.assert_invalid(region_cycle, "region dependency graph is cyclic")

    def test_boundaries_shapes_and_dtype_roles_are_rejected_early(self) -> None:
        plan = build_chain(length=2)
        wrong_boundary = dataclasses.replace(
            plan, entry_node_ids=("node.0001",)
        )
        self.assert_invalid(wrong_boundary, "entry_node_ids must equal")

        wrong_shape_node = dataclasses.replace(
            plan.nodes[0], hidden_shape=(plan.d_model + 1,)
        )
        wrong_shape = dataclasses.replace(
            plan, nodes=(wrong_shape_node, plan.nodes[1])
        )
        self.assert_invalid(wrong_shape, "hidden_shape must be")

        wrong_output_shape = dataclasses.replace(
            plan,
            output_aggregate={"type": "mean", "output_shape": [99]},
        )
        self.assert_invalid(wrong_output_shape, "output_aggregate.output_shape")

        layer = build_single_layer(receiver_count=2)
        mismatched_readout = dataclasses.replace(
            layer.nodes[0], selector_read_shape=(1,)
        )
        wrong_region_signature = dataclasses.replace(
            layer,
            nodes=(mismatched_readout, layer.nodes[1]),
        )
        self.assert_invalid(wrong_region_signature, "one common fixed shape")

        wrong_dtype = dataclasses.replace(
            plan,
            dtype_roles={
                "hidden": "float128",
                "parameter": "runtime",
                "state": "runtime",
                "readout": "runtime",
            },
        )
        self.assert_invalid(wrong_dtype, "must remain symbolic")

        prematurely_typed = dataclasses.replace(
            plan,
            dtype_roles={
                "hidden": "float32",
                "parameter": "runtime",
                "state": "runtime",
                "readout": "runtime",
            },
        )
        self.assert_invalid(prematurely_typed, "must remain symbolic")

    def test_profile_timing_and_state_owner_rules(self) -> None:
        plan = build_singleton()
        n_post = dataclasses.replace(
            plan,
            regions=(
                dataclasses.replace(plan.regions[0], selector_timing="post"),
            ),
        )
        self.assert_invalid(n_post, "profile N only supports content")

        stateful_node = dataclasses.replace(
            plan.nodes[0],
            state_shape=(2,),
            state_owner="shared-state",
            update={
                "type": "ema",
                "formula_id": "state.ema.v1",
                "state_shape": [2],
            },
            selector_read_shape=(1,),
            selector_read={
                "type": "content_state_linear",
                "formula_id": "TEST-READ-PROJ-V1",
                "out_dim": 1,
            },
        )
        bo_post = dataclasses.replace(
            plan.regions[0], profile="BO", selector_timing="post"
        )
        shared_state = dataclasses.replace(
            plan, nodes=(stateful_node,), regions=(bo_post,)
        )
        self.assert_invalid(shared_state, "must own its mutable state")

        owned_node = dataclasses.replace(
            stateful_node, state_owner=stateful_node.node_id
        )
        owned = dataclasses.replace(shared_state, nodes=(owned_node,))
        self.assertIs(owned.validate(), owned)

        sd_post = dataclasses.replace(
            owned, regions=(dataclasses.replace(bo_post, profile="SD"),)
        )
        self.assert_invalid(sd_post, "standard SD does not support post-update")

    def test_schema_gate_and_static_category_aggregation(self) -> None:
        base = build_chain(length=2)
        wrong_shape_node = dataclasses.replace(
            base.nodes[0], hidden_shape=(base.d_model + 1,)
        )
        topology_and_formula = dataclasses.replace(
            base,
            nodes=(wrong_shape_node, base.nodes[1]),
            terminal_node_ids=(base.entry_node_ids[0],),
        )
        self.assert_invalid(
            topology_and_formula,
            "hidden_shape must be",
            failure_codes=("plan.formula", "plan.topology"),
        )

        malformed_node = dataclasses.replace(
            wrong_shape_node, node_id=" invalid.node"
        )
        schema_gated = dataclasses.replace(
            topology_and_formula,
            nodes=(malformed_node, base.nodes[1]),
        )
        gated_error = self.assert_invalid(
            schema_gated,
            "surrounding Unicode whitespace",
            failure_codes=("plan.schema",),
        )
        self.assertNotIn("hidden_shape must be", str(gated_error))
        self.assertNotIn("terminal_node_ids must equal", str(gated_error))

    def test_malformed_public_declarations_are_schema_gated(self) -> None:
        base = build_chain(length=2)
        hb = build_small_hb()
        cases = (
            (
                "d-model-object",
                lambda: dataclasses.replace(base, d_model=object()),
                "d_model must be a positive integer",
            ),
            (
                "topology-kind-scalar",
                lambda: dataclasses.replace(base, topology_kind=1),
                "topology_kind must be a string",
            ),
            (
                "output-aggregate-scalar",
                lambda: dataclasses.replace(base, output_aggregate=1),
                "output_aggregate must be a JSON object",
            ),
            (
                "output-aggregate-nonfinite",
                lambda: dataclasses.replace(
                    base,
                    output_aggregate={
                        "type": "custom",
                        "formula": "x",
                        "value": float("inf"),
                    },
                ),
                "output_aggregate must be a JSON-safe object",
            ),
            (
                "dtype-roles-nonfinite",
                lambda: dataclasses.replace(
                    base,
                    dtype_roles={
                        "hidden": float("nan"),
                        "parameter": "runtime",
                        "state": "runtime",
                        "readout": "runtime",
                    },
                ),
                "dtype_roles must be a JSON-safe object",
            ),
            (
                "builder-surrogate",
                lambda: dataclasses.replace(
                    base, builder={"source": "bad.\ud800"}
                ),
                "builder must be a JSON-safe object",
            ),
            (
                "nodes-scalar",
                lambda: dataclasses.replace(base, nodes=1),
                "nodes must be a JSON array",
            ),
            (
                "edges-string",
                lambda: dataclasses.replace(base, edges="edge.0000"),
                "edges must be a JSON array",
            ),
            (
                "regions-scalar",
                lambda: dataclasses.replace(base, regions=1),
                "regions must be a JSON array",
            ),
            (
                "entry-string",
                lambda: dataclasses.replace(
                    base, entry_node_ids=base.entry_node_ids[0]
                ),
                "entry_node_ids must be a JSON array",
            ),
            (
                "terminal-string",
                lambda: dataclasses.replace(
                    base, terminal_node_ids=base.terminal_node_ids[0]
                ),
                "terminal_node_ids must be a JSON array",
            ),
            (
                "hidden-shape-scalar",
                lambda: dataclasses.replace(
                    base,
                    nodes=(
                        dataclasses.replace(base.nodes[0], hidden_shape=1),
                        base.nodes[1],
                    ),
                ),
                "hidden_shape must be a JSON array",
            ),
            (
                "hidden-shape-element-object",
                lambda: dataclasses.replace(
                    base,
                    nodes=(
                        dataclasses.replace(
                            base.nodes[0], hidden_shape=(object(),)
                        ),
                        base.nodes[1],
                    ),
                ),
                "hidden_shape must contain only integer dimensions",
            ),
            (
                "operation-config-scalar",
                lambda: dataclasses.replace(
                    base,
                    nodes=(
                        dataclasses.replace(base.nodes[0], update=1),
                        base.nodes[1],
                    ),
                ),
                "update must be a JSON object",
            ),
            (
                "operation-type-object",
                lambda: dataclasses.replace(
                    base,
                    nodes=(
                        dataclasses.replace(
                            base.nodes[0],
                            update={
                                "type": {},
                                "formula_id": "update.none.v1",
                            },
                        ),
                        base.nodes[1],
                    ),
                ),
                "update.type must be a string",
            ),
            (
                "region-config-surrogate",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(
                            base.regions[0],
                            score={
                                "type": "custom",
                                "formula": "score.\ud800",
                            },
                        ),
                        base.regions[1],
                    ),
                ),
                "score must be a JSON-safe object",
            ),
            (
                "region-members-string",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(
                            base.regions[0], node_ids=base.nodes[0].node_id
                        ),
                        base.regions[1],
                    ),
                ),
                "node_ids must be a JSON array",
            ),
            (
                "control-dependencies-string",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        base.regions[0],
                        dataclasses.replace(
                            base.regions[1],
                            control_dependencies=base.regions[0].region_id,
                        ),
                    ),
                ),
                "control_dependencies must be a JSON array",
            ),
            (
                "profile-scalar",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(base.regions[0], profile=1),
                        base.regions[1],
                    ),
                ),
                "profile must be a string",
            ),
            (
                "selector-timing-scalar",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(base.regions[0], selector_timing=1),
                        base.regions[1],
                    ),
                ),
                "selector_timing must be a string",
            ),
            (
                "k-max-string",
                lambda: dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(base.regions[0], k_max="one"),
                        base.regions[1],
                    ),
                ),
                "k_max must be an integer",
            ),
            (
                "hb-line-string",
                lambda: dataclasses.replace(
                    hb,
                    regions=(
                        dataclasses.replace(hb.regions[0], line="zero"),
                    )
                    + hb.regions[1:],
                ),
                "line must be null or an integer",
            ),
            (
                "hb-phase-scalar",
                lambda: dataclasses.replace(
                    hb,
                    regions=(
                        dataclasses.replace(hb.regions[0], phase=1),
                    )
                    + hb.regions[1:],
                ),
                "phase must be null or a string",
            ),
        )
        for case_name, construct, message in cases:
            with self.subTest(case=case_name):
                plan = construct()
                self.assert_invalid(
                    plan,
                    message,
                    failure_codes=("plan.schema",),
                )

    def test_type_correct_invalid_values_retain_semantic_categories(self) -> None:
        base = build_singleton()
        hb = build_small_hb()
        cases = (
            (
                "registered-formula-dispatch",
                dataclasses.replace(
                    base,
                    output_aggregate={
                        "type": "node_softmax",
                        "formula_id": "agg.mean.v1",
                    },
                ),
                "plan.formula",
            ),
            (
                "profile-value",
                dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(base.regions[0], profile="invalid"),
                    ),
                ),
                "plan.formula",
            ),
            (
                "hb-negative-line",
                dataclasses.replace(
                    hb,
                    regions=(dataclasses.replace(hb.regions[0], line=-1),)
                    + hb.regions[1:],
                ),
                "plan.topology",
            ),
            (
                "hb-empty-phase",
                dataclasses.replace(
                    hb,
                    regions=(dataclasses.replace(hb.regions[0], phase=""),)
                    + hb.regions[1:],
                ),
                "plan.topology",
            ),
        )
        for case_name, plan, expected_code in cases:
            with self.subTest(case=case_name):
                with self.assertRaises(PlanValidationError) as raised:
                    plan.validate()
                self.assertEqual(
                    raised.exception.failure_codes, (expected_code,)
                )

    def test_malformed_binding_declarations_are_binding_invalid(self) -> None:
        plan = build_singleton()
        cases = (
            ("root-scalar", 1),
            (
                "dtype-object",
                {
                    "hidden": {},
                    "parameter": "float32",
                    "state": "float32",
                    "readout": "float32",
                },
            ),
            (
                "dtype-nonfinite",
                {
                    "hidden": float("nan"),
                    "parameter": "float32",
                    "state": "float32",
                    "readout": "float32",
                },
            ),
            (
                "dtype-surrogate",
                {
                    "hidden": "float32.\ud800",
                    "parameter": "float32",
                    "state": "float32",
                    "readout": "float32",
                },
            ),
        )
        for case_name, dtype_roles in cases:
            with self.subTest(case=case_name):
                binding = ConcreteBinding(dtype_roles)
                with self.assertRaises(PlanValidationError) as raised:
                    binding.validate_for(plan)
                self.assertEqual(
                    raised.exception.failure_codes, ("binding.invalid",)
                )

    def test_reference_ids_are_checked_by_the_schema_gate(self) -> None:
        base = build_chain(length=2)
        invalid_reference = " invalid.reference"
        cases = (
            (
                "entry",
                dataclasses.replace(
                    base, entry_node_ids=(invalid_reference,)
                ),
            ),
            (
                "terminal",
                dataclasses.replace(
                    base, terminal_node_ids=(invalid_reference,)
                ),
            ),
            (
                "region-member",
                dataclasses.replace(
                    base,
                    regions=(
                        dataclasses.replace(
                            base.regions[0], node_ids=(invalid_reference,)
                        ),
                        base.regions[1],
                    ),
                ),
            ),
            (
                "control-dependency",
                dataclasses.replace(
                    base,
                    regions=(
                        base.regions[0],
                        dataclasses.replace(
                            base.regions[1],
                            control_dependencies=(invalid_reference,),
                        ),
                    ),
                ),
            ),
        )
        for reference_kind, plan in cases:
            with self.subTest(reference_kind=reference_kind):
                self.assert_invalid(
                    plan,
                    "surrounding Unicode whitespace",
                    failure_codes=("plan.schema",),
                )

        numeric_entry = dataclasses.replace(
            base, entry_node_ids=(1,)  # type: ignore[arg-type]
        )
        self.assert_invalid(
            numeric_entry,
            "entry node ID must be a nonempty string",
            failure_codes=("plan.schema",),
        )

    def test_node_scalar_types_are_checked_by_the_schema_gate(self) -> None:
        base = build_singleton()
        invalid_owner = dataclasses.replace(
            base.nodes[0], state_owner=1  # type: ignore[arg-type]
        )
        self.assert_invalid(
            dataclasses.replace(base, nodes=(invalid_owner,)),
            "state owner ID must be a nonempty string",
            failure_codes=("plan.schema",),
        )

        invalid_forced_active = dataclasses.replace(
            base.nodes[0], forced_active=1  # type: ignore[arg-type]
        )
        self.assert_invalid(
            dataclasses.replace(base, nodes=(invalid_forced_active,)),
            "forced_active must be a boolean",
            failure_codes=("plan.schema",),
        )

    def test_static_capacity_is_topology_but_k_declaration_is_formula(self) -> None:
        base = build_single_layer(receiver_count=2, k=1)
        oversized_capacity = dataclasses.replace(
            base,
            regions=(dataclasses.replace(base.regions[0], k_max=3),),
        )
        self.assert_invalid(
            oversized_capacity,
            "k_max exceeds its fixed size",
            failure_codes=("plan.topology",),
        )

        invalid_declaration = dataclasses.replace(
            base,
            regions=(dataclasses.replace(base.regions[0], k_max=0),),
        )
        self.assert_invalid(
            invalid_declaration,
            "k_max must be a positive integer",
            failure_codes=("plan.formula",),
        )

    def test_forced_active_requires_singleton_and_exact_k_one(self) -> None:
        plan = build_single_layer(receiver_count=2, k=1)
        forced_node = dataclasses.replace(plan.nodes[0], forced_active=True)
        invalid = dataclasses.replace(
            plan, nodes=(forced_node, plan.nodes[1])
        )
        self.assert_invalid(invalid, "independent singleton region")

        singleton = build_singleton()
        invalid_k = dataclasses.replace(
            singleton,
            regions=(
                dataclasses.replace(
                    singleton.regions[0],
                    k_requested={
                        "type": "fixed",
                        "formula_id": "k.fixed.v1",
                        "value": 2,
                    },
                ),
            ),
        )
        self.assert_invalid(invalid_k, "fixed K must be")
        self.assert_invalid(invalid_k, "must request exactly one")

    def test_fixed_score_is_keyed_by_static_node_id(self) -> None:
        plan = build_diamond()
        region = plan.region_by_id("region.branches")
        bad_score = dataclasses.replace(
            region,
            score={
                "type": "fixed",
                "formula_id": "score.fixed-by-node.v1",
                "values_by_node": {"node.branch.a": 0.0},
            },
        )
        invalid = dataclasses.replace(
            plan,
            regions=tuple(
                bad_score if item.region_id == bad_score.region_id else item
                for item in plan.regions
            ),
        )
        self.assert_invalid(invalid, "fixed score keys must equal")

    def test_variable_k_is_only_the_bounded_requested_k_input(self) -> None:
        plan = build_diamond()
        region = plan.region_by_id("region.branches")
        valid_input = dataclasses.replace(
            region,
            k_max=2,
            k_requested={
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "requested_k",
                "minimum": 1,
                "maximum": 2,
            },
        )
        valid = dataclasses.replace(
            plan,
            regions=tuple(
                valid_input if item.region_id == region.region_id else item
                for item in plan.regions
            ),
        )
        self.assertIs(valid.validate(), valid)

        bad_field = dataclasses.replace(
            valid_input,
            k_requested={
                "type": "input",
                "formula_id": "k.input.v1",
                "field": "some_other_value",
                "minimum": 1,
                "maximum": 2,
            },
        )
        invalid = dataclasses.replace(
            plan,
            regions=tuple(
                bad_field if item.region_id == region.region_id else item
                for item in plan.regions
            ),
        )
        self.assert_invalid(invalid, "input K field must be 'requested_k'")

        arbitrary_formula = dataclasses.replace(
            valid_input,
            k_requested={
                "type": "dynamic",
                "formula": "2",
                "inputs": [],
                "minimum": 1,
                "maximum": 2,
                "timing": "content",
            },
        )
        invalid_formula = dataclasses.replace(
            plan,
            regions=tuple(
                arbitrary_formula if item.region_id == region.region_id else item
                for item in plan.regions
            ),
        )
        self.assert_invalid(invalid_formula, "type must be fixed or input")

    def test_custom_operation_requires_formula(self) -> None:
        plan = build_singleton()
        custom_node = dataclasses.replace(
            plan.nodes[0], aggregate={"type": "custom"}
        )
        invalid = dataclasses.replace(plan, nodes=(custom_node,))
        self.assert_invalid(invalid, "custom operation must declare its formula")

    def test_hb_constraints_and_line_barrier(self) -> None:
        hb = build_small_hb()
        line_sequence = [region.line for region in hb.topological_regions]
        self.assertEqual(line_sequence, sorted(line_sequence))
        self.assertEqual(set(hb.entry_node_ids), {hb.nodes[0].node_id})

        reversed_edge = dataclasses.replace(
            hb.edges[0], source=hb.edges[0].target, target=hb.edges[0].source
        )
        invalid_direction = dataclasses.replace(
            hb, edges=(reversed_edge,) + hb.edges[1:]
        )
        self.assert_invalid(invalid_direction, "must point to a deeper line")

        general_with_line = dataclasses.replace(hb, topology_kind="general")
        self.assert_invalid(general_with_line, "must not declare HB line/phase")

        bad_label = dataclasses.replace(hb.edges[0], label="data")
        invalid_label = dataclasses.replace(hb, edges=(bad_label,) + hb.edges[1:])
        self.assert_invalid(invalid_label, "invalid source label")

        malformed_line_region = dataclasses.replace(
            hb.regions[0], line="zero"  # type: ignore[arg-type]
        )
        malformed_line = dataclasses.replace(
            hb,
            regions=(malformed_line_region,) + hb.regions[1:],
        )
        self.assert_invalid(
            malformed_line,
            "line must be null or an integer",
            failure_codes=("plan.schema",),
        )


if __name__ == "__main__":
    unittest.main()
