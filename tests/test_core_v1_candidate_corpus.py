from __future__ import annotations

import hashlib
import unittest
from collections import Counter

from tide.generators import (
    CORE_V1_CANDIDATE_CORPUS_SEED,
    CORE_V1_CANDIDATE_CORPUS_SIZE,
    CORE_V1_CANDIDATE_VJP_SIZE,
    generate_core_v1_candidate_corpus,
)


_EXPECTED_MOTIF_COUNTS = {
    "singleton": 16,
    "single-layer-r2": 16,
    "single-layer-r8": 16,
    "chain": 16,
    "diamond": 16,
    "unequal-path": 16,
    "multi-entry-terminal": 16,
    "mixed-regions": 16,
    "forced-backbone": 16,
    "small-hb": 16,
    "generated-dag": 96,
}


def _generated_topology_signature(case):
    plan = case.plan
    return (
        plan.topology_kind,
        plan.entry_node_ids,
        plan.terminal_node_ids,
        tuple(
            (node.node_id, node.region_id, node.forced_active)
            for node in plan.nodes
        ),
        tuple(
            (edge.edge_id, edge.source, edge.target, edge.label)
            for edge in plan.edges
        ),
        tuple(
            (
                region.region_id,
                region.node_ids,
                region.control_dependencies,
                region.line,
                region.phase,
            )
            for region in plan.regions
        ),
    )


class CoreV1CandidateCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = generate_core_v1_candidate_corpus()

    def test_fixed_slots_are_valid_fixed_k_and_have_the_declared_vjp_subset(self):
        self.assertEqual(len(self.corpus), CORE_V1_CANDIDATE_CORPUS_SIZE)
        self.assertEqual(
            Counter(case.motif for case in self.corpus),
            Counter(_EXPECTED_MOTIF_COUNTS),
        )
        self.assertEqual(
            tuple(case.ordinal for case in self.corpus),
            tuple(range(CORE_V1_CANDIDATE_CORPUS_SIZE)),
        )
        self.assertEqual(
            tuple(case.case_id.split(".", 2)[1] for case in self.corpus),
            tuple(
                f"ql-{ordinal:04d}"
                for ordinal in range(CORE_V1_CANDIDATE_CORPUS_SIZE)
            ),
        )
        self.assertEqual(sum(case.vjp for case in self.corpus), 64)
        self.assertEqual(
            tuple(case.ordinal for case in self.corpus if case.vjp),
            tuple(range(0, CORE_V1_CANDIDATE_CORPUS_SIZE, 4)),
        )
        self.assertEqual(
            Counter(case.motif for case in self.corpus if case.vjp),
            Counter(
                {
                    **{motif: 4 for motif in _EXPECTED_MOTIF_COUNTS if motif != "generated-dag"},
                    "generated-dag": 24,
                }
            ),
        )
        self.assertEqual(CORE_V1_CANDIDATE_VJP_SIZE, 64)

        for case in self.corpus:
            with self.subTest(case=case.case_id):
                self.assertIs(case.plan.validate(), case.plan)
                self.assertEqual(case.generation_seed, CORE_V1_CANDIDATE_CORPUS_SEED)
                for region in case.plan.regions:
                    self.assertEqual(region.k_requested["type"], "fixed")
                    self.assertEqual(region.k_requested["value"], region.k_max)

    def test_identity_is_deterministic_and_logical_hashes_are_unique(self):
        repeated = generate_core_v1_candidate_corpus(
            CORE_V1_CANDIDATE_CORPUS_SEED
        )

        def identity(cases):
            return tuple(
                (
                    case.case_id,
                    case.plan.canonical_hash(),
                    case.parameter_seed,
                    case.input_seed,
                    case.features,
                    case.vjp,
                )
                for case in cases
            )

        actual = identity(self.corpus)
        self.assertEqual(actual, identity(repeated))
        self.assertEqual(
            len({case.case_id for case in self.corpus}),
            CORE_V1_CANDIDATE_CORPUS_SIZE,
        )
        self.assertEqual(
            len({case.plan.canonical_hash() for case in self.corpus}),
            CORE_V1_CANDIDATE_CORPUS_SIZE,
        )
        digest = hashlib.sha256(
            "\n".join(
                f"{case.case_id}\t{case.plan.canonical_hash()}\t"
                f"{case.parameter_seed}\t{case.input_seed}\t{int(case.vjp)}"
                for case in self.corpus
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            digest,
            "8497fccea52a958373ae5963c433a0f8420874005c88639ebd9e35d51fec6111",
        )

        changed_seed = generate_core_v1_candidate_corpus(
            CORE_V1_CANDIDATE_CORPUS_SEED + 1
        )
        self.assertNotEqual(
            tuple(case.plan.canonical_hash() for case in self.corpus),
            tuple(case.plan.canonical_hash() for case in changed_seed),
        )
        generated = [
            case for case in self.corpus if case.motif == "generated-dag"
        ]
        self.assertEqual(
            len({_generated_topology_signature(case) for case in generated}), 96
        )

    def test_declared_operator_and_semantic_coverage_is_broad(self):
        counts = Counter(
            feature for case in self.corpus for feature in case.features
        )
        expected = {
            *(f"motif:{motif}" for motif in _EXPECTED_MOTIF_COUNTS),
            "profile:N/content",
            "profile:SD/content",
            "profile:SD/pre",
            "profile:BO/content",
            "profile:BO/pre",
            "profile:BO/post",
            "state:none",
            "state:ema",
            "state:gdn",
            "state:attention_window",
            "selector-read:content",
            "selector-read:content_norm",
            "selector-read:content_linear",
            "selector-read:content_state_linear",
            "selector-read:content_state_summary_linear",
            "ffn-read:read.ffn.zero.v1",
            "ffn-read:read.ffn.ema.v1",
            "ffn-read:read.ffn.gdn.v1",
            "ffn-read:read.ffn.attention-window.v1",
            "aggregate:mean",
            "aggregate:edge_softmax",
            "aggregate:edge_linear_mean",
            "score:fixed",
            "score:constant",
            "score:read_sum",
            "score:linear",
            "score:mlp",
            "compute:identity",
            "compute:affine_residual",
            "compute:double_residual_swiglu",
            "emit:hard",
            "emit:hst",
            "emit:softp",
            "output-aggregate:mean",
            "output-aggregate:node_softmax",
            "budget:top-1",
            "budget:top-2",
            "budget:all",
            "shape:d2",
            "shape:d3",
            "shape:d4",
            "shape:d7",
            "k:fixed",
            "routing:forced-active",
            "boundary:multi-entry",
            "boundary:multi-terminal",
            "topology:hb",
        }
        self.assertFalse(expected - counts.keys(), sorted(expected - counts.keys()))
        self.assertTrue(
            all(counts[feature] >= 16 for feature in expected),
            sorted((feature, counts[feature]) for feature in expected if counts[feature] < 16),
        )
        self.assertNotIn("k:input", counts)
        self.assertEqual(
            {region.score["type"] for case in self.corpus for region in case.plan.regions},
            {"fixed", "constant", "read_sum", "linear", "mlp"},
        )
        self.assertEqual(
            {
                node.selector_read["type"]
                for case in self.corpus
                for node in case.plan.nodes
            },
            {
                "content",
                "content_norm",
                "content_linear",
                "content_state_linear",
                "content_state_summary_linear",
            },
        )
        for d_model in (2, 3, 4, 7):
            self.assertEqual(counts[f"shape:d{d_model}"], 64)
        self.assertEqual(counts["output-aggregate:mean"], 128)
        self.assertEqual(counts["output-aggregate:node_softmax"], 128)

    def test_seed_domain_is_strict(self):
        for invalid in (-1, True, 1.5, "7"):
            with self.subTest(seed=invalid):
                with self.assertRaises(ValueError):
                    generate_core_v1_candidate_corpus(invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
