"""Unequal physical input activity, original clocks and connected windows."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys
import unittest

from tidegraph.planning.input_pattern import coordinates
from tidegraph.planning.model import fingerprint
from tidegraph.planning.recipes import prepare
from tidegraph.planning.workload import validate_plan

RECIPE = dict(topology="fc", width=8, region_width=2, depth=2)


class InputPatternTest(unittest.TestCase):
    def test_dense_identity_is_unchanged_and_pattern_changes_only_workload(self):
        dense, inventory = prepare(RECIPE, samples=3, positions=4)
        # Captured from the unmodified public generator at cbb0964.
        self.assertEqual(dense["sha256"], "389fe93332373e9e29a9977764b454096efe599bb2ca59eb2067a306c5e087f0")
        self.assertNotIn("input_pattern", dense)
        uneven, other = prepare(RECIPE, samples=3, positions=4, input_pattern="staggered-v1")
        self.assertEqual(dense["model_id"], uneven["model_id"])
        self.assertEqual(inventory, other)
        self.assertEqual(dense["memory_floor"], uneven["memory_floor"])
        self.assertNotEqual(dense["sha256"], uneven["sha256"])

    def test_hand_enumerated_occurrences_are_distinct_from_global_positions(self):
        plan, _ = prepare(RECIPE, samples=3, positions=4, input_pattern="staggered-v1")
        self.assertEqual(list(coordinates(plan)), [
            (0, 0, 1, 0), (0, 1, 0, 0), (0, 1, 1, 1),
            (0, 2, 0, 1), (0, 2, 1, 2), (0, 3, 0, 2),
            (1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 1),
            (1, 1, 1, 1), (1, 2, 0, 2),
            (2, 0, 0, 0), (2, 0, 1, 0), (2, 1, 0, 1)])

    def test_unknown_or_inconsistent_policy_refuses_before_materialization(self):
        for pattern in ("unknown", True, {}, 1):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, "input pattern"):
                prepare(RECIPE, input_pattern=pattern)
        plan, _ = prepare(RECIPE, input_pattern="staggered-v1")
        for key, value in (("input_pattern", None), ("input_policy", "dense")):
            bad = deepcopy(plan)
            bad[key] = value
            bad["sha256"] = fingerprint({k: v for k, v in bad.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "input pattern"):
                validate_plan(bad)

    def test_complete_two_window_schedules_keep_real_activity_and_cache_differences(self):
        import torch
        from tidegraph import Continuation
        from tidegraph.compare import equivalent
        from tidegraph.greedy import run as prefill
        from tidegraph.reference import run as reference
        from tidegraph.planning.materialize import materialize
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from weighted_benchmark_cases import window

        plan, _ = prepare(RECIPE, samples=3, positions=4, input_pattern="staggered-v1")
        bound, inputs, span = materialize(plan, max_bytes=1_000_000, dtype=torch.float64)
        self.assertEqual(Counter(x.batch for x in inputs), {0: 6, 1: 5, 2: 3})
        # Same physical broadcast root at a populated sample/time, distinct samples.
        self.assertIs(inputs[1].value, inputs[2].value)
        self.assertFalse(torch.equal(inputs[0].value, inputs[6].value))
        initial = Continuation(bound.graph.identity, 3)
        scalar_state, packed_state = initial, initial
        with torch.no_grad():
            for index in range(2):
                xs = window(inputs, span, plan["positions"], index, "cpu")
                actual = prefill(bound.graph, bound.model, packed_state, xs,
                                 (index+1)*span, sealed_until=(index+1)*span)
                expected = reference(bound.graph, bound.model, scalar_state, xs,
                                     (index+1)*span, sealed_until=(index+1)*span)
                equivalent(actual, expected)
                scalar_state, packed_state = expected.continuation, actual.continuation
                self.assertEqual({x.time-inputs[i].time for i, x in enumerate(xs)}, {index*span})
        self.assertGreater(len(set(Counter(e["batch"] for e in actual.trace).values())), 1)
        self.assertGreater(len(set(Counter(m.batch for m in actual.messages).values())), 1)
        cache_lengths = {state.slots["times"].numel() for state in actual.continuation.states.values()}
        self.assertGreater(len(cache_lengths), 1)
        ledger = defaultdict(list)
        for index in range(2):
            for x in window(inputs, span, plan["positions"], index, "cpu"):
                ledger[x.batch, x.port].append(x.position)
        for counters in ledger.values():
            self.assertEqual(counters, list(range(len(counters))))

    def test_unequal_training_windows_keep_gradients_optimizer_and_exact_resume(self):
        import tempfile
        from types import SimpleNamespace
        import torch
        from tidegraph.compare import equivalent
        from tidegraph.planning.materialize import materialize
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
        from weighted_benchmark_runners import Eager
        from weighted_benchmark_resume import check

        plan, _ = prepare(RECIPE, samples=3, positions=3, input_pattern="staggered-v1")
        create = lambda: materialize(plan, max_bytes=1_000_000, dtype=torch.float64)
        config = SimpleNamespace(samples=3, positions=3)
        args = SimpleNamespace(mode="training", dtype="float64", scoring_dtype="float64", schedule="prefill",
                               placement="native", native_library=None, implementation="python", profile=False,
                               workers=1, packing="packed", steps=1)
        device = torch.device("cpu")
        bound, inputs, span = create()
        candidate = Eager(bound, inputs, span, config, args, device)
        bound, inputs, span = create()
        independent = Eager(bound, inputs, span, config, args, device, reference=True)
        try:
            for index in range(2):
                actual, loss = candidate.step(index)
                expected, reference_loss = independent.step(index)
                equivalent(actual, expected)
                torch.testing.assert_close(torch.tensor(loss, dtype=torch.float64),
                                           torch.tensor(reference_loss, dtype=torch.float64), atol=1e-10, rtol=1e-8)
                equivalent(candidate.parameters(), independent.parameters())
                equivalent(candidate.training_state(), independent.training_state())
            with tempfile.TemporaryDirectory() as out:
                self.assertTrue(check(candidate, create, config, args, device, Path(out))["exact"])
        finally:
            candidate.close()
            independent.close()


if __name__ == "__main__":
    unittest.main()
