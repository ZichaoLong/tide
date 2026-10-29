"""Public recipes reach independent schedules, full training state and disk resume.

Local formula/Jacobian tests remain separate anchors. FP64 avoids claiming that
these small checks resolve the recorded larger FP32 numerical boundaries.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.planning.recipes import prepare
from tidegraph.planning.materialize import materialize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_benchmark_runners import Eager
from weighted_benchmark_resume import check


CASES = [(top, family, {}) for top in ("fc", "hb", "lh")
         for family in ("kv", "affine", "delta", "gated_delta", "ssd", "mixed")]
CASES += [("hb", "kv", dict(profile="textbook")), ("hb", "mixed", dict(profile="textbook")),
          ("fc", "kv", dict(regularizer="bern")), ("fc", "kv", dict(regularizer="ent", ent_alpha=1.5)),
          ("hb", "kv", dict(selection="regional_total_zero")), ("lh", "kv", dict(selection="unit_attention"))]


@pytest.mark.parametrize("topology,family,options", CASES)
def test_recipe_complete_training_and_new_owner(tmp_path, topology, family, options):
    core = os.environ.get("TIDE_BUILD_DIR")
    if not core:
        pytest.skip("explicit matching native library required")
    plan, _ = prepare(dict(topology=topology, family=family, width=8, region_width=2, depth=1, **options),
                      samples=2, positions=2)
    create = lambda: materialize(plan, max_bytes=10_000_000, dtype=torch.float64)
    config = SimpleNamespace(samples=2, positions=2)
    args = SimpleNamespace(mode="training", dtype="float64", scoring_dtype="float64", schedule="prefill",
                           placement="native", native_library=core, implementation="native", profile=False,
                           workers=2, packing="packed", steps=1)
    b, xs, span = create();candidate = Eager(b, xs, span, config, args, torch.device("cpu"))
    b, xs, _ = create();reference = Eager(b, xs, span, config, args, torch.device("cpu"), reference=True)
    for step in range(2):
        actual, loss = candidate.step(step)
        expected, reference_loss = reference.step(step)
        equivalent(actual, expected)
        torch.testing.assert_close(torch.tensor(loss, dtype=torch.float64),
                                   torch.tensor(reference_loss, dtype=torch.float64), atol=1e-10, rtol=1e-8)
        equivalent(candidate.parameters(), reference.parameters())
        equivalent(candidate.training_state(), reference.training_state())
    assert check(candidate, create, config, args, torch.device("cpu"), tmp_path)["exact"]
