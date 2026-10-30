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
from tidegraph.planning.facts import describe
from tidegraph.planning.execution import ExecutionSpec
from dataclasses import asdict
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_benchmark_runners import Eager
from weighted_benchmark_resume import check


CASES = [(top, family, {}) for top in ("fc", "hb", "lh")
         for family in ("kv", "affine", "delta", "gated_delta", "ssd", "mixed")]
CASES += [("fc", "kv", dict(profile="textbook")), ("fc", "mixed", dict(profile="textbook")),
          ("hb", "kv", dict(profile="textbook")), ("hb", "mixed", dict(profile="textbook")),
          ("fc", "kv", dict(regularizer="bern")), ("fc", "kv", dict(regularizer="ent", ent_alpha=1.5)),
          ("hb", "kv", dict(selection="regional_total_zero")), ("lh", "kv", dict(selection="unit_attention"))]


def case_name(case):
    top, family, options = case
    variant = options.get("profile", options.get("regularizer", options.get("selection", "baseline")))
    return f"{top}-{family}-{variant}"


def assert_intent(plan, topology, family, options, depth):
    g, memories = plan["graph"], plan["memories"]
    expected = {"kv", "gated_delta", "ssd"} if family == "mixed" else {family}
    assert {c.get("memory_family", "kv") for c in memories} == expected
    assert len(g["edges"]) > 0 and len(set(g["ranks"])) > 1
    assert g["counts"]["reachable_nodes"] == g["nodes"] == g["counts"]["output_reachable_nodes"]
    if topology == "fc":
        assert g["ranks"] == list(range(1, depth+1))
        assert len(g["edges"]) == (depth-1)*2**2
    else:
        assert len(g["ranks"]) > len(set(g["ranks"]))  # Same-rank independent regions.
    assert plan["classes"]["edges"] == [a+1 for a, _, _ in g["edges"]]
    if options.get("profile") == "textbook":
        kv = [c for c in memories if "memory_family" not in c]
        assert any(any(c["full_frequencies"]) for c in kv)
        if topology != "fc":
            assert any(c["memory_read"] == "qry" and c["read_frequencies"] for c in kv)
            assert any(c["region_score"] == "att" for c in plan["regions"])
    return expected


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("topology,family,options", CASES, ids=[case_name(c) for c in CASES])
def test_recipe_complete_training_and_new_owner(tmp_path, record_property, topology, family, options, implementation):
    core = os.environ.get("TIDE_BUILD_DIR")
    # FC middle regions occur at ranks 4,8: two are needed for both GDN and SSD.
    depth = (8 if family == "mixed" else 4 if options.get("profile") == "textbook" else 2) if topology == "fc" else 1
    plan, _ = prepare(dict(topology=topology, family=family, width=8, region_width=2, depth=depth, **options),
                      samples=2, positions=2)
    execution = ExecutionSpec(client="python-native" if implementation == "native" else "pytorch",
                              dtype="float64", scoring_dtype="float64", packing="packed",
                              workers=2 if implementation == "native" else 1).record()
    record_property("core5_case", json.dumps(dict(case=case_name((topology, family, options)),
        facts=describe(plan), execution=execution, mode="training", updates=2)))
    expected_families = assert_intent(plan, topology, family, options, depth)
    if implementation == "native" and not core:
        pytest.skip("explicit matching native library required")
    create = lambda: materialize(plan, max_bytes=10_000_000, dtype=torch.float64)
    config = SimpleNamespace(samples=2, positions=2)
    args = SimpleNamespace(mode="training", dtype="float64", scoring_dtype="float64", schedule="prefill",
                           placement="native", native_library=core, implementation=implementation, profile=False,
                           workers=execution["requested"]["workers"], packing="packed", steps=1)
    b, xs, span = create();candidate = Eager(b, xs, span, config, args, torch.device("cpu"))
    assert json.loads(json.dumps([asdict(c) for c in b.memories])) == plan["memories"]
    b, xs, _ = create();reference = Eager(b, xs, span, config, args, torch.device("cpu"), reference=True)
    for step in range(2):
        actual, loss = candidate.step(step)
        expected_result, reference_loss = reference.step(step)
        equivalent(actual, expected_result)
        torch.testing.assert_close(torch.tensor(loss, dtype=torch.float64),
                                   torch.tensor(reference_loss, dtype=torch.float64), atol=1e-10, rtol=1e-8)
        equivalent(candidate.parameters(), reference.parameters())
        equivalent(candidate.training_state(), reference.training_state())
        assert {plan["memories"][e["node"]].get("memory_family", "kv") for e in actual.trace} == expected_families
    assert check(candidate, create, config, args, torch.device("cpu"), tmp_path)["exact"]
    record_property("core5_checks", json.dumps(dict(post_measurement_complete_records=True,
        parameter_values_checked=True, final_parameter_gradients_checked=True, optimizer_values_checked=True,
        exact_resume=dict(state="passed", exact=True))))
