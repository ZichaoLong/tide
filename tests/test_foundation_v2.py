"""Concrete v2 option propagation and complete-boundary schedule transitions."""
import json
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from foundation_execute import Execution
from foundation_policy import resolve
from foundation_measure import execute_pass
from tidegraph.compare import equivalent, objective
from tidegraph.blocks import canonicalize
from tidegraph.records import Result
from isolated_cases import vjp
from test_specialized_topologies import loss_vjp

SUITE = json.loads((Path(__file__).resolve().parents[1]/"benchmarks/foundation-v2.json").read_text())
CASES = [(c, v) for c in SUITE["configurations"] for v in c["variants"]]


def small(config, dtype):
    c = dict(config, execution_schema="v2", width=8, batch=2, sequence=4, dtype=str(dtype).split(".")[-1])
    if c.get("prefill_positions"): c["prefill_positions"] = 2
    if c.get("training_window"): c["training_window"] = 2
    return c


@pytest.mark.parametrize("config,variant", CASES, ids=[c["id"]+"-"+v for c, v in CASES])
def test_policy_actual_paths_and_transition(dtype, config, variant):
    c = small(config, dtype)
    ref = Execution(c, "python-layered" if c["graph"] == "settle" else "python-stream", trace=True)
    engine = Execution(c, variant, workers=2, trace=True)
    expected = ref.advance(4)
    events, messages, outputs, stats = [], [], [], {}
    for stop, phase in engine.cuts():
        part = engine.advance(stop, phase=phase)
        events += part.trace; messages += part.messages; outputs += part.outputs
        for key, value in part.stats.items():
            stats[key] = max(stats.get(key, 0), value) if key.startswith("max_") else stats.get(key, 0)+value
    actual = canonicalize(engine.graph, Result(engine.q, events, outputs, messages, stats))
    equivalent(expected, actual)
    variables = dict(engine.model.named_parameters())
    ref_variables = dict(ref.model.named_parameters())
    # objective() is already the declared scalar loss; do not square it again.
    equivalent(loss_vjp(objective(expected), ref_variables), loss_vjp(objective(actual), variables))
    for root in (lambda r: r.trace[0]["content"], lambda r: r.outputs[0][-1]):
        equivalent(vjp(root(expected), ref_variables), vjp(root(actual), variables))
    resolved = engine.policy["resolved"]
    assert "attention_packing" not in engine.options
    if variant.endswith("optimized"):
        assert stats["batched_full_events"] > 0 and stats["batched_aggregate_events"] > 0
        assert not stats.get("semantic_full_replays", 0) and not stats.get("semantic_aggregate_replays", 0)
    if resolved["schedule"]["prefill"]:
        assert stats.get("max_state_sequence", 0) > 1


@pytest.mark.parametrize("variant", ["native-frontier-optimized", "python-frontier-optimized"])
def test_prefill_decode_timings_have_separate_denominators(dtype, variant):
    config = small(next(c for c in SUITE["configurations"] if c["id"] == "T01"), dtype)
    result = execute_pass(Execution(config, variant, workers=2), ["nograd-forward"])
    assert [s["phase"] for s in result["segments"]] == ["prefill", "streaming", "streaming"]
    assert [s["effective_input_positions"] for s in result["segments"]] == [4, 2, 2]
    assert result["segments"][0]["execution_paths"]["max_state_sequence"] == 2
    assert all(s["execution_paths"]["max_state_sequence"] <= 1 for s in result["segments"][1:])


@pytest.mark.parametrize("options,variant", [({"full_autograd":"batched","packed":False},"native-frontier"),
    ({"prefill":True},"native-stream"), ({"batch_next":True},"python-frontier"),
    ({"attention_packing":"single"},"native-frontier"), ({"defer_state_release":True},"native-frontier"),
    ({"invented":True},"native-frontier")])
def test_unsupported_explicit_policies_are_rejected(options, variant):
    config = next(c for c in SUITE["configurations"] if c["id"] == "T01")
    with pytest.raises(ValueError): resolve(config, variant, 2, options)
