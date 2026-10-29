"""Public configuration identities, textbook bindings and independent topology checks."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import Continuation
from tidegraph.reference import run
from tidegraph.weighted import KVConfig, RegionalConfig, hb_lattice
from tidegraph.planning.model import resolve_model, workload, fingerprint
from tidegraph.planning.materialize import materialize
from tidegraph.planning.recipes import prepare, model_recipe, scaled_recipe
from tidegraph.planning.topology import layered, lh_inspired
from tidegraph.planning.workload import validate_plan


def test_defaults_are_resolved_before_identity_and_execution_is_separate():
    graph = layered("fc", depth=2, region_width=2)
    a, _ = resolve_model(graph, {}, {})
    b, _ = resolve_model(graph, asdict(KVConfig()), asdict(RegionalConfig()))
    assert a == b
    p, _ = workload(a, samples=1, positions=1)
    q, _ = workload(a, samples=8, positions=512)
    assert p["model_id"] == q["model_id"] and p["sha256"] != q["sha256"]
    assert q["span"] == 512*q["stride"]
    bad = deepcopy(q)
    bad["regions"][0]["allocation_base"] = "mu"
    bad["sha256"] = fingerprint({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="inapplicable"):
        validate_plan(bad)


def test_hb_offsets_match_public_topology_and_textbook_counts():
    offsets = ((0, 1),)*6
    planned = layered("hb", depth=3, region_width=64, offsets=offsets)
    public = hb_lattice(2, 3, 64, offsets=offsets)
    assert (planned["nodes"], planned["regions"], len(planned["edges"])) == (1408, 22, 3584)
    assert planned["edges"] == [[e.source, e.target, e.delay] for e in public.graph.edges]
    assert planned["ranks"] == list(public.ranks)
    assert len(set(planned["ranks"])) == 7  # No artificial staggering of regions.
    lh = lh_inspired(depth=2, region_width=8)
    assert any(delay == 2 for _, _, delay in lh["edges"])
    assert all(delay == lh["ranks"][lh["node_regions"][b]]-lh["ranks"][lh["node_regions"][a]]
               for a, b, delay in lh["edges"])
    assert lh["counts"]["reachable_nodes"] == lh["counts"]["output_reachable_nodes"] == lh["nodes"]


def test_textbook_positions_reads_and_sparse_budget_are_explicit():
    m, _ = model_recipe(topology="hb", profile="textbook", width=512, region_width=2, depth=1,
                        sparse_denominator=8)
    configs = [m["memories"][i] for i in (0, 2, 6)]
    assert [c["memory_read"] for c in configs] == ["qry", "sum", "sum"]
    assert [len(c["full_frequencies"][0]) for c in configs] == [32, 0, 16]
    assert configs[0]["window"] == 512*5 and configs[1]["window"] == 0
    assert len(configs[0]["read_frequencies"]) == 16
    assert [c["region_score"] for c in m["regions"]] == ["att", "ind", "ind", "att"]
    assert all(c["query_weight"] == "received" for c in m["regions"] if c["region_score"] == "att")
    assert all(c["k"] == 1 for c in m["regions"])


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd", "mixed"])
def test_recipe_materialization_and_backward_reaches_real_module_parameters(family):
    p, inv = prepare(dict(topology="hb", family=family, width=8, region_width=2, depth=1,
                         sparse_denominator=2), positions=2)
    b, xs, span = materialize(p, max_bytes=10000000, dtype=torch.float64)
    assert sum(x.numel() for x in b.model.parameters() if x.requires_grad) == inv["totals"]["trainable"]
    result = run(b.graph, b.model, Continuation(b.graph.identity, 1), xs, span, sealed_until=span)
    assert result.outputs
    torch.stack([value.square().sum() for *_, value in result.outputs]).sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in b.model.parameters())
    for state in result.continuation.states.values():
        assert type(state.last_time) is int and type(state.observations) is int
        for name in ("time", "times", "conv_times"):
            if name in state.slots:
                assert state.slots[name].dtype == torch.int64 and not state.slots[name].requires_grad


def test_natural_scale_and_no_torch_cli(tmp_path):
    m, inv, scale = scaled_recipe(10**9, topology="fc", region_width=8)
    assert scale["actual_parameters"] == inv["totals"]["trainable"]
    assert abs(scale["relative_difference"]) < .15
    assert all(c["ffn_width"] == 4*c["width"] for c in m["memories"])
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-S", str(root/"scripts/prepare_core5_study.py"),
        "--topology", "hb", "--target-parameters", "1000000000", "--output-dir", str(tmp_path/"plan")],
        capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path/"plan/workload.json").read_text())
    validate_plan(plan)
    assert plan["recipe"]["sparse_budget"]["requested_fraction"] == 1/8


def test_small_numeric_types_and_inapplicable_fields_refuse_without_tensors():
    with pytest.raises(ValueError, match="does not apply"):
        model_recipe(selection="regional_total_zero", regularizer="bern")
    with pytest.raises(ValueError, match="sum memory_read"):
        model_recipe(memory_overrides={"read_frequencies": (1.,)})
    with pytest.raises(ValueError, match="integer"):
        model_recipe(width=True)


def test_execution_and_measurement_identity_separate_placement_and_precision():
    from tidegraph.planning import ExecutionSpec, MeasurementSpec
    a = ExecutionSpec().record()
    b = ExecutionSpec(client="python-native", device="npu:0", placement="mixed-b",
                      intraop_threads=32, workers=8).record()
    assert a["execution_id"] != b["execution_id"]
    assert a["effective"]["packing"] == "packed"
    stream = ExecutionSpec(schedule="streaming").record()
    assert stream["requested"]["packing"] == "auto" and stream["effective"]["packing"] == "scalar"
    x = MeasurementSpec().record("model-input", b["execution_id"])
    y = MeasurementSpec(steps=3).record("model-input", b["execution_id"])
    assert x["measurement_id"] != y["measurement_id"]
    for kwargs in ({"client": "pytorch", "placement": "resident", "device": "npu:0"},
                   {"device": "npu:0", "dtype": "float64"}, {"operator": "unimplemented-fusion"},
                   {"device": "npu:0", "peer_devices": ["npu:0", "npu:0"]},
                   {"schedule": "streaming", "packing": "packed"}):
        with pytest.raises(ValueError):
            ExecutionSpec(**kwargs)
