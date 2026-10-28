"""Static-budget algebra, physical topology, aliasing and actual execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import Continuation
from tidegraph.reference import run
from tidegraph.weighted import fully_connected, hb_lattice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_topology import topology
from weighted_workload_plan import make_plan, validate_plan, canonical, match_ffn, count, LEGACY_TARGETS
from weighted_plan_materialize import load, materialize
from weighted_workload_plan import core3_body_target


@pytest.mark.parametrize("kind", ["fc", "hb"])
def test_static_layered_topology_matches_public_definition(kind):
    g = topology(kind, depth=2, region_width=3)
    spec = fully_connected(2, 3) if kind == "fc" else hb_lattice(2, 2, 3)
    assert g["node_regions"] == [n.region for n in spec.graph.nodes]
    assert g["edges"] == [[e.source, e.target, e.delay] for e in spec.graph.edges]
    assert g["ranks"] == list(spec.ranks)
    if kind == "hb":
        assert len(set(g["ranks"])) < len(g["ranks"])


def test_historical_topology_and_parent_classes():
    g = topology("ranked", region_width=32)
    assert (g["nodes"], len(g["edges"])) == (480, 2208)
    p, inv = make_plan(g, width=2048, samples=512, positions=12)
    assert p["classes"]["edges"] == [a+1 for a,_,_ in g["edges"]]
    assert p["memory_floor"]["logical_parameter_bytes"] == 4*inv["totals"]["unique"]
    assert not p["memory_floor"]["complete_admission"]
    assert LEGACY_TARGETS == {"add-body": 9262008512, "attention-body": 17315072192}


@pytest.mark.parametrize("width", [64, 2048])
@pytest.mark.parametrize("memory", ["add", "attention"])
def test_scaled_core3_body_matches_historical_packet(width, memory):
    from flow_topology import make_packet
    graph = topology("ranked", region_width=32)
    packet = make_packet(graph=graph, width=width, memory=memory, vocab=50304)
    expected = packet["counts"]["parameters"]-2*50304*width
    assert core3_body_target(graph, width, memory) == expected
    plan, _ = make_plan(graph, width=width, match_core3=memory, samples=1, positions=1)
    assert plan["parameter_match"]["target_body"] == expected
    assert abs(plan["parameter_match"]["difference"]) <= 1.5*width
    assert plan["parameter_match"]["reference"]["task_head_parameters_excluded"]


def test_recurrent_dimensions_make_budget_feasible_and_charge_matrix_state():
    graph=topology("ranked",region_width=32)
    for family in ("gated_delta","ssd"):
        plan,_=make_plan(graph,family=family,width=2048,key_width=64,value_width=128,heads=2,
                         target=LEGACY_TARGETS["add-body"],samples=1,positions=2)
        assert abs(plan["parameter_match"]["difference"])<=3072
        assert plan["memories"][0]["key_width"]==64
        m=plan["memory_floor"]
        assert m["resident_raw_initial_state_old_bytes"]==3*m["resident_raw_one_snapshot_bytes"]
    with pytest.raises(ValueError,match="do not apply"):
        make_plan(graph,key_width=64)


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_budget_is_real_ffn_parameters_with_bounded_residual(family):
    p, _ = make_plan(topology("hb"), width=6, family=family)
    minimum = deepcopy(p)
    for c in minimum["memories"]: c["ffn_width"] = 1
    base = count(minimum)["totals"]["trainable"]
    matched = match_ffn(p, base+1234)
    actual = count(matched)["totals"]["trainable"]
    widths = [c["ffn_width"] for c in matched["memories"]]
    assert actual == base+18*sum(x-1 for x in widths)
    assert max(widths)-min(widths) <= 1
    assert abs(actual-base-1234) <= 9
    with pytest.raises(ValueError, match="minimum"): match_ffn(p, base-1)


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("kind", ["hb", "dag", "pdg"])
def test_plan_materializes_and_executes_real_ports(family, kind):
    p, inv = make_plan(topology(kind, layers=2, region_width=2), width=4,
                       samples=1, positions=1, family=family)
    b, xs, span = materialize(p, max_bytes=1000000)
    assert sum(x.numel() for x in b.model.parameters() if x.requires_grad) == inv["totals"]["trainable"]
    assert all(x.value is xs[0].value for x in xs)
    result = run(b.graph, b.model, Continuation(b.graph.identity, 1), xs, span, sealed_until=span)
    assert result.outputs and result.trace
    if kind == "pdg": assert result.continuation.pending
    torch.stack([x.square().sum() for *_,x in result.outputs]).sum().backward()
    assert any(x.grad is not None for x in b.model.parameters())
    with pytest.raises(ValueError, match="ceiling"): materialize(p, max_bytes=1)


def test_no_torch_cli_and_hash_refusals(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path/"plan"
    result = subprocess.run([sys.executable, "-S", str(root/"scripts/prepare_weighted_workload.py"),
        "--topology", "ranked", "--region-width", "32", "--width", "2048", "--samples", "512",
        "--positions", "12", "--match", "attention-body", "--output-dir", str(out)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    p, c = load(out/"workload.json")
    assert c.width == 2048 and abs(p["parameter_match"]["difference"]) <= 3072
    p["span"] += 1
    with pytest.raises(ValueError, match="hash"): validate_plan(p)
    p["sha256"] = hashlib.sha256(canonical({k:v for k,v in p.items() if k != "sha256"})).hexdigest()
    with pytest.raises(ValueError, match="time domain"): validate_plan(p)


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1}, {"samples": 0}, {"positions": 129},
                                  {"width": False}, {"k": 3}, {"source_mode": "adapted"}])
def test_invalid_plans_refuse_before_allocation(kwargs):
    with pytest.raises(ValueError): make_plan(topology("hb", region_width=2), **kwargs)
