"""Large topology metadata is exact; small executions verify public initialization."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import pytest
import torch
from tidegraph import GraphConfig, GraphRuntime
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from library_cases import cases, MEMORIES


def test_real_topologies_and_fully_active_case_contract():
    records = {name:(config,meta) for name,config,meta in cases()}
    assert len(records) == 8
    assert len(records["benchmark-P02"][0].graph.nodes) == 8192
    for name in ("active64-timed-dag", "active64-settle"):
        config, metadata = records[name]
        assert len(config.graph.nodes) == 64 and len(config.graph.edges) == 768
        assert {n.memory for n in config.graph.nodes} == set(MEMORIES)
        assert metadata["expected_observed_nodes"] == 64
        assert all(r.budget == 16 for r in config.graph.regions)


def test_explicit_scale_initialization_and_supplied_weights(dtype):
    cfg = GraphConfig.from_dict(dict(schema_version=1, family="pdg", topology=dict(kind="chain"),
        model=dict(width=4, dtype=str(dtype).split(".")[-1], scale_init=.25)))
    runtime = GraphRuntime(cfg, device="cpu")
    for name in ("input_scale", "output_scale", "agg_scale", "edge_scale"):
        assert all(p.item() == .25 for p in getattr(runtime.model, name))
    with torch.no_grad():
        runtime.model.input_scale[0].fill_(.5)
    supplied = GraphRuntime(cfg, device="cpu", model=runtime.model)
    assert supplied.model.input_scale[0].item() == .5
    assert supplied.manifest()["model_origin"] == "caller"
    assert GraphConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).identity == cfg.identity
    with pytest.raises(ValueError, match="scale_init"):
        replace(cfg, scale_init=float("nan"))
