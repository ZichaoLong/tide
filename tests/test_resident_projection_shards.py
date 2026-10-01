"""Compact physical projection ownership, continuation and canonical training."""
import json
import pytest
import torch
from tidegraph import ResidentPlacement
from tidegraph.compare import equivalent
from test_resident_training import target, training_case
from resident_training_cases import runtime, inputs


@pytest.mark.parametrize("family,schedule", [("pdg", "greedy"), ("timed-dag", "streaming"), ("settle", "greedy")])
def test_projection_owner_banks_and_complete_training(target, family, schedule, tmp_path):
    r = runtime(family, target, schedule, full="lh-silu-rms-v1", aggregation="all_softmax",
                emission="slot_affine", model_device="cpu", resident_workspace_bytes=1024**3)
    cpu = runtime(family, "cpu", full="lh-silu-rms-v1", aggregation="all_softmax", emission="slot_affine")
    count = len(r.execution_graph.nodes)
    owners = ResidentPlacement(devices=(target, f"npu:{torch.device(target).index+1}"), full_owners=tuple(n % 2 for n in range(count)),
                               state_owners=tuple((n+1) % 2 for n in range(count)))
    x = torch.sin(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)*.19)*.1
    with torch.no_grad(), r.session(2, placement=owners) as s:
        oracle = cpu.session(2)
        for start, stop in ((0, 2), (2, 4)):
            a, kw = inputs(s, x, start, stop);s.advance_device(a, **kw)
            a, kw = inputs(oracle, x, start, stop);expected = oracle.advance(a, **kw)
            result = s.result();equivalent(expected, result)
            banks = [v for k, v in result.stats.items() if k.startswith("projection_parameter_bytes_device_")]
            assert result.stats["projection_shards"] == len(banks) == 2
            assert result.stats["projection_peer_packet_bytes"] > 0
            assert min(banks) > 0 and max(banks) < sum(banks)
            assert result.stats["planned_headroom_bytes"] >= 0
    # Includes cross-owner parameter aliases, scale-zero connections, public
    # gradients/masters/slots, three nonzero updates and checkpoint restore.
    statistics = training_case(target, family, schedule, "adamw", tmp_path, full="lh-silu-rms-v1",
                  aggregation="all_softmax", emission="slot_affine", placement=owners,
                  model_device="cpu", root_modes=("all", "all", "all"))
    (tmp_path/"projection-shards.json").write_text(json.dumps(dict(
        family=family, schedule=schedule, forward_statistics=result.stats, reverse_statistics=statistics)))


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_projection_bank_only_on_remote_owner(target, dtype_name):
    import os
    from tidegraph import Edge, Graph, Node, Region, GraphConfig, GraphRuntime, ExecutionOptions, ExecutionPlacement, ResidentLimits
    graph = Graph((Node(0, identity=True), Node(1, emission="slot_affine")),
                  (Edge(0, 1, 1),), (Region(1), Region(1)), (0,), (1,))
    cfg = GraphConfig("pdg", graph, width=4, dtype=dtype_name)
    candidate = GraphRuntime(cfg, device=target, model_device="cpu",
        native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
        options=ExecutionOptions(implementation="native", schedule="greedy", trace=False,
            placement=ExecutionPlacement(preset="resident"), resident_limits=ResidentLimits(workspace_bytes=1024**3)))
    cpu = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, trace=False))
    placement = ResidentPlacement(devices=(target, f"npu:{torch.device(target).index+1}"), full_owners=(0, 1), state_owners=(1, 0))
    x = torch.full((1, 3, 4), .125, dtype=getattr(torch, dtype_name))
    with torch.no_grad(), candidate.session(1, placement=placement) as s:
        oracle = cpu.session(1)
        for start, stop in ((0, 1), (1, 3)):
            a, kw = inputs(s, x, start, stop);s.advance_device(a, **kw)
            a, kw = inputs(oracle, x, start, stop);expected = oracle.advance(a, **kw)
            result = s.result();equivalent(expected, result,
                atol=2e-3 if dtype_name=="float16" else 1e-6, rtol=2e-2 if dtype_name=="float16" else 1e-5)
            assert result.stats["projection_shards"] == 1
            assert f"projection_parameter_bytes_device_{torch.device(target).index}" not in result.stats
            assert result.stats[f"projection_parameter_bytes_device_{torch.device(target).index+1}"] > 0
