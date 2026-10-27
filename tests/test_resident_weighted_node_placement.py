"""Physical node placement, complete regions and cuts; independent CPU schedule.

Unlike sample replicas, every sample crosses both logical devices. Local
formula anchors remain those of docs/resident-weighted-placement.md.
"""
import os
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent
from tidegraph.reference import run
from tidegraph.weighted import WeightedResident
from resident_test_target import target_device
from test_weighted_execution import setup
from weighted_precision_reference import reference_model, widen, assert_records


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("topology", ["hb", "pdg"])
def test_nodes_cross_devices_keep_complete_records_and_relocate_cut(topology, storage_dtype, tmp_path):
    device = torch.device(target_device())
    assert device.index == 0
    b, _, xs, stop = setup(storage_dtype, topology)
    model = reference_model(b.model) if storage_dtype == torch.float16 else b.model
    devices = [str(torch.device(device.type, v % 2)) for v in range(len(b.graph.nodes))]
    moved = [str(torch.device(device.type, 1-v % 2)) for v in range(len(b.graph.nodes))]
    q = Continuation(b.graph.identity, 1)
    def owner(placement, q):
        return WeightedResident(b.graph, b.model, device=str(device), node_devices=placement,
            native_library=os.environ["TIDE_BUILD_DIR"], resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
            continuation=q, limits=dict(queue=64, stages=64, cache=16))
    head = [x for x in xs if x.time < 3]
    tail = [x for x in xs if x.time >= 3]
    with torch.no_grad():
        candidate = owner(devices, q)
        actual = candidate.advance(head, stop=3, sealed_until=3)
        expected = run(b.graph, model, q, widen(head), 3, sealed_until=3)
        assert_records(actual, expected, storage_dtype)
        assert actual.stats["node_peer_devices"] == 1 and actual.stats["node_peer_packet_bytes"] > 0
        manifest = candidate.manifest()
        assert manifest["requested_node_devices"] == manifest["effective_node_devices"] == devices
        save(tmp_path/"cut.pt", b.graph, b.model, candidate.checkpoint())
        candidate.close()
        q = load(tmp_path/"cut.pt", b.graph, b.model)
        equivalent(q, actual.continuation, rtol=0, atol=0)
        candidate = owner(moved, q)
        actual = candidate.advance(tail, stop=stop, sealed_until=stop)
        expected = run(b.graph, model, expected.continuation, widen(tail), stop, sealed_until=stop)
        assert_records(actual, expected, storage_dtype)
        empty = candidate.advance([], stop=stop, sealed_until=stop)
        equivalent(empty.continuation, actual.continuation, rtol=0, atol=0)
        assert not empty.trace
        candidate.close()
