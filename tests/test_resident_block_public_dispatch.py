"""Static default admission: never infer a Settle certificate or retry numerics."""
from dataclasses import replace
import pytest
import torch
from tidegraph.weighted import KVConfig, RecurrentConfig, bind_layered, fully_connected
from tidegraph.weighted.resident_blocks import block_resident_capability
from tidegraph.weighted.resident_training import WeightedResidentTraining


def fixture(memory=None):
    return bind_layered(fully_connected(2, 2), memory=memory or KVConfig(width=2, summary_width=2, memory_read="qry"), dtype=torch.float32)


@pytest.mark.parametrize("read", ["qry", "sum"])
def test_default_requires_certificate_and_dispatches_by_capability(monkeypatch, read):
    spec, binding = fixture(KVConfig(width=2, summary_width=2, memory_read=read))
    assert block_resident_capability(spec, binding.model)[0]
    assert not block_resident_capability(spec.graph, binding.model)[0]
    import tidegraph.weighted.resident_blocks as blocks
    calls = []
    sentinel = object()
    def construct(*args, **kwargs):
        calls.append((args, kwargs)); return sentinel
    monkeypatch.setattr(blocks, "WeightedBlockResidentTraining", construct)
    assert WeightedResidentTraining(spec, binding.model, device="npu:0", native_library="native", resident_library="resident") is sentinel
    assert len(calls) == 1 and calls[0][0][0] is spec


def test_incompatible_local_or_physical_schedule_stays_general():
    spec, binding = fixture(RecurrentConfig(width=2, summary_width=2, memory_family="affine"))
    assert not block_resident_capability(spec, binding.model)[0]
    assert type(WeightedResidentTraining.__new__(WeightedResidentTraining, spec, binding.model)) is WeightedResidentTraining
    spec, binding = fixture()
    graph = replace(spec.graph, nodes=(replace(spec.graph.nodes[0], emit_period=2), *spec.graph.nodes[1:]))
    assert not block_resident_capability(replace(spec, graph=graph), binding.model)[0]
    assert not block_resident_capability(spec, binding.model, device="npu:0", node_devices=["npu:0", "npu:1"])[0]


def test_memory_function_can_differ_between_complete_regions():
    topology = fully_connected(2, 2)
    sums = KVConfig(width=2, summary_width=2, memory_read="sum")
    queries = replace(sums, memory_read="qry")
    memories = [sums if node.region == 0 else queries for node in topology.graph.nodes]
    spec, binding = bind_layered(topology, memory=memories, dtype=torch.float32)
    assert block_resident_capability(spec, binding.model)[0]
    assert not hasattr(binding.model.nodes[0].kernel, "read_q")
    memories[1] = queries
    spec, binding = bind_layered(topology, memory=memories, dtype=torch.float32)
    supported, reason = block_resident_capability(spec, binding.model)
    assert not supported and "compatible KV geometry" in reason


def test_default_never_falls_back_after_selected_owner_failure(monkeypatch):
    spec, binding = fixture()
    import tidegraph.weighted.resident_blocks as blocks
    def fail(*args, **kwargs):
        raise RuntimeError("actual capacity or numerical refusal")
    monkeypatch.setattr(blocks, "WeightedBlockResidentTraining", fail)
    with pytest.raises(RuntimeError, match="actual capacity"):
        WeightedResidentTraining(spec, binding.model)
