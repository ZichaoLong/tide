"""Independent shape, ownership and live-state witnesses for memory extensions."""
from dataclasses import replace

import pytest
import torch

from benchmarks.capacity import budget, placement_budget, scale_configs
from benchmarks.continuous import state_memory
from benchmarks.statistics import ParameterInventory, RuntimeAudit, audit_model
from benchmarks.workloads import activity_fixture, paired_input
from tide import ClockConfig, ModelConfig, TideModel, TopologyConfig, place_model
from tide.parameters import NodeLinear, NodeOwners
from tide.precision import precision_context


def _memory_parameters(kind, d, k, f):
    # Count each named mathematical projection and the SSD small vectors.
    if kind == "K":
        return 2*d*k + d*(d//4) + d*k + k*d + 3*d*f
    h = d//32
    return d*(2*d+64+h) + 2*d*(d//4) + d*d + 3*d*f + 3*h + d + 5*(d+64)


@pytest.mark.parametrize("memory", ["X", "S", "X_reverse"])
def test_memory_parameter_shapes_match_independent_formulas(memory):
    d, k, f = 64, 32, 80
    config = ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1),
                         d=d, memory=memory, kv_width=None if memory == "S" else k, ffn_width=f)
    model = TideModel(config)
    for kind, module in model.memories.items():
        assert sum(p.numel() for p in module.parameters()) == model.kinds.count(kind)*_memory_parameters(kind, d, k, f)
    assert ModelConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("placement", [None, "layer", "region"])
def test_narrow_reverse_planned_state_equals_actual_payload(precision, placement):
    config = ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1),
                         d=64, memory="X_reverse", kv_width=32, ffn_width=80,
                         clock=ClockConfig(kv_window=7, ssd_stride=2))
    planned = budget(config, 2, 9, precision, "inference")
    wide = budget(replace(config, kv_width=None), 2, 9, precision, "inference")
    layout = planned["state_layout"]
    assert (layout["content_width"], layout["summary_width"], layout["kv_payload_width"], layout["kv_heads"]) == (64, 16, 32, 1)
    element_bytes = 2 if precision == "bf16" else 4
    assert wide["state_layout"]["K"]-layout["K"] == 2*layout["kv_slots"]*2*32*element_bytes
    model = TideModel(config)
    expected = planned["bounded_state_bytes"]
    if placement is not None:
        plan = placement_budget(config, 2, 9, precision, partition_kind=placement)
        model = place_model(model, ["cpu", "cpu"], **{placement+"_partitions": plan[placement+"_partitions"]})
        expected = sum(row["cold_state_payload_bytes"] for row in plan["partitions"])
        assert sum(p.numel() for p in model.parameters()) == plan["unique_parameters"]
    with torch.no_grad(), precision_context("cpu", precision):
        result = model(torch.randn(2, 9, 64))
    assert state_memory(result.state)["payload_bytes"] == expected


def test_narrow_attention_work_counts_actual_qkv_heads():
    config = ModelConfig(d=64, kv_width=32, allocation="pool", clock=ClockConfig(kv_window=18))
    model = TideModel(config)
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(2, 5, 64, 81))
    # All FC owners write and act. Visible causal keys are 1,2,3,3,3.
    expected = len(model.graph.nodes)*2*(1+2+3+3+3)*1
    assert audit["work"]["kv_attention_effective_pairs"] == expected
    assert audit["node_matrices"]["memories.K.kv"]["dense_multiply_adds"] == len(model.graph.nodes)*2*5*64*(2*32)


def test_memory_shared_full_touch_keeps_independent_ffn():
    d, k, f, rank, depth = 64, 32, 80, 8, 4
    model = TideModel(ModelConfig(d=d, kv_width=k, ffn_width=f, memory_sharing="region_shared", budget=1))
    activity_fixture(model, "sparse03")
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(2, 3, d, 91))
    # One active owner per region: one Q/O base use, its private query/output
    # adapters, and three fully independent FFN matrices. No FFN adapters.
    full = 2*d*k + (k//32)*rank*(d+32) + rank*(k+d) + 3*d*f
    assert audit["stages"]["full"]["mean"] == depth*full
    assert not model.memories["K"].ffn.up.shared
    assert model.memories["K"].output.shared


def test_memory_vector_classification_uses_its_own_ownership():
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="exchange", n=2, depth=1, hubs=1),
                                 d=64, memory="S", memory_sharing="region_shared"))
    inventory = ParameterInventory(model)
    vector_atoms = [atom for atom in inventory.classes if atom[0] == "memories.S.a_log.value"]
    ffn_atoms = [atom for atom in inventory.classes if atom[0] == "memories.S.ffn.up.weight"]
    assert vector_atoms and ffn_atoms
    assert {inventory.classes[atom] for atom in vector_atoms} == {"base"}
    assert {inventory.classes[atom] for atom in ffn_atoms} == {"D", "H"}
    assert inventory.report()["unique_parameters"] == sum(p.numel() for p in model.parameters())


def test_matrix_audit_separates_uniform_gemm_from_repeated_group_bmm():
    # Runtime dispatch records actual tensors independently of semantic counts.
    layer = NodeLinear(4, 3, 5, groups=[0, 0, 1, 1], rank=1)
    with RuntimeAudit() as runtime:
        layer(torch.randn(2, 2, 3), NodeOwners.from_static([0, 1], device="cpu"))
        layer(torch.randn(4, 2, 3), NodeOwners.from_static([0, 1, 2, 3], device="cpu"))
    shapes = runtime.report()["matrix_shapes"]
    assert any(row["operator"] == "aten.mm.default" and row["inputs"] == [[4, 3], [3, 5]] for row in shapes)
    assert any(row["operator"] == "aten.bmm.default" and row["inputs"] == [[4, 2, 3], [4, 3, 5]] for row in shapes)
    model = TideModel(ModelConfig(topology=TopologyConfig(kind="lattice", n=8, depth=1), d=64,
                                 memory_sharing="region_shared", region_batch_size=2, allocation="pool"))
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(1, 2, 64, 123))
    memory = [row for row in audit["node_matrix_dispatch"] if row["path"] == "memories.K.kv"]
    assert memory
    assert {row["base_operator"] for row in memory} == {"mm", "bmm"}
    assert sum(row["repeated_base_weight_rows"] for row in memory) == 2*(8-1)


def test_fused_projections_keep_logical_cost_but_count_one_base_contraction(monkeypatch):
    model = TideModel(ModelConfig(d=64, allocation="pool", aggregation="incoming_light_64",
                                 reception_qkv_backend="fused", memory_sharing="region_shared"))
    monkeypatch.setattr(model.memories["K"].query, "uses_fused_projection", lambda x: True)
    with torch.no_grad():
        _, _, audit = audit_model(model, paired_input(1, 2, 64, 123))
    dispatch = audit["node_matrix_dispatch"]
    query = [row for row in dispatch if row["path"].startswith("memories.K.query")]
    assert query and all(row["path"] == "memories.K.query.fused_heads" for row in query)
    assert all(row["out_features"] == 64 for row in query)
    reception = [row for row in dispatch if row["path"].startswith("receive.")]
    assert any(row["path"] == "receive.fused_qkv" and row["out_features"] == 192 for row in reception)
    assert not {row["path"] for row in reception} & {"receive.q", "receive.k", "receive.v"}
    # Mathematical projection work remains reported for each original matrix.
    assert all(f"memories.K.query.heads.{i}" in audit["node_matrices"] for i in range(2))
    assert all(f"receive.{name}" in audit["node_matrices"] for name in ("q", "k", "v"))


@pytest.mark.parametrize("change,expected", [
    ({}, 8778358224), ({"memory": "S"}, 8494364112), ({"memory": "X_reverse"}, 8523946832),
    ({"kv_width": 384}, 7872388560), ({"kv_width": 256}, 7570398672),
    ({"kv_width": 128}, 7268408784), ({"memory_sharing": "region_shared"}, 6936315376),
])
def test_reference_9b_actual_constructor_counts_without_payload(change, expected):
    config = replace(scale_configs()["hb-9b"], **change)
    with torch.device("meta"):
        model = TideModel(config)
    assert all(p.is_meta for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == expected
    assert (sum(n.role == "D" for n in model.graph.nodes), sum(n.role == "H" for n in model.graph.nodes), len(model.graph.regions)) == (768, 80, 44)
