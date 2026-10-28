"""Independent static formula -> actual Parameter identities, never forward."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import Edge, Graph, Node, Region
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, SourceBinding, bind, hb_lattice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_inventory import inventory, class_widths


def graph_record(graph):
    return dict(node_regions=[x.region for x in graph.nodes],
                edges=[[x.source, x.target, x.delay] for x in graph.edges],
                inputs=list(graph.inputs), outputs=list(graph.outputs))


def test_json_class_width_keys_are_exact_metadata():
    assert class_widths({"0": 4, "9223372036854775807": 5}) == {0: 4, 2**63-1: 5}
    for widths in ({True: 4}, {1.2: 4}, {"01": 4}, {"1.0": 4}, {"-1": 4}, {"1": 4, 1: 4}):
        with pytest.raises(ValueError, match="int64"):
            class_widths(widths)
    with pytest.raises(ValueError, match="node regions"):
        inventory(dict(node_regions=[2**62], edges=[], inputs=[0], outputs=[0]), {}, {})


def compare(bound, planned):
    # This expectation comes from materialized nn.Modules, independent of the
    # static shape/alias formulas. Include frozen transport Parameter objects.
    parameters = dict(bound.model.named_parameters(remove_duplicate=False))
    assert parameters.keys() == planned["roles"].keys()
    groups = {}
    for name, parameter in parameters.items():
        row = planned["roles"][name]
        assert list(parameter.shape) == row["shape"], name
        assert parameter.requires_grad == row["trainable"], name
        groups.setdefault(id(parameter), set()).add(name)
    assert {frozenset(x) for x in groups.values()} == {
        frozenset(x["aliases"]) for x in planned["owners"].values()}
    assert sum(x.numel() for x in bound.model.parameters() if x.requires_grad) == planned["totals"]["trainable"]
    assert sum(x.numel() for x in bound.model.parameters() if not x.requires_grad) == planned["totals"]["frozen"]


MEMORIES = [KVConfig(width=6, heads=2, memory_read=mode, summary_width=3, adapter_rank=2, ffn_width=7)
            for mode in ("sum", "qry")]
MEMORIES += [RecurrentConfig(memory_family=family, memory_read=mode, width=6, summary_width=3,
                            key_width=2, value_width=4, ffn_width=7,
                            **({} if family in {"affine", "delta"} else dict(heads=4, groups=2 if family == "ssd" else 1, conv_size=3)))
             for family in ("affine", "delta", "gated_delta", "ssd") for mode in ("sum", "state")]
REGIONS = [RegionalConfig(), RegionalConfig(local_score="match", region_score="att", region_evidence="grouped", region_key="content"),
           RegionalConfig(region_score="att"), RegionalConfig(region_control_mode="fixed", total_fraction=.3),
           RegionalConfig(selection_mode="constrained", send_gain="context"),
           RegionalConfig(selection_mode="hard_selection", gain_logit=.2),
           RegionalConfig(selection_mode="direct_attention", region_score="att", region_key="content")]


@pytest.mark.parametrize("memory", MEMORIES, ids=lambda c: c.__class__.__name__+"-"+getattr(c, "memory_family", "kv")+"-"+c.memory_read)
@pytest.mark.parametrize("region", REGIONS, ids=lambda c: c.selection_mode+"-"+c.local_score+"-"+c.region_score+"-"+str(c.region_evidence))
def test_all_memory_and_descriptor_roles_match_actual_parameters(memory, region):
    graph = Graph((Node(0), Node(0), Node(1)), (Edge(0, 2, 1), Edge(1, 2, 2), Edge(0, 2, 1)),
                  (Region(2), Region(1)), (0, 1), (2,))
    bound = bind(graph, memories=memory, regions=region)
    compare(bound, inventory(graph_record(graph), asdict(memory), asdict(region)))


@pytest.mark.parametrize("mode", ["region", "node", "source", "adapted"])
def test_source_classes_shared_parameters_heterogeneous_memories(mode):
    graph = hb_lattice(2, 1, 2, offsets=((0, 1), (0,))).graph
    classes = SourceClasses(tuple(e.source+1 for e in graph.edges), (0,)*len(graph.inputs))
    source = SourceBinding(mode, 2 if mode == "adapted" else None)
    memories = [MEMORIES[i % len(MEMORIES)] for i in range(len(graph.nodes))]
    regions = REGIONS[1]
    width = ({label: 2+label%3 for label in set(classes.edges+classes.inputs)} if mode == "source" else 5)
    bound = bind(graph, memories=memories, regions=regions, source_classes=classes,
                 source_binding=source, source_hidden_width=width)
    compare(bound, inventory(graph_record(graph), list(map(asdict, memories)), asdict(regions),
                             classes=classes, source_binding=source, source_hidden_width=width))


def test_explicit_alias_freezing_and_published_small_parameter_count():
    graph = hb_lattice(2, 1, 2, offsets=((0, 1), (0,))).graph
    bound = bind(graph)
    plain = inventory(graph_record(graph), KVConfig(), RegionalConfig())
    assert plain["totals"]["trainable"] == 2460  # Independent recorded fixture count.
    compare(bound, plain)
    a, b = "nodes.1.kernel.summary", "nodes.0.kernel.summary"
    bound.model.nodes[1].kernel.summary = bound.model.nodes[0].kernel.summary
    bound.model.nodes[0].kernel.summary.requires_grad_(False)
    tied = inventory(graph_record(graph), KVConfig(), RegionalConfig(), aliases={a: b}, frozen=[a])
    compare(bound, tied)
    assert tied["totals"]["trainable"] == plain["totals"]["trainable"]-16


def test_cli_planning_dependency_boundary_and_large_count_without_tensors():
    scripts = str(Path(__file__).resolve().parents[1]/"scripts")
    program = """
import sys
sys.path.insert(0, sys.argv[1])
from flow_topology import ranked_graph
from weighted_inventory import inventory
g=ranked_graph()
m=dict(width=2048, summary_width=1024, adapter_rank=2, ffn_width=4096)
r=dict(hidden_width=1024)
p=inventory(g,m,r,source_hidden_width=2048)
assert p['totals']['trainable']==20398698735
assert 'torch' not in sys.modules
print(p['totals']['trainable'])
"""
    result = subprocess.run([sys.executable, "-S", "-c", program, scripts], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_invalid_alias_and_shared_source_width_fail_before_allocation():
    graph = dict(node_regions=[0, 0], edges=[[0, 1, 1]], inputs=[0], outputs=[1])
    with pytest.raises(ValueError, match="shapes disagree"):
        inventory(graph, {}, {}, classes=dict(edges=[1], inputs=[0]), source_hidden_width={0: 2, 1: 3})
    with pytest.raises(ValueError, match="alias"):
        inventory(graph, {}, {}, aliases={"nodes.0.kernel.summary": "nodes.1.kernel.adapter_u"})
    with pytest.raises(ValueError, match="int64"):
        inventory(graph, {}, {}, classes=dict(edges=[0], inputs=[False]))
