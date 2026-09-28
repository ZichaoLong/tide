"""Finite library qualification cases; benchmark graph builders are developer-only."""
from dataclasses import replace
import json
from pathlib import Path
from tidegraph import Graph, GraphConfig, Node, Edge, Region, ExecutionOptions
from tidegraph.fiber_attention import PROFILE
from tidegraph.topology import topology

MEMORIES = ("ema", "linear", "delta", "delta-rule-v1", "ssm", "attention", "lh-add-repeat-v1", PROFILE)


def mixed_node(region, index):
    return Node(region, memory=MEMORIES[index % len(MEMORIES)], full="swiglu" if index % 2 else "tanh",
                aggregation="sum" if index % len(MEMORIES) == 7 else ("sum", "mean", "weighted_mean", "active_softmax")[index % 4],
                query_heads=2, kv_heads=1 if index % len(MEMORIES) == 5 else 2,
                window=3 if index % len(MEMORIES) == 5 else 0)


def cases():
    """Requested model widths remain real; qualify() records reduced dimensions."""
    policy = ExecutionOptions(implementation="native", mode="hst", full_autograd="batched", aggregate_autograd="batched")
    graph, ranks = topology("layered", layers=[16] * 4)
    graph = replace(graph, nodes=tuple(mixed_node(n.region, i) for i,n in enumerate(graph.nodes)))
    for family in ("timed-dag", "settle"):
        yield f"active64-{family}", GraphConfig(family, graph, width=128, ranks=ranks if family == "settle" else (),
                                                 execution=policy, scale_init=.25), dict(origin="fully active mixed modules", expected_observed_nodes=64)
    nodes = tuple(mixed_node(v, v) for v in range(32))
    edges = (tuple(Edge(v, (v+1)%32, 1) for v in range(32))
             + tuple(Edge(0, v, 1+v%3) for v in range(1,32)) + (Edge(0,1,1),))
    graph = Graph(nodes, edges, tuple(Region(1) for _ in nodes), (0,), (31,))
    yield "feedback32", GraphConfig("pdg", graph, width=128, execution=policy, scale_init=.25), dict(
        origin="fully active ring + fanout hub + varied delays + parallel edges", expected_observed_nodes=32)
    from foundation_workloads import graph_for
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "benchmarks/foundation-v1.json").read_text())
    for config in suite["configurations"]:
        if config["id"] not in {"P01", "P02", "T02", "S01", "A02"}:
            continue
        graph, spec = graph_for(config)
        yield "benchmark-" + config["id"], GraphConfig(config["graph"], graph, width=config["width"],
            ranks=spec.ranks if spec else (), execution=policy, scale_init=.8), dict(
                origin="exact foundation-v1 graph records; qualifier initialization and reduced tensors",
                benchmark=config, graph_identity=graph.identity)
