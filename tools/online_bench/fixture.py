"""One model definition for independent public CPU/accelerator consumers.

The matching C++ builder uses the same named integer initializer. No candidate
receives another candidate's events, schedules, numerical results or gradients.
"""
import math
import torch
from tidegraph.graph import Graph, Node, Edge, Region
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph
from tidegraph.origins import InputOrigin
from tidegraph.ports import PortLayout
from tidegraph.source_domain import SourceDomain


def source_values(name, shape, seed):
    """Bounded int64 arithmetic, then an exactly specified FP32 source array."""
    key = seed % 2147483647
    for byte in name.encode("ascii"):
        key = (key * 131 + byte) % 2147483647
    x = (torch.arange(math.prod(shape), dtype=torch.int64) + key) % 2147483647
    for _ in range(3):
        x = (x * 1103515245 + 12345) % 2147483647
    return ((x % 65536).float() - 32768).mul_(2 ** -20).reshape(shape)


def body_graph(packet):
    g, c = packet["graph"], packet["workload"]
    attention = c["memory"] == "attention"
    return Graph(tuple(Node(r, clear=c["clear"], memory=("lh-fiber-attention-all-softmax-repeat-v1"
                              if attention else "lh-add-repeat-v1"), full="lh-silu-rms-v1",
                            aggregation="sum" if attention else "all_softmax",
                            emission="slot_affine", readout="norm-fp32-v1", query_heads=4, kv_heads=4)
                       for r in g["node_regions"]),
                 tuple(Edge(*edge) for edge in g["edges"]),
                 tuple(Region(c["budget"]) for _ in range(g["regions"])),
                 tuple(g["inputs"]), tuple(g["outputs"]))


def build_model(packet, *, dtype=torch.float32, device="cpu", node_devices=None):
    graph = body_graph(packet)
    c = packet["workload"]; d = c["width"]
    owners = [torch.device(device)]*len(graph.nodes) if node_devices is None else [torch.device(v) for v in node_devices]
    if len(owners) != len(graph.nodes):
        raise ValueError("consumer node_devices must cover every body node")
    # Construct only shape/program metadata. Replacing every tensor below avoids
    # allocating an unused D×D backbone for every LH node at full scale.
    with torch.device("meta"):
        model = Model(graph, d, dtype=dtype)
    constants = {}
    def constant(shape, kind="zero", *, owner=device):
        key = (tuple(shape), kind, str(owner))
        if key not in constants:
            value = (torch.eye(d) if kind == "identity" else
                     torch.full(shape, {"zero": 0., "one": 1., "retention": .99, "decay": .01}[kind]))
            constants[key] = torch.nn.Parameter(value.to(device=owner, dtype=dtype), requires_grad=False)
        return constants[key]
    def parameter(name, shape, ones=False, *, owner=device):
        value = torch.ones(shape) if ones else source_values(name, shape, c["seed"])
        return torch.nn.Parameter(value.to(device=owner, dtype=dtype))
    for v, w in enumerate(model.nodes):
        owner = owners[v]
        w.decay = w.bias = w.read = constant((d,), owner=owner)
        w.weight = constant((d, d), owner=owner)
        for name, value in list(w.extra.items()):
            if name == "lh_norm_weight":
                w.extra[name] = parameter(f"node/{v}/norm", (d,), True, owner=owner)
            elif name in ("fiber_qkv", "fiber_out"):
                w.extra[name] = parameter(f"node/{v}/{name}", value.shape, owner=owner)
            elif name == "add_retention":
                w.extra[name] = constant((), "retention", owner=owner)
            elif name == "fiber_decay":
                w.extra[name] = constant((), "decay", owner=owner)
            elif name == "fiber_pool" or name.startswith("agg_logit_"):
                w.extra[name] = parameter(f"node/{v}/{name}", value.shape, True, owner=owner)
            else:
                w.extra[name] = constant(value.shape, owner=owner)
        for slot, binding in enumerate(graph.port_indexes[1].row(v)):
            kind, index = binding
            w.extra[f"emit_w_{slot}"] = (parameter(f"edge/{index}/projection", (d, d), owner=owner)
                                         if kind else constant((d, d), "identity", owner=owner))
    for name in ("input_scale", "agg_scale", "edge_scale", "output_scale"):
        setattr(model, name, torch.nn.ParameterList(constant((), "one") for _ in getattr(model, name)))
    embedding = parameter("embedding", (c["vocab"], d))
    head = parameter("head", (c["vocab"], d))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad) + embedding.numel() + head.numel()
    if count != packet["counts"]["parameters"]:
        raise ValueError(f"materialized parameter count {count} disagrees with packet")
    return graph, model, embedding, head


def encode(packet, graph, model):
    if packet["graph"]["kind"] == "ranked-local":
        return SettleGraph(graph, tuple(packet["graph"]["ranks"])).embed(model)
    n, r, e = len(graph.nodes), len(graph.regions), len(graph.edges)
    ports, domain = graph.ports, graph.domain
    edges = (graph.edges + tuple(Edge(n, v, 1) for v in graph.inputs)
             + tuple(Edge(v, n + 1, 1) for v in graph.outputs))
    layout = PortLayout(ports.edge_source + tuple(range(len(graph.inputs))) + ports.output,
                        ports.edge_target + ports.input + tuple(range(len(graph.outputs))), (0,), (0,))
    encoded = Graph(graph.nodes + (Node(r, identity=True), Node(r+1, identity=True)), edges,
                    graph.regions + (Region(1), Region(1)), (n,), (n+1,), layout,
                    tuple(InputOrigin(e+p, p, packet["workload"]["stride"]) for p in range(len(graph.inputs))),
                    SourceDomain(domain.edge_target + domain.input + tuple(range(len(graph.outputs))), (0,)))
    # The same generic boundary adapter as Settle, without claiming rank-aligned
    # equivalence for delayed arrivals. Only the static wiring differs.
    with torch.device("meta"):
        em = Model(encoded, model.width, dtype=model.nodes[0].bias.dtype)
    from tidegraph.ops import BoundaryWeights
    tail = [BoundaryWeights(model.width, model.nodes[0].bias.dtype).to(model.nodes[0].bias.device) for _ in range(2)]
    em.nodes = torch.nn.ModuleList([*model.nodes, *tail])
    def one():
        return torch.nn.Parameter(model.nodes[0].bias.new_ones(()), requires_grad=False)
    em.input_scale = torch.nn.ParameterList([one()]); em.output_scale = torch.nn.ParameterList([one()])
    em.agg_scale = torch.nn.ParameterList([*model.agg_scale, *model.input_scale, *model.output_scale])
    em.edge_scale = torch.nn.ParameterList([*model.edge_scale, *(one() for _ in (*graph.inputs, *graph.outputs))])
    return encoded, em
