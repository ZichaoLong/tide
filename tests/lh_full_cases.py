import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.ops import Model


def fixture(dtype, profile, clear=False):
    g = Graph(tuple(Node(0 if v < 2 else 1, memory="lh-add-repeat-v1", full=profile,
                         clear=clear, emission="slot_affine", readout="norm-fp64-v1") for v in range(3)),
              (Edge(0, 2, 2), Edge(0, 2, 3), Edge(1, 2, 3)),
              (Region(1, selector="lh-count-affect-v1"), Region(1)), (0, 1), (0, 1, 2))
    m = Model(g, width=3, dtype=dtype)
    # Share the normalization parameters while keeping graph-local signaling
    # projections separate (the nodes have different outgoing slot domains).
    for name in ("lh_norm_weight", "lh_norm_bias"):
        if name in m.nodes[0].extra:
            m.nodes[1].extra[name] = m.nodes[0].extra[name]
    leaves = dict(m.named_parameters()); q = Continuation(g.identity, 3); xs = []
    for b in range(3):
        for v in range(3):
            value = torch.tensor([.2+b/10, -.3-v/10, .1], dtype=dtype, requires_grad=True)
            leaves[f"initial.{b}.{v}"] = value; q.states[b, v] = State(value)
    for b, times in enumerate(((0, 2, 4), (1, 4))):
        for p in range(2):
            for i, t in enumerate(times):
                value = torch.tensor([.3+b/10, -.2+p/10, .15+t/10], dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{p}.{i}"] = value; xs.append(External(b, p, i, t, value))
    return g, m, q, xs, leaves
