import torch
from tidegraph import Continuation, Edge, External, Graph, Node, PortLayout, Region
from tidegraph.ops import Model


def fixture(dtype, kind, budget=2):
    nodes = (Node(0, memory="ssm", emission="slot_affine", emit_phases=(-1, -2, -1), aggregation=kind),
             Node(0, memory="attention", query_heads=2, kv_heads=1, aggregation=kind),
             Node(1, memory="delta", aggregation=kind))
    g = Graph(nodes, (Edge(0, 2, 2), Edge(0, 2, 2), Edge(1, 2, 2)), (Region(budget), Region(1)),
              (0, 0, 1, 1, 2, 2), (0, 1, 2), PortLayout((0, 1, 0), (2, 3, 0), (1, 0, 0, 1, 4, 1), (2, 1, 0)))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 2)
    variables = dict(m.named_parameters()); xs = []
    for b in range(2):
        for v in range(3):
            state = m.nodes[v].initial()
            state.value = torch.full_like(state.value, 0.1+b/10, requires_grad=True)
            q.states[b, v] = state; variables[f"initial.{b}.{v}"] = state.value
        for p, times in ((0, (0, 1, 3)), (1, (1,) if b == 0 else ()), (2, (0, 1, 3)), (4, (2, 3))):
            for i, time in enumerate(times):
                value = torch.sin(torch.arange(4, dtype=dtype)/7+(p+b+time)/9).requires_grad_()
                variables[f"input.{b}.{p}.{time}"] = value
                xs.append(External(b, p, i, time, value))
    return g, m, q, xs, variables
