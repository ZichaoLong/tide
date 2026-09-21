import torch
from tidegraph import Continuation, Edge, External, Graph, Node, PortLayout, Region
from tidegraph.ops import Model


def fixture(dtype, memory="ema", clear=False):
    nodes = (Node(0, clear=clear, memory=memory, full="swiglu", query_heads=2, kv_heads=1,
                  emission="slot_affine", emit_period=2, emit_phases=(0, -2, -1)),
             Node(0, memory=memory, full="swiglu", query_heads=2, kv_heads=1,
                  emission="slot_affine", emit_period=2, emit_phases=(-1, 1)),
             Node(1, memory=memory, full="swiglu", query_heads=2, kv_heads=1, emission="slot_affine"))
    g = Graph(nodes, (Edge(0, 2, 2), Edge(0, 2, 2), Edge(1, 2, 3)),
              (Region(2), Region(1)), (0, 1), (0, 1, 2),
              PortLayout((1, 0, 0), (0, 1, 2), (0, 0), (2, 1, 0)))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 2)
    variables = dict(m.named_parameters()); xs = []
    for b in range(2):
        for n in range(3):
            state = m.nodes[n].initial()
            state.value = torch.full_like(state.value, 0.12 + 0.02*b, requires_grad=True)
            state.slots = {k: torch.full((1, *t.shape[1:]) if memory == "attention" else t.shape,
                                        0.1, dtype=dtype, requires_grad=True) for k, t in state.slots.items()}
            if memory == "attention":
                state.observations = 1
            q.states[b, n] = state
            variables[f"initial.{b}.{n}.value"] = state.value
            variables.update({f"initial.{b}.{n}.{k}": t for k, t in state.slots.items()})
        for p in range(2):
            for i, time in enumerate((0, 1, 3, 4)):
                x = torch.sin(torch.arange(4, dtype=dtype) * 0.2 + (b+p+time) / 7).requires_grad_()
                variables[f"input.{b}.{p}.{time}"] = x
                xs.append(External(b, p, i, time, x))
    return g, m, q, xs, variables
