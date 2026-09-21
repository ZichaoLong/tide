import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.ops import Model


def fixture(dtype, selected_only=False):
    kinds = ("ema", "ssm", "linear", "delta", "attention")
    g = Graph(tuple(Node(int(i > 2), memory=k, full="swiglu", clear=i == 1,
                         query_heads=2, window=3) for i, k in enumerate(kinds)),
              (Edge(0, 2, 2), Edge(0, 2, 3), Edge(1, 3, 1), Edge(2, 4, 1), Edge(3, 0, 2), Edge(4, 1, 3)),
              (Region(2, observe_all=not selected_only), Region(1)), (0, 1, 3), (2, 4))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 3)
    x = torch.sin(torch.arange(96, dtype=dtype) * 0.2).reshape(2, 3, 4, 4).requires_grad_()
    variables = dict(m.named_parameters()) | {"input": x}
    for b in range(3):
        for v in range(5):
            s = m.nodes[v].initial()
            s.value = torch.full_like(s.value, 0.07, requires_grad=True)
            s.slots = {name: torch.full_like(t, 0.11, requires_grad=True) for name, t in s.slots.items()}
            q.states[b, v] = s
            variables[f"initial.{b}.{v}.value"] = s.value
            variables.update({f"initial.{b}.{v}.{name}": t for name, t in s.slots.items()})
    xs = [External(b, p, i, t, x[b, p, i]) for b in range(2) for p in range(3)
          for i, t in enumerate((0, 1, 4, 7) if b == 0 else (0, 5))]
    return g, m, q, xs, variables
