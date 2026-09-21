import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import objective
from tidegraph.ops import Model


def fixture(dtype, window=3, adoption="all", cyclic=False, aligned=False):
    nodes = tuple(Node(0 if i < 2 else 1, clear=adoption == "clear" and i < 2,
                       memory="attention", full="swiglu", query_heads=2,
                       kv_heads=1 if i != 1 else 2, window=window) for i in range(3))
    g = Graph(nodes, (Edge(0, 2, 1), Edge(1, 2, 1 if aligned else 2)) + ((Edge(2, 0, 1),) if cyclic else ()),
              (Region(1, observe_all=adoption != "selected"), Region(1)), (0, 1), (2,))
    m = Model(g, width=4, dtype=dtype)
    x = torch.sin(torch.arange(96, dtype=dtype) * 0.19).reshape(3, 2, 4, 4).requires_grad_()
    q = Continuation(g.identity, 3)
    variables = dict(m.named_parameters()) | {"input": x}
    for b in range(3):
        for v in range(3):
            s = m.nodes[v].initial(); s.observations = b
            s.value = torch.full_like(s.value, 0.11, requires_grad=True)
            s.slots = {k: (torch.arange(b * t.shape[1] * t.shape[2], dtype=dtype).reshape(b, *t.shape[1:])
                           * (0.13 if k == "key" else -0.17)).requires_grad_() for k, t in s.slots.items()}
            q.states[b, v] = s
            variables[f"initial.{b}.{v}.value"] = s.value
            variables.update({f"initial.{b}.{v}.{k}": t for k, t in s.slots.items()})
    xs = [External(b, p, i, t, x[b, p, i]) for b in range(3) for p in range(2)
          for i, t in enumerate((1, 3, 6, 8) if b < 2 else (1, 6))]
    return g, m, q, xs, x, variables


def vjp(result, variables):
    return dict(zip(variables, torch.autograd.grad(objective(result), list(variables.values()), allow_unused=True)))
