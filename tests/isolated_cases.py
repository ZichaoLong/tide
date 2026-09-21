"""Independent leaves make structural gradient differences observable."""
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.ops import Model


def fixture(dtype, kind):
    g = Graph(tuple(Node(0 if n < 2 else 1, memory=kind, full="swiglu",
                         query_heads=2, kv_heads=1, window=4) for n in range(3)),
              (Edge(0, 2, 5), Edge(1, 2, 5)), (Region(1), Region(1)), (0, 1), (0, 1))
    m = Model(g, width=4, dtype=dtype)
    q = Continuation(g.identity, 2)
    variables = dict(m.named_parameters())
    xs = []
    for b in range(2):
        upstream = torch.nn.Parameter(torch.full((4,), 0.7 + b / 10, dtype=dtype))
        variables[f"upstream.{b}"] = upstream
        for n in range(2):
            state = m.nodes[n].initial()
            state.value = torch.full_like(state.value, 0.1 + b / 10, requires_grad=True)
            state.slots = {name: torch.full((1, *t.shape[1:]) if kind == "attention" else t.shape,
                                          0.13 + b / 10, dtype=dtype, requires_grad=True)
                           for name, t in state.slots.items()}
            if kind == "attention":
                state.observations = 1
            q.states[b, n] = state
            variables[f"initial.{b}.{n}.value"] = state.value
            variables.update({f"initial.{b}.{n}.{k}": t for k, t in state.slots.items()})
            for position, time in enumerate((1, 2, 3)):
                x = torch.sin(torch.arange(4, dtype=dtype) * 0.3 + (b+n+time) * 0.2).requires_grad_()
                variables[f"input.{b}.{n}.{time}"] = x
                xs.append(External(b, n, position, time, x * upstream))
    return g, m, q, xs, variables


def roots(result):
    first = next(e for e in result.trace if e["batch"] == 0 and e["time"] == 1)
    values = {"output": next(x for b, _, _, x in result.outputs if b == 0),
              "pending": next(a.value for a in result.continuation.pending if a.batch == 0),
              "state": result.continuation.states[0, 0].value}
    values.update({"state." + k: t for k, t in result.continuation.states[0, 0].slots.items()})
    values.update({"trace." + k: first[k] for k in ("content", "proposal", "descriptor", "control", "next")})
    values.update({"trace.slot." + k: t for k, t in first["proposal_slots"].items()})
    return values


def vjp(root, variables, zero=False):
    loss = root.square().sum() * (0.0 if zero else 0.7)
    return dict(zip(variables, torch.autograd.grad(loss, list(variables.values()),
                                                 allow_unused=True, retain_graph=True)))
