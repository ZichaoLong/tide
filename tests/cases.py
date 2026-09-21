import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.ops import Model


def ring(dtype, stop=7):
    graph = Graph((Node(0), Node(0, clear=True), Node(1), Node(2)),
                  (Edge(0, 2, 1), Edge(1, 2, 1), Edge(2, 0, 2), Edge(2, 1, 2),
                   Edge(2, 3, 1), Edge(0, 2, 1)),
                  (Region(1), Region(1, count_priority=False), Region(1)), (0, 1, 3), (2, 3))
    model = Model(graph, dtype=dtype)
    x = (torch.arange(36, dtype=dtype).reshape(12, 3) / 50 - 0.2).requires_grad_()
    external = []
    i = 0
    for batch in range(2):
        for port in range(3):
            for position, time in enumerate((0, 3)):
                external.append(External(batch, port, position, time, x[i])); i += 1
    q = Continuation(graph.identity, 2)
    initial = torch.full((2, 4, 3), 0.13, dtype=dtype, requires_grad=True)
    for b in range(2):
        for v in range(4):
            q.states[b, v] = State(initial[b, v])
    return graph, model, q, external, x, initial


def gradients(loss, model, x, initial):
    tensors = dict(model.named_parameters()) | {"input": x, "initial": initial}
    grads = torch.autograd.grad(loss, list(tensors.values()), allow_unused=True)
    return dict(zip(tensors, grads))
