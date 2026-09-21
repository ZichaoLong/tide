"""Independent history/input leaves and small exact regional recurrences."""
import torch
from tidegraph import Continuation, External, Graph, History, Node, Region
from tidegraph.ops import Model


def fixture(dtype, *, policy="all"):
    g = Graph(tuple(Node(0, clear=policy == "clear", next_state="control-blend-v1" if policy == "blend" else "adopt-v1")
                    for _ in range(3)) + (Node(1),), (),
              (Region(1, observe_all=policy != "selected", count_priority=False, read_mode="content", selector="tensor-history-v1"),
               Region(1, selector="tensor-history-v1")), (0, 1), (0, 1))
    m = Model(g, width=1, dtype=dtype)
    with torch.no_grad():
        for w in m.nodes:
            w.decay.zero_(); w.read.fill_(1); w.weight.fill_(.2); w.bias.zero_()
        for p in [*m.input_scale, *m.output_scale]:
            p.fill_(1)
        m.regions[0].alpha.fill_(.5); m.regions[0].bias.copy_(torch.tensor([.25, -.25, .125], dtype=dtype))
    leaves = dict(m.named_parameters()); history = {}; xs = []
    for b in range(3):
        h = torch.tensor(2. if b < 2 else 7., dtype=dtype, requires_grad=True)
        leaves[f"history.{b}"] = h
        history[b, 0] = History(node_maps={"selected": {}}, tensors={"memory": h})
        if b == 2:
            continue
        for v, values in enumerate(((1., 0.), (3., 2.))):
            for i, value in enumerate(values):
                x = torch.tensor([value], dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{v}.{i}"] = x; xs.append(External(b, v, i, (1, 4)[i], x))
    return g, m, Continuation(g.identity, 3, history=history), xs, leaves


def linear_vjp(root, leaves):
    return dict(zip(leaves, torch.autograd.grad(root.sum(), list(leaves.values()), allow_unused=True, retain_graph=True)))
