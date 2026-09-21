import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.ops import Model


def blend_fixture(dtype, *, clear=False, observe_all=True):
    g = Graph(tuple(Node(0, clear=clear and v == 0, next_state="control-blend-v1") for v in range(3)), (),
              (Region(1, observe_all=observe_all, count_priority=False, read_mode="content"),), (0, 1), (0, 1))
    m = Model(g, width=1, dtype=dtype)
    with torch.no_grad():
        for w in m.nodes:
            w.decay.zero_(); w.read.zero_(); w.weight.fill_(.2); w.bias.zero_()
        for p in [*m.input_scale, *m.output_scale]:
            p.fill_(1)
    leaves = dict(m.named_parameters()); states = {}; xs = []
    for b in range(2):
        for v, value in enumerate((2., 4., 7.)):
            initial = torch.tensor([value], dtype=dtype, requires_grad=True)
            leaves[f"initial.{b}.{v}"] = initial; states[b, v] = State(initial)
        for v, values in enumerate(((1., 2.), (3., 0.))):
            for i, value in enumerate(values):
                x = torch.tensor([value], dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{v}.{i}"] = x; xs.append(External(b, v, i, (1, 4)[i], x))
    return g, m, Continuation(g.identity, 2, states=states), xs, leaves
