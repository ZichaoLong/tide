"""Read-mode fixtures with separated input/state leaves and a permanently idle node."""
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.frontier import run as frontier
from tidegraph.native import Native


def fixture(dtype, read_mode, *, observe_all=True, clear=False):
    g = Graph(tuple(Node(0, clear=clear) for _ in range(4)), (),
              (Region(1, observe_all=observe_all, count_priority=False, read_mode=read_mode),),
              (0, 1, 2), (0, 1, 2))
    m = Model(g, width=1, dtype=dtype)
    with torch.no_grad():
        for w in m.nodes:
            w.decay.zero_(); w.read.fill_(1); w.weight.fill_(0.2); w.bias.zero_()
        for scale in [*m.input_scale, *m.output_scale]:
            scale.fill_(1)
    leaves = dict(m.named_parameters())
    states, xs = {}, []
    for b in range(2):
        for v, initial in enumerate((4., 0., 3., 7.)):
            value = torch.tensor([initial], dtype=dtype, requires_grad=True)
            leaves[f"initial.{b}.{v}"] = value
            states[b, v] = State(value)
        for p, values in enumerate(((0., 2.), (2., 0.), (1., 0.))):
            for i, value in enumerate(values):
                x = torch.tensor([value], dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{p}.{i}"] = x
                xs.append(External(b, p, i, (2, 5)[i], x))
    return g, m, Continuation(g.identity, 2, states=states), xs, leaves


def execute(implementation, g, m, q, xs, stop=6, mode="hst"):
    if implementation == "reference":
        return run(g, m, q, xs, stop, sealed_until=stop, mode=mode)
    if implementation.startswith("python"):
        return frontier(g, m, q, xs, stop, sealed_until=stop, mode=mode,
                        prefill=implementation != "python-causal")
    algorithm = "frontier" if "frontier" in implementation else "streaming"
    engine = Native(g, m, algorithm=algorithm, workers=1 if implementation == "native-serial" else 3,
                    packed=implementation != "native-serial", prefill=implementation != "native-causal", mode=mode)
    return engine.run(q, xs, stop, sealed_until=stop)
