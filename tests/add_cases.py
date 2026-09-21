"""Lazy Add fixtures with independent leaves, irregular clocks and full fibers."""
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.lazy_add import decode
from tidegraph.ops import Model

PROFILE = "lh-add-repeat-v1"


def fixture(dtype, policy="all", cyclic=False):
    g = Graph(tuple(Node(0 if v < 2 else 1, memory=PROFILE, clear=policy == "clear",
                         next_state="control-blend-v1" if policy == "blend" else "adopt-v1",
                         readout="norm-fp64-v1") for v in range(3)),
              (Edge(0, 2, 2), Edge(1, 2, 5)) + ((Edge(2, 0, 1),) if cyclic else ()),
              (Region(1, observe_all=policy != "selected", read_mode="old" if policy == "old" else "proposal",
                      selector="lh-count-affect-v1"), Region(1)), (0, 1, 0), (0, 1, 2))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        for v, w in enumerate(m.nodes):
            w.extra["add_retention"].fill_(.71 + .04*v)
    leaves = dict(m.named_parameters()); q = Continuation(g.identity, 3); xs = []
    for b in range(3):
        for v in range(3):
            value = torch.tensor([.3+b/10, -.2-v/20], dtype=dtype, requires_grad=True)
            leaves[f"initial.{b}.{v}"] = value; q.states[b, v] = State(value)
    for b, times in enumerate(((2, 5, 6), (0, 4))):
        for p in range(3):
            for pos, t in enumerate(times):
                x = torch.tensor([.2 + t/20+p/10, -.1+b/10], dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{p}.{pos}"] = x; xs.append(External(b, p, pos, t, x))
    return g, m, q, xs, leaves


def native_decode(weights, state, cut):
    import _tide_native as core
    w = core.NodeWeights(weights.decay, weights.weight, weights.bias, weights.read)
    w.extra = dict(weights.extra.items())
    s = core.State(state.value, state.last_time, state.observations, state.slots)
    return core.decode_add_repeat(w, s, cut)


def physical(result, model, native=False):
    q = result.continuation
    fn = native_decode if native else decode
    return {owner: fn(model.nodes[owner[1]], state, q.cut) for owner, state in q.states.items()}
