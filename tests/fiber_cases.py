"""Ragged source fibers, independently differentiable caches and tick clocks."""
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.fiber_attention import PROFILE, decode_bias
from tidegraph.ops import Model


def fixture(dtype, policy="all", cyclic=False, full="tanh"):
    g = Graph(tuple(Node(0 if v < 2 else 1, memory=PROFILE, query_heads=2, kv_heads=2,
                         clear=policy == "clear", full=full) for v in range(3)),
              (Edge(0, 2, 2), Edge(1, 2, 5)) + ((Edge(2, 0, 1),) if cyclic else ()),
              (Region(1, observe_all=policy != "selected", read_mode="old" if policy == "old" else "proposal"),
               Region(1)), (0, 1, 0, 1, 2), (0, 1, 2))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 3)
    leaves = dict(m.named_parameters()); xs = []
    for b in range(3):
        for v, w in enumerate(m.nodes):
            rows = (b+v) % 3
            state = State(torch.full((4,), .1+v/20, dtype=dtype, requires_grad=True), slots={
                "key": torch.full((rows, 2, 2), .1+b/10, dtype=dtype, requires_grad=True),
                "value": (torch.arange(rows*4, dtype=dtype).reshape(rows, 2, 2)/20+.3).requires_grad_(),
                "log_bias": torch.full((rows,), -.2-v/10, dtype=dtype, requires_grad=True)})
            q.states[b, v] = state
            leaves[f"initial.{b}.{v}.value"] = state.value
            leaves.update({f"initial.{b}.{v}.{k}": t for k, t in state.slots.items()})
    for b, times in enumerate(((2, 5, 6), (0, 4))):
        for p in range(5):
            for pos, time in enumerate(times if p < 2 else times[::2]):
                x = torch.sin(torch.arange(4, dtype=dtype)*.2+time/10+p/3+b).requires_grad_()
                if p == 3 and pos == 0:
                    x = torch.zeros(4, dtype=dtype, requires_grad=True)
                leaves[f"input.{b}.{p}.{pos}"] = x; xs.append(External(b, p, pos, time, x))
    return g, m, q, xs, leaves


def native_decode(w, state, cut):
    import _tide_native as core
    nw = core.NodeWeights(w.decay, w.weight, w.bias, w.read); nw.extra = dict(w.extra.items())
    return core.decode_fiber_bias(nw, core.State(state.value, state.last_time, state.observations, state.slots), cut)


def physical(result, model, native=False):
    fn = native_decode if native else decode_bias; q = result.continuation
    return {owner: fn(model.nodes[owner[1]], state, q.cut) for owner, state in q.states.items()}


def scalar_model(dtype):
    g = Graph((Node(0, memory=PROFILE),), (), (Region(1),), (0, 0), (0,))
    m = Model(g, width=1, dtype=dtype); w = m.nodes[0]
    with torch.no_grad():
        w.extra["fiber_qkv"].fill_(1); w.extra["fiber_out"].fill_(1)
        for scale in m.input_scale:
            scale.fill_(1)
    return g, m, w
