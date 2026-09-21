"""Local-time anchors and an independent projection of global event records."""
from dataclasses import replace
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.blocks import canonicalize
from tidegraph.ops import Model

PROFILES = ("ema", "ssm", "linear", "delta", "attention", "lh-add-repeat-v1",
            "lh-fiber-attention-all-softmax-repeat-v1")


def fixture(dtype, clock, memory="lh-add-repeat-v1", policy="all"):
    node = Node(0, memory=memory, query_heads=2, kv_heads=2 if "fiber" in memory else 1,
                window=3 if memory == "attention" else 0, clear=policy == "clear")
    base = Graph((node, node), (), (Region(1, observe_all=policy != "selected", count_priority=False),),
                 (0, 0, 1, 1), (0, 1))
    encoded = replace(base, nodes=tuple(replace(n, state_clock=clock) for n in base.nodes))
    m, em = Model(base, width=4, dtype=dtype), Model(encoded, width=4, dtype=dtype)
    m.nodes[1] = m.nodes[0]; em.nodes[1] = em.nodes[0]
    for name in ("decay", "weight", "bias", "read", "extra"):
        setattr(em.nodes[0], name, getattr(m.nodes[0], name))
    for name in ("regions", "input_scale", "agg_scale", "edge_scale", "output_scale"):
        setattr(em, name, getattr(m, name))
    if "add_retention" in m.nodes[0].extra:
        with torch.no_grad():
            m.nodes[0].extra["add_retention"].fill_(.83)
    leaves = dict(m.named_parameters()); states, xs = {}, []
    for b in range(3):
        for v, w in enumerate(m.nodes):
            state = w.initial(); state.value = torch.full_like(state.value, .07*(1+b), requires_grad=True)
            state.slots = {name: torch.full((1, *t.shape[1:]) if name in {"key", "value", "log_bias"} else t.shape,
                                           .03*(b+1), dtype=dtype, requires_grad=True) for name, t in state.slots.items()}
            if "key" in state.slots:
                state.observations = 1
            states[b, v] = state; leaves[f"initial.{b}.{v}"] = state.value
            leaves.update({f"initial.{b}.{v}.{name}": t for name, t in state.slots.items()})
        for port in range(4):
            times = (0, 1, 4, 6) if port % 2 == 0 else (1, 4) if b == 0 else ()
            if b == 2:
                times = tuple(t for t in times if t != 1)
            for pos, time in enumerate(times):
                value = torch.sin(torch.arange(4, dtype=dtype)/6+(port+b+time)/8).requires_grad_()
                leaves[f"input.{b}.{port}.{time}"] = value
                xs.append(External(b, port, pos, time, value))
    q = Continuation(base.identity, 3, states=states)
    eq = replace(q.fork(), identity=encoded.identity)
    return base, m, q, encoded, em, eq, xs, [replace(x, time=clock.to_global(x.time)) for x in xs], leaves


def project(result, base, clock, edge_map=None, emit_map=None):
    def atom(a):
        return replace(a, time=clock.to_local(a.time),
                       position=clock.to_local(a.position) if a.kind else a.position,
                       source=edge_map[a.source] if a.kind and edge_map is not None else a.source)
    def history(h):
        return replace(h, last_time=clock.to_local(h.last_time))
    q = result.continuation
    projected = replace(q, identity=base.identity, cut=clock.cut(q.cut),
                        states={o: clock.local_state(s) for o, s in q.states.items()},
                        history={o: history(h) for o, h in q.history.items()},
                        pending=[atom(a) for a in q.pending],
                        ledger={o: (p, clock.to_local(t)) for o, (p, t) in q.ledger.items()})
    events = []
    for e in result.trace:
        event = dict(e, time=clock.to_local(e["time"]), history=history(e["history"]),
                     fiber=sorted((atom(a) for a in e["fiber"]), key=lambda a: a.key()))
        if emit_map is not None and "emitted" in e:
            mapped = [(emit_map[s], v) for s, v in e["emitted"].items()]
            assert len(set(s for s, _ in mapped)) == len(mapped)
            event["emitted"] = dict(sorted(mapped))
        events.append(event)
    outputs = [(b, clock.to_local(t), p, value) for b, t, p, value in result.outputs]
    return canonicalize(base, replace(result, continuation=projected, trace=events, outputs=outputs,
                                      messages=[atom(a) for a in result.messages]))


def loss_grad(result, leaves):
    from tidegraph.compare import objective
    return dict(zip(leaves, torch.autograd.grad(objective(result), list(leaves.values()),
                                                allow_unused=True, retain_graph=True)))
