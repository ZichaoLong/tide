"""Physical phase aliases with an independent unsplit graph as numerical anchor."""
from dataclasses import replace
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, SourceDomain
from tidegraph.blocks import canonicalize
from tidegraph.ops import Model

PROFILES = ("sum", "mean", "weighted_mean", "active_softmax", "all_softmax",
            "fiber-linear", "fiber-active-softmax", "fiber-all-softmax")


def fixture(dtype, profile="all_softmax", *, cyclic=False, clear=False):
    memory = f"lh-fiber-attention-{profile[6:]}-repeat-v1" if profile.startswith("fiber-") else "ema"
    aggregation = "sum" if profile.startswith("fiber-") else profile
    nodes = (Node(0), Node(0), Node(1, memory=memory, aggregation=aggregation, clear=clear,
                                   query_heads=2, kv_heads=2))
    edges = (Edge(0, 2, 2), Edge(1, 2, 2)) + ((Edge(2, 0, 3),) if cyclic else ())
    base = Graph(nodes, edges, (Region(2), Region(1)), (0, 1, 2), (0, 1, 2))
    # Deliberately permuted aliases: physical ID order differs from logical slots.
    mapping = (1, 0, 0, 1) + ((2,) if cyclic else ())
    domain = SourceDomain(tuple(base.domain.edge_target[e] for e in mapping), base.domain.input)
    encoded = replace(base, nodes=(replace(nodes[0], emit_period=2, emit_phases=(0, 1, -1)),
                                   replace(nodes[1], emit_period=2, emit_phases=(1, 0, -1)), nodes[2]),
                      edges=tuple(edges[e] for e in mapping), source_domain=domain)
    model, em = Model(base, width=4, dtype=dtype), Model(encoded, width=4, dtype=dtype)
    # Programs retain their own emission policy; parameters share by the explicit
    # logical mapping, including physical receive/send scales. No implicit tying.
    for a, b in zip(em.nodes, model.nodes):
        for name in ("decay", "weight", "bias", "read", "extra"):
            setattr(a, name, getattr(b, name))
    em.regions = model.regions; em.input_scale = model.input_scale; em.output_scale = model.output_scale
    em.agg_scale = torch.nn.ParameterList(model.agg_scale[e] for e in mapping)
    em.edge_scale = torch.nn.ParameterList(model.edge_scale[e] for e in mapping)
    if profile.startswith("fiber-"):
        with torch.no_grad():
            model.nodes[2].extra["fiber_pool"].copy_(torch.tensor([.1, .7, -.3], dtype=dtype))
    leaves = dict(model.named_parameters()); states, xs = {}, []
    for b in range(3):
        for v, w in enumerate(model.nodes):
            state = w.initial(); state.value = torch.full_like(state.value, .1+b/20, requires_grad=True)
            states[b, v] = state; leaves[f"initial.{b}.{v}"] = state.value
        for port in range(3):
            times = (3,) if port == 2 and b == 0 else (6,) if port == 2 and b == 1 else () if port == 2 else (
                (1,) if b == 1 and port == 1 else (0, 1, 4, 6))
            for pos, time in enumerate(times):
                x = torch.sin(torch.arange(4, dtype=dtype)/5 + (b+port+time)/7)
                if (b, port, time) == (2, 0, 4):
                    x.zero_()  # Present zero still creates a source row.
                x.requires_grad_(); leaves[f"input.{b}.{port}.{time}"] = x
                xs.append(External(b, port, pos, time, x))
    q = Continuation(base.identity, 3, states=states)
    return base, model, q, encoded, em, replace(q.fork(), identity=encoded.identity), xs, leaves, mapping


def project(result, base, encoded, mapping):
    def atom(a):
        return replace(a, source=mapping[a.source]) if a.kind else a
    events = []
    for event in result.trace:
        outputs = {}
        for slot, value in event.get("emitted", {}).items():
            kind, physical = encoded.port_indexes[1].row(event["node"])[slot]
            logical = base.ports.edge_source[mapping[physical]] if kind else base.ports.output[physical]
            assert logical not in outputs
            outputs[logical] = value
        e = dict(event, fiber=sorted((atom(a) for a in event["fiber"]), key=lambda a: a.key()))
        if "emitted" in event:
            e["emitted"] = dict(sorted(outputs.items()))
        events.append(e)
    q = replace(result.continuation, identity=base.identity, pending=[atom(a) for a in result.continuation.pending])
    return canonicalize(base, replace(result, continuation=q, trace=events, messages=[atom(a) for a in result.messages]))
