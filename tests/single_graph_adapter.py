"""Independent bounded LH fixture embedding; no native scheduler calls here."""
from dataclasses import replace
import torch
from tidegraph import Edge, Graph, SourceDomain, StateClock
from tidegraph.ops import Model
from tidegraph.ports import PortLayout
from tidegraph.records import Result
from tidegraph.blocks import canonicalize


class SingleGraph:
    def __init__(self, body, model, readout, read_model, layers):
        assert layers > 0 and len(body.outputs) == 1 and not body.origins
        assert len(readout.nodes) == len(readout.regions) == 1 and not readout.edges
        assert readout.inputs == (0,)*layers and readout.outputs == (0,)
        assert all(n.state_clock == StateClock() and not n.emit_phases for n in body.nodes)
        assert all(n.emission == "slot_affine" for n in body.nodes)
        self.body, self.readout = body, readout
        self.body_clock, self.read_clock = StateClock(layers+1, 0, layers), StateClock(layers+1, layers, 1)
        self.n, self.regions = len(body.nodes), len(body.regions)
        incoming, outgoing = [0]*(self.n+1), [0]*(self.n+1)
        edges, sources, targets, domains, edge_scale, agg_scale = [], [], [], [], [], []
        self.edge_origin = []; self.output_origin = [[] for _ in body.nodes]
        phases = [[] for _ in body.nodes]

        def add(source, target, delay, phase, old_slot, logical, origin, send, receive):
            edges.append(Edge(source, target, delay)); self.edge_origin.append(origin)
            sources.append(outgoing[source]); outgoing[source] += 1
            targets.append(incoming[target]); incoming[target] += 1
            domains.append(logical); phases[source].append(phase)
            self.output_origin[source].append(old_slot)
            edge_scale.append(send); agg_scale.append(receive)

        # Phase-major physical IDs differ from the C++ edge-major construction.
        for phase in range(layers):
            for e, edge in enumerate(body.edges):
                assert edge.delay == 1
                add(edge.source, edge.target, 2 if phase == layers-1 else 1, phase,
                    body.ports.edge_source[e], body.domain.edge_target[e], e,
                    model.edge_scale[e], model.agg_scale[e])
        for phase in range(layers):
            add(body.outputs[0], self.n, layers-phase, phase, body.ports.output[0],
                readout.domain.input[phase], -1, model.output_scale[0], read_model.input_scale[phase])
        inputs = []
        for node in body.inputs:
            inputs.append(incoming[node]); incoming[node] += 1
        nodes = [replace(n, state_clock=self.body_clock, emit_period=layers+1, emit_phases=phases[v])
                 for v, n in enumerate(body.nodes)]
        nodes.append(replace(readout.nodes[0], region=self.regions, state_clock=self.read_clock))
        self.graph = Graph(tuple(nodes), tuple(edges), body.regions+readout.regions, body.inputs, (self.n,),
                           PortLayout(sources, targets, inputs, (0,)),
                           source_domain=SourceDomain(domains, body.domain.input))
        assert self.graph.source_counts == body.source_counts+readout.source_counts
        self.model = Model(self.graph, width=model.width, dtype=model.nodes[0].bias.dtype)
        for v, (old, new) in enumerate(zip([*model.nodes, *read_model.nodes], self.model.nodes)):
            for name in ("decay", "weight", "bias", "read"):
                setattr(new, name, getattr(old, name))
            extra = {k: p for k, p in old.extra.items() if not k.startswith(("emit_w_", "emit_b_"))}
            if v < self.n:
                for slot, origin in enumerate(self.output_origin[v]):
                    for prefix in ("emit_w_", "emit_b_"):
                        extra[f"{prefix}{slot}"] = old.extra[f"{prefix}{origin}"]
            else:
                extra = dict(old.extra.items())
            new.extra = torch.nn.ParameterDict(extra)
        self.model.regions = torch.nn.ModuleList([*model.regions, *read_model.regions])
        self.model.input_scale = model.input_scale; self.model.output_scale = read_model.output_scale
        self.model.edge_scale = torch.nn.ParameterList(edge_scale)
        self.model.agg_scale = torch.nn.ParameterList(agg_scale)

    def inputs(self, xs):
        return [replace(x, time=self.body_clock.to_global(x.time)) for x in xs]

    def body_atom(self, a):
        assert not a.kind or self.edge_origin[a.source] >= 0
        return replace(a, time=self.body_clock.to_local(a.time),
                       position=self.body_clock.to_local(a.position) if a.kind else a.position,
                       source=self.edge_origin[a.source] if a.kind else a.source)

    def body_view(self, result):
        clock = self.body_clock; q = result.continuation
        def history(h):
            return replace(h, last_time=clock.to_local(h.last_time))
        projected = replace(q, identity=self.body.identity, cut=clock.cut(q.cut),
                            states={o: clock.local_state(s) for o, s in q.states.items() if o[1] < self.n},
                            history={o: history(h) for o, h in q.history.items() if o[1] < self.regions},
                            pending=[self.body_atom(a) for a in q.pending if self.edge_origin[a.source] >= 0],
                            ledger={o: (p, clock.to_local(t)) for o, (p, t) in q.ledger.items()})
        events = []
        for e in result.trace:
            if e["node"] == self.n:
                continue
            event = dict(e, time=clock.to_local(e["time"]), history=history(e["history"]),
                         fiber=sorted((self.body_atom(a) for a in e["fiber"]), key=lambda a: a.key()))
            if "emitted" in e:
                pairs = [(self.output_origin[e["node"]][s], value) for s, value in e["emitted"].items()]
                assert len(set(s for s, _ in pairs)) == len(pairs)
                event["emitted"] = dict(sorted(pairs))
            events.append(event)
        outputs = [(a.batch, clock.to_local(a.position), 0, a.value)
                   for a in result.messages if self.edge_origin[a.source] < 0]
        messages = [self.body_atom(a) for a in result.messages if self.edge_origin[a.source] >= 0]
        return canonicalize(self.body, Result(projected, events, outputs, messages, {}))

    def read_view(self, q):
        """State/history quotient: deliberately does not reconstruct an input ledger."""
        def history(h):
            return replace(h, last_time=self.read_clock.to_local(h.last_time),
                           node_maps={name: {v-self.n: x for v, x in values.items()}
                                      for name, values in h.node_maps.items()})
        return replace(q, identity=self.readout.identity, cut=self.read_clock.cut(q.cut),
                       states={(b, 0): self.read_clock.local_state(s) for (b, v), s in q.states.items() if v == self.n},
                       history={(b, 0): history(h) for (b, r), h in q.history.items() if r == self.regions},
                       ledger={}, pending=[])

    def buffer(self, q):
        return sorted(((a.batch, self.body_clock.to_local(a.position), 0, a.value)
                       for a in q.pending if self.edge_origin[a.source] < 0), key=lambda o: (o[1], o[0], o[2]))

    def outputs(self, result):
        return [(b, self.read_clock.to_local(t), p, value) for b, t, p, value in result.outputs]
