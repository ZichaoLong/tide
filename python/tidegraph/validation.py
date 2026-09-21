"""Fail early on malformed windows, states, tensor shapes and message identity."""
import torch
from .records import Atom


def validate_window(graph, model, continuation, external, stop, sealed_until):
    from .full import validate_program
    from .aggregate import validate_program as validate_aggregate
    offsets = graph.port_indexes[1].offsets
    incoming = graph.port_indexes[0].offsets
    for node, spec in enumerate(graph.nodes):
        validate_program(model.nodes[node], spec, offsets[node+1] - offsets[node])
        validate_aggregate(model.nodes[node], spec, incoming[node+1] - incoming[node])
    q = continuation
    if q.identity != graph.identity or q.batch_size < 1:
        raise ValueError("continuation identity or batch size mismatch")
    if not 0 <= q.cut <= stop <= sealed_until < 2**63:
        raise ValueError("window requires monotonic cut and explicit input seal")
    reference = model.nodes[0].bias
    def tensor(x):
        if x.device.type != "cpu" or x.dtype != reference.dtype or x.shape != (model.width,):
            raise ValueError("incompatible tensor device, dtype or shape")
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite input/state")
    for (b, v), state in q.states.items():
        if not 0 <= b < q.batch_size or not 0 <= v < len(graph.nodes):
            raise ValueError("invalid state owner")
        if not -1 <= state.last_time < q.cut or state.observations < 0:
            raise ValueError("invalid state clock")
        tensor(state.value)
        model.nodes[v].validate(state)
        for value in state.slots.values():
            if value.device.type != "cpu" or value.dtype != reference.dtype or not torch.isfinite(value).all():
                raise ValueError("incompatible state slot dtype/device/value")
    for (b, r), counts in q.history.items():
        if not 0 <= b < q.batch_size or not 0 <= r < len(graph.regions):
            raise ValueError("invalid history owner")
        if any(v < 0 or v >= len(graph.nodes) or graph.nodes[v].region != r or c < 0
               for v, c in counts.items()):
            raise ValueError("invalid selector history")
    ledger = dict(q.ledger)
    for (b, p), (position, time) in ledger.items():
        if not (0 <= b < q.batch_size and 0 <= p < len(graph.inputs)
                and position >= 0 and 0 <= time < q.cut):
            raise ValueError("invalid input ledger")
    atoms = []
    for x in sorted(external, key=lambda e: (e.batch, e.port, e.position)):
        if not (0 <= x.batch < q.batch_size and 0 <= x.port < len(graph.inputs)
                and x.position >= 0 and q.cut <= x.time < stop):
            raise ValueError("invalid external coordinate")
        old_position, old_time = ledger.get((x.batch, x.port), (-1, -1))
        if x.position != old_position + 1 or x.time <= old_time:
            raise ValueError("port positions must be contiguous and times strictly increase")
        tensor(x.value)
        ledger[x.batch, x.port] = x.position, x.time
        atoms.append(Atom(x.batch, graph.inputs[x.port], x.time, 0, x.port, x.position, x.value))
    seen = set()
    for a in q.pending:
        if a.kind != 1 or not 0 <= a.source < len(graph.edges):
            raise ValueError("invalid pending edge")
        e = graph.edges[a.source]
        key = a.batch, a.source, a.position
        if (key in seen or not 0 <= a.batch < q.batch_size or a.node != e.target
                or not 0 <= a.position < q.cut or a.time != a.position + e.delay
                or not q.cut <= a.time < 2**63):
            raise ValueError("invalid pending message coordinate")
        seen.add(key)
        tensor(a.value)
    return atoms, ledger
