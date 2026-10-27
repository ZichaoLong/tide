"""Core-5 Settle §2.4, §9–11: general ports and an independent region schedule.

The time code is a domain restriction. Embedding changes neither graph nor
records. This module never runs a reference scheduler to obtain a candidate plan.
Contract and independent anchors: docs/settle-embedding.md.
"""
from collections import defaultdict
from .blocks import canonicalize, deliver, evaluate_block
from .coordinates import integers
from .records import External, Result
from .validation import validate_window


def port_external(spec, values, positions, port, batch, start_occurrence):
    positions = tuple(positions)
    integers("SettleGraph port schedule", port, batch, start_occurrence, *positions)
    if spec.binding != "ports":
        raise ValueError("port schedule requires ports binding")
    if (values.ndim != 2 or len(values) != len(positions)
            or not 0 <= port < len(spec.graph.inputs) or batch < 0
            or start_occurrence < 0 or start_occurrence + len(positions) >= 2**63):
        raise ValueError("invalid SettleGraph port schedule shape/owner/occurrence")
    rank = spec.rank(spec.graph.inputs[port])
    if (any(t < 0 or t > (2**63-1-rank)//spec.stride for t in positions)
            or any(a >= b for a, b in zip(positions, positions[1:]))):
        raise ValueError("port schedule requires increasing representable positions")
    return [External(batch, port, start_occurrence+k, spec.stride*t+rank, values[k])
            for k, t in enumerate(positions)]


def validate_alignment(spec, q, external):
    """Reject coordinates outside the fixed LegalTimeCode before execution."""
    def aligned(time, node):
        if time < 0 or time % spec.stride != spec.rank(node):
            raise ValueError("record violates SettleGraph time code")
    for row in external:
        aligned(row.time, spec.graph.inputs[row.port])
    for (_, port), (_, time) in q.ledger.items():
        aligned(time, spec.graph.inputs[port])
    for a in q.pending:
        aligned(a.time, a.node)
        aligned(a.position, spec.graph.edges[a.source].source)
    for (_, node), state in q.states.items():
        if state.last_time != -1:
            aligned(state.last_time, node)
    for (_, region), history in q.history.items():
        if history.last_time != -1 and history.last_time % spec.stride != spec.ranks[region]:
            raise ValueError("history violates SettleGraph time code")


def run_ports(spec, model, continuation, external, stop, *, sealed_until,
              mode="hard", zeta=1.0, prefill=True, packed=True,
              full_autograd="replay", aggregate_autograd="replay"):
    """Exact independent region-major schedule, including intra-position cuts.

    Readiness follows strict region ranks and sealed external inputs; only
    actual nonempty fibers become frames. Sparse logical time is never scanned.
    ``prefill=False`` uses scalar frames with the same dependency order.
    """
    from .block_policy import validate
    if spec.binding != "ports":
        raise ValueError("run_ports requires ports binding")
    if mode not in {"hard", "hst", "softp"}:
        raise ValueError("invalid emit mode")
    policy = dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd)
    validate(model, **policy)
    external = list(external)
    g = spec.graph
    atoms, ledger = validate_window(g, model, continuation, external, stop, sealed_until)
    validate_alignment(spec, continuation, external)
    q = continuation.fork(); q.ledger = ledger
    fibers = defaultdict(list)
    for a in [*q.pending, *atoms]:
        fibers[a.batch, a.node, a.time].append(a)
    events, messages, outputs = [], [], []
    stats = {"region_blocks": 0}
    for region in sorted(range(len(g.regions)), key=lambda r: (spec.ranks[r], r)):
        ready = defaultdict(set)
        for b, node, time in fibers:
            if g.nodes[node].region == region and q.cut <= time < stop:
                ready[time, b].add(node)
        frames = [(b, region, time, members) for (time, b), members in sorted(ready.items())]
        groups = [frames] if prefill and frames else [[frame] for frame in frames]
        for group in groups:
            block, counters = evaluate_block(g, model, q, group, fibers, mode=mode, zeta=zeta,
                                             prefill=prefill, **policy)
            deliver(g, model, block, fibers, messages, outputs, packed=packed, stats=stats)
            events.extend(block); stats["region_blocks"] += 1
            for name, value in counters.items():
                stats[name] = max(stats.get(name, 0), value) if name.startswith("max_") else stats.get(name, 0)+value
    q.pending = [a for bucket in fibers.values() for a in bucket if a.kind == 1 and a.time >= stop]
    q.cut = stop
    return canonicalize(g, Result(q, events, outputs, messages, stats))


def decode_slots(spec, result, start_position, stop_position):
    """§6.1/§10.3: decode produced records; absent coordinates are None (⊥).

    Caller supplies the complete-position interval represented by ``result``.
    No values or state are recomputed. Tensor handles retain their VJP roots.
    """
    integers("SettleGraph slot interval", start_position, stop_position)
    if (spec.binding != "ports" or result.continuation.identity != spec.graph.identity
            or not 0 <= start_position <= stop_position
            or result.continuation.cut != stop_position*spec.stride or result.continuation.pending):
        raise ValueError("slot decoding requires a complete position interval")
    edges = {(b, t, e): None for b in range(result.continuation.batch_size)
             for t in range(start_position, stop_position) for e in range(len(spec.graph.edges))}
    ports = {(b, t, p): None for b in range(result.continuation.batch_size)
             for t in range(start_position, stop_position) for p in range(len(spec.graph.outputs))}
    for atom in result.messages:
        key = atom.batch, atom.position//spec.stride, atom.source
        if key not in edges or edges[key] is not None:
            raise ValueError("message outside interval or duplicate edge slot")
        edges[key] = atom.value
    for b, time, port, value in result.outputs:
        key = b, time//spec.stride, port
        if key not in ports or ports[key] is not None:
            raise ValueError("output outside interval or duplicate port slot")
        ports[key] = value
    return edges, ports
