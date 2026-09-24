"""Packed Python tick scheduler; scalar reference.py remains an independent oracle."""
from collections import defaultdict
import heapq
from .block_policy import validate
from .blocks import evaluate_block, deliver, canonicalize
from .records import Result
from .validation import validate_window


def run(graph, model, initial, external, stop, *, sealed_until, mode="hard", zeta=1.0,
        trace=True, packed=True, full_autograd="replay", aggregate_autograd="replay"):
    policy = dict(packed=packed, full_autograd=full_autograd, aggregate_autograd=aggregate_autograd)
    validate(model, **policy)
    if mode not in {"hard", "hst", "softp"}:
        raise ValueError("invalid emit mode")
    atoms, ledger = validate_window(graph, model, initial, external, stop, sealed_until)
    q = initial.fork(); q.ledger = ledger
    fibers, scheduled = defaultdict(list), defaultdict(set)
    times, queued = [], set()

    def schedule(atom):
        if atom.time < stop:
            scheduled[atom.time].add((atom.batch, graph.nodes[atom.node].region, atom.node))
            if atom.time not in queued:
                heapq.heappush(times, atom.time); queued.add(atom.time)

    for atom in list(q.pending)+atoms:
        fibers[atom.batch, atom.node, atom.time].append(atom)
        schedule(atom)
    events, messages, outputs, stats = [], [], [], {}
    while times:
        time = heapq.heappop(times)
        groups = defaultdict(lambda: defaultdict(set))
        for b, r, v in scheduled.pop(time):
            groups[r][b].add(v)
        for region, samples in sorted(groups.items()):
            frames = [(b, region, time, nodes) for b, nodes in sorted(samples.items())]
            block, counters = evaluate_block(graph, model, q, frames, fibers, mode=mode, zeta=zeta,
                                              prefill=False, **policy)
            sent = []
            deliver(graph, model, block, fibers, sent, outputs)
            for atom in sent:
                schedule(atom)
            if trace:
                events.extend(block); messages.extend(sent)
            stats["visited_edges"] = stats.get("visited_edges", 0)+len(sent)
            for name, value in counters.items():
                stats[name] = max(stats.get(name, 0), value) if name.startswith("max_") else stats.get(name, 0)+value
            for e in block:
                fibers.pop((e["batch"], e["node"], time))
    q.pending = [a for xs in fibers.values() for a in xs if a.kind == 1 and a.time >= stop]
    q.cut = stop
    return canonicalize(graph, Result(q, events, outputs, messages, stats))
