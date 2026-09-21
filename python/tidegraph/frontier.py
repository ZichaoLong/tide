"""Finite-input TimedDAG frontier relative to the region-block contract.

Potential event domains provide absence/completeness certificates. They do not
depend on route replay. General cycles use the streaming interpreter instead.
"""
from collections import defaultdict
from .blocks import canonicalize, deliver, evaluate_block
from .records import Result
from .validation import validate_window


def plan(graph, atoms, start, stop, max_events=1000000):
    order = graph.topological_order()
    domains = [set() for _ in graph.nodes]
    total = 0
    def add(v, batch, time):
        nonlocal total
        if start <= time < stop and (batch, time) not in domains[v]:
            domains[v].add((batch, time)); total += 1
            if total > max_events:
                raise ValueError("frontier potential-event limit exceeded; use streaming or a smaller window")
    for a in atoms:
        add(a.node, a.batch, a.time)
    offsets, edges = graph.adjacency()
    for v in order:
        for batch, time in domains[v]:
            for edge_id in edges[offsets[v]:offsets[v + 1]]:
                edge = graph.edges[edge_id]
                add(edge.target, batch, time + edge.delay)
    frames, dependencies = defaultdict(set), defaultdict(set)
    for v, domain in enumerate(domains):
        for batch, time in domain:
            frames[batch, graph.nodes[v].region, time].add(v)
    for edge in graph.edges:
        for batch, time in domains[edge.source]:
            target = batch, graph.nodes[edge.target].region, time + edge.delay
            if target in frames:
                dependencies[target].add((batch, graph.nodes[edge.source].region, time))
    return frames, dependencies


def run(graph, model, continuation, external, stop, *, sealed_until, mode="hard", zeta=1.0,
        trace=True, prefill=True, max_events=1000000):
    if mode not in {"hard", "hst", "softp"}:
        raise ValueError("invalid emit mode")
    inputs, ledger = validate_window(graph, model, continuation, external, stop, sealed_until)
    q = continuation.fork(); q.ledger = ledger
    atoms = list(q.pending) + inputs
    frames, dependencies = plan(graph, atoms, q.cut, stop, max_events)
    owners, fibers = defaultdict(list), defaultdict(list)
    for key in sorted(frames):
        owners[key[:2]].append(key)
    for a in atoms:
        fibers[a.batch, a.node, a.time].append(a)
    cursor = {owner: 0 for owner in owners}
    done, events, outputs, messages = set(), [], [], []
    stats = {"frontier_stages": 0, "region_blocks": 0, "state_blocks": 0, "state_steps": 0, "full_blocks": 0}
    while len(done) < len(frames):
        blocks = []
        for owner, keys in owners.items():
            block = []
            index = cursor[owner]
            while index < len(keys) and dependencies[keys[index]] <= done:
                block.append(keys[index]); index += 1
            if block:
                blocks.append((owner, block))
        if not blocks:
            raise RuntimeError("frontier failed to make progress")
        stats["frontier_stages"] += 1
        region_blocks = defaultdict(list)
        for owner, keys in blocks:
            region_blocks[owner[1]].extend(keys)
        for keys in region_blocks.values():
            ready = [(b, r, t, frames[b, r, t]) for b, r, t in sorted(keys, key=lambda k: (k[2], k[0]))]
            block_events, block_stats = evaluate_block(graph, model, q, ready, fibers, mode=mode, zeta=zeta, prefill=prefill)
            deliver(graph, model, block_events, fibers, messages, outputs)
            events.extend(block_events)
            for name, count in block_stats.items():
                stats[name] = stats.get(name, 0) + count
            stats["region_blocks"] += 1
        for owner, keys in blocks:
            cursor[owner] += len(keys); done.update(keys)
    q.pending = [a for bucket in fibers.values() for a in bucket if a.kind == 1 and a.time >= stop]
    q.cut = stop
    stats["candidate_events"] = len(events)
    return canonicalize(graph, Result(q, events if trace else [], outputs, messages if trace else [], stats))
