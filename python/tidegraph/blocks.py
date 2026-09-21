"""Exact region blocks: causal state/selection followed by packed Full."""
from collections import defaultdict
import torch
from .ops import select
from .records import Atom, State
from .packing import prepare_sequences
from . import autograd


def evaluate_block(graph, model, q, frames, fibers, *, mode, zeta, prefill=True):
    """Frames belong to one region across samples, with complete fibers and ordered time.

    Contract: block contains no unresolved Full->fiber edge. Persistent region
    history is processed causally; node state scans are allowed only if state
    adoption/reset is independent of selection.
    """
    events, by_node, by_sequence, by_frame = [], defaultdict(list), defaultdict(list), defaultdict(list)
    stats = {"state_blocks": 0, "state_steps": 0, "full_blocks": 0, "state_sequence_calls": 0,
             "semantic_state_replays": 0, "semantic_full_replays": 0}
    for batch, region, time, nodes in frames:
        for node in sorted(nodes):
            atoms = sorted(fibers.get((batch, node, time), []), key=lambda a: a.key())
            if not atoms:
                continue
            event = dict(batch=batch, node=node, time=time, fiber=atoms, content=model.aggregate(atoms))
            by_node[node].append(event); by_sequence[batch, node].append(event)
            by_frame[batch, time].append(event); events.append(event)
    for node in by_node:
        region = graph.regions[graph.nodes[node].region]
        if prefill and model.nodes[node].can_prefill and region.observe_all and not graph.nodes[node].clear:
            sequences = [(owner, es) for owner, es in by_sequence.items() if owner[1] == node]
            stats["state_blocks"] += len(sequences)
            if torch.is_grad_enabled():
                stats["semantic_state_replays"] += sum(len(es) for _, es in sequences)
            stats["state_sequence_calls"] += prepare_sequences(model.nodes[node], sequences, q)
    for batch, region_id, time, _ in frames:
        es = by_frame[batch, time]
        if not es:
            continue
        region = graph.regions[region_id]
        for e in es:
            node = e["node"]
            if "proposal" not in e:
                old = q.states.get((batch, node), model.nodes[node].initial())
                prop, desc = model.nodes[node].prepare(old, e["content"], time)
                e.update(proposal_state=prop, proposal=prop.value, descriptor=desc)
                stats["state_steps"] += 1
        nodes = [e["node"] for e in es]
        active, controls, history = select(nodes, {e["node"]: e["descriptor"] for e in es},
                                           q.history.get((batch, region_id), {}), region)
        q.history[batch, region_id] = history
        for e in es:
            node = e["node"]
            old = q.states.get((batch, node), model.nodes[node].initial())
            proposal_slots = e["proposal_state"].slots
            cmp = e.pop("proposal_state") if region.observe_all or node in active else old
            e.pop("proposal_state", None)
            next_state = model.nodes[node].next(cmp, graph.nodes[node].clear and node in active)
            q.states[batch, node] = next_state
            e.update(active=node in active, control=controls[node], comparison=cmp.value, next=next_state.value,
                     history=dict(history), proposal_slots=proposal_slots, comparison_slots=cmp.slots, next_slots=next_state.slots)
    for node, es in by_node.items():
        active = [e for e in es if e["active"]]
        if not active:
            continue
        with torch.no_grad():
            values = model.nodes[node].full(torch.stack([e["comparison"] for e in active]),
                                           torch.stack([e["content"] for e in active]),
                                           torch.stack([e["control"] for e in active]), mode, zeta)
        stats["full_blocks"] += 1
        for event, value in zip(active, values):
            if torch.is_grad_enabled():
                reference = model.nodes[node].full(event["comparison"], event["content"], event["control"], mode, zeta)
                value = autograd.value(value, reference); stats["semantic_full_replays"] += 1
            event["full"] = value
    return events, stats


def deliver(graph, model, events, fibers, messages, outputs):
    offsets, edges = graph.adjacency()
    for e in events:
        if not e["active"]:
            continue
        node, batch, time = e["node"], e["batch"], e["time"]
        for edge_id in edges[offsets[node]:offsets[node + 1]]:
            edge = graph.edges[edge_id]
            arrival = time + edge.delay
            if arrival >= 2**63:
                raise ValueError("logical time overflow")
            a = Atom(batch, edge.target, arrival, 1, edge_id, time, e["full"] * model.edge_scale[edge_id])
            fibers[batch, edge.target, arrival].append(a); messages.append(a)
        for port, source in enumerate(graph.outputs):
            if source == node:
                outputs.append((batch, time, port, e["full"] * model.output_scale[port]))


def canonicalize(graph, result):
    result.trace.sort(key=lambda e: (e["time"], e["batch"], e["node"]))
    result.outputs.sort(key=lambda o: (o[1], o[0], graph.outputs[o[2]], o[2]))
    result.messages.sort(key=lambda a: (a.position, a.batch, graph.edges[a.source].source, a.source))
    result.continuation.pending.sort(key=lambda a: a.key())
    return result
