"""Exact region blocks: causal state/selection followed by packed Full."""
from collections import defaultdict
import torch
from .ops import select
from .records import Atom, State
from .packing import prepare_sequences
from .full import FullInput, evaluate as evaluate_full
from .aggregate import evaluate as evaluate_aggregate


def evaluate_block(graph, model, q, frames, fibers, *, mode, zeta, prefill=True):
    """Frames belong to one region across samples, with complete fibers and ordered time.

    Contract: block contains no unresolved Full->fiber edge. Persistent region
    history is processed causally; node state scans are allowed only if state
    adoption/reset is independent of selection.
    """
    events, by_node, by_sequence, by_frame = [], defaultdict(list), defaultdict(list), defaultdict(list)
    stats = {"state_blocks": 0, "state_steps": 0, "full_blocks": 0, "state_sequence_calls": 0,
             "semantic_state_replays": 0, "semantic_full_replays": 0, "full_scalar_fallback_steps": 0}
    for batch, region, time, nodes in frames:
        for node in sorted(nodes):
            atoms = sorted(fibers.get((batch, node, time), []), key=lambda a: a.key())
            if not atoms:
                continue
            event = dict(batch=batch, node=node, time=time, fiber=atoms)
            by_node[node].append(event); by_sequence[batch, node].append(event)
            by_frame[batch, time].append(event); events.append(event)
    for node in by_node:
        es = by_node[node]
        evaluate_aggregate(graph, model, es, packed=True)
        stats["aggregate_calls"] = stats.get("aggregate_calls", 0) + 1
        if torch.is_grad_enabled():
            stats["semantic_aggregate_replays"] = stats.get("semantic_aggregate_replays", 0) + len(es)
        if not model.nodes[node].aggregate_program.joint_batch:
            stats["aggregate_scalar_fallback_steps"] = stats.get("aggregate_scalar_fallback_steps", 0) + len(es)
        region = graph.regions[graph.nodes[node].region]
        if prefill and model.nodes[node].can_prefill and region.observe_all and not graph.nodes[node].clear:
            sequences = [(owner, es) for owner, es in by_sequence.items() if owner[1] == node]
            stats["state_blocks"] += len(sequences)
            if torch.is_grad_enabled():
                stats["semantic_state_replays"] += sum(len(es) for _, es in sequences)
            stats["state_sequence_calls"] += prepare_sequences(model.nodes[node], sequences, q)
            if not getattr(getattr(model.nodes[node], "kernel", None), "joint_sequence", True):
                stats["state_scalar_sequence_steps"] = stats.get("state_scalar_sequence_steps", 0) + sum(len(es) for _, es in sequences)
    for batch, region_id, time, _ in frames:
        es = by_frame[batch, time]
        if not es:
            continue
        region = graph.regions[region_id]
        for e in es:
            node = e["node"]
            if "proposal" not in e:
                old = q.states.get((batch, node), model.nodes[node].initial())
                prop, desc = model.nodes[node].prepare(old, e["_content"], time)
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
                     history=dict(history), proposal_slots=proposal_slots, comparison_slots=cmp.slots, next_slots=next_state.slots,
                     _comparison_state=cmp)
    for node, es in by_node.items():
        active = [e for e in es if e["active"]]
        if not active:
            continue
        requests = [FullInput(e["_comparison_state"], e["time"], e["_content"], e["control"]) for e in active]
        offsets = graph.port_indexes[1].offsets
        values = evaluate_full(model.nodes[node], requests, offsets[node+1] - offsets[node], mode, zeta, packed=True)
        stats["full_blocks"] += 1
        if not model.nodes[node].full_program.joint_batch:
            stats["full_scalar_fallback_steps"] += len(active)
        if torch.is_grad_enabled():
            stats["semantic_full_replays"] += len(active)
        for event, value in zip(active, values):
            event["full"], event["emitted"] = value.value, value.emitted
    for event in events:
        event.pop("_comparison_state")
        event.pop("_content")
    return events, stats


def deliver(graph, model, events, fibers, messages, outputs):
    index = graph.port_indexes[1]
    for e in events:
        if not e["active"]:
            continue
        node, batch, time = e["node"], e["batch"], e["time"]
        for slot, value in e["emitted"].items():
            kind, source = index.bindings[index.offsets[node] + slot]
            if kind == 1:
                edge = graph.edges[source]
                arrival = time + edge.delay
                if arrival >= 2**63:
                    raise ValueError("logical time overflow")
                a = Atom(batch, edge.target, arrival, 1, source, time, value * model.edge_scale[source])
                fibers[batch, edge.target, arrival].append(a); messages.append(a)
            else:
                outputs.append((batch, time, source, value * model.output_scale[source]))


def canonicalize(graph, result):
    result.trace.sort(key=lambda e: (e["time"], e["batch"], e["node"]))
    result.outputs.sort(key=lambda o: (o[1], o[0], graph.outputs[o[2]], o[2]))
    result.messages.sort(key=lambda a: (a.position, a.batch, graph.edges[a.source].source, a.source))
    result.continuation.pending.sort(key=lambda a: a.key())
    return result
