"""Exact region blocks: causal state/selection followed by packed Full."""
from collections import defaultdict
import torch
from .region import evaluate as select
from .records import Atom, State
from .packing import prepare_sequences
from .full import FullInput, evaluate as evaluate_full
from .aggregate import evaluate as evaluate_aggregate
from .next import NextInput, evaluate as evaluate_next


def evaluate_block(graph, model, q, frames, fibers, *, mode, zeta, prefill=True,
                   packed=True, full_autograd="replay", aggregate_autograd="replay"):
    """Frames belong to one region across samples, with complete fibers and ordered time.

    Contract: block contains no unresolved Full->fiber edge. Persistent region
    history is processed causally; node state scans are allowed only if state
    adoption/reset is independent of selection.
    """
    events, by_node, by_sequence, by_frame = [], defaultdict(list), defaultdict(list), defaultdict(list)
    by_time = defaultdict(list)
    stats = {"state_blocks": 0, "state_steps": 0, "full_blocks": 0, "state_sequence_calls": 0,
             "semantic_state_replays": 0, "semantic_full_replays": 0, "full_scalar_fallback_steps": 0}
    for batch, region, time, nodes in frames:
        for node in sorted(nodes):
            atoms = sorted(fibers.get((batch, node, time), []), key=lambda a: a.key())
            if not atoms:
                continue
            event = dict(batch=batch, node=node, time=time, fiber=atoms)
            by_node[node].append(event); by_sequence[batch, node].append(event)
            by_time[time].append(event)
            by_frame[batch, time].append(event); events.append(event)
    for node in by_node:
        es = by_node[node]
        evaluate_aggregate(graph, model, es, packed=packed, aggregate_autograd=aggregate_autograd)
        stats["aggregate_calls"] = stats.get("aggregate_calls", 0) + (1 if packed else len(es))
        if torch.is_grad_enabled() and packed:
            key = "batched_aggregate_events" if aggregate_autograd == "batched" else "semantic_aggregate_replays"
            stats[key] = stats.get(key, 0) + len(es)
        if not model.nodes[node].aggregate_program.joint_batch:
            stats["aggregate_scalar_fallback_steps"] = stats.get("aggregate_scalar_fallback_steps", 0) + len(es)
        region = graph.regions[graph.nodes[node].region]
        fallback = None
        if not prefill:
            fallback = "state_prefill_disabled"
        elif not getattr(getattr(model.nodes[node], "kernel", None), "sequence_contract", True):
            fallback = "state_prefill_no_sequence_contract"
        elif not region.observe_all:
            fallback = "state_prefill_selected_adoption"
        elif graph.nodes[node].clear:
            fallback = "state_prefill_selected_clear"
        elif not model.nodes[node].next_program.comparison_identity:
            fallback = "state_prefill_blocked_next"
        if fallback:
            stats[fallback] = stats.get(fallback, 0) + len(es)
            stats["state_prefill_fallback_events"] = stats.get("state_prefill_fallback_events", 0) + len(es)
        if prefill and model.nodes[node].can_prefill and region.observe_all and not graph.nodes[node].clear:
            sequences = [(owner, es) for owner, es in by_sequence.items() if owner[1] == node]
            stats["state_blocks"] += len(sequences)
            if torch.is_grad_enabled():
                stats["semantic_state_replays"] += sum(len(es) for _, es in sequences)
            groups = [sequences] if packed else [[s] for s in sequences]
            for group in groups:
                stats["state_sequence_calls"] += prepare_sequences(model.nodes[node], group, q, region.read_mode)
            stats["max_state_batch"] = max(stats.get("max_state_batch", 0), len(sequences) if packed else 1)
            stats["max_state_sequence"] = max(stats.get("max_state_sequence", 0), max(len(es) for _, es in sequences))
            stats["read_calls"] = stats.get("read_calls", 0) + 1
            if torch.is_grad_enabled():
                stats["semantic_read_replays"] = stats.get("semantic_read_replays", 0) + len(es)
            if not model.nodes[node].read_program.joint_batch:
                stats["read_scalar_batch_steps"] = stats.get("read_scalar_batch_steps", 0) + len(es)
            if not getattr(getattr(model.nodes[node], "kernel", None), "joint_sequence", True):
                stats["state_scalar_sequence_steps"] = stats.get("state_scalar_sequence_steps", 0) + sum(len(es) for _, es in sequences)
    previous_time = None
    for batch, region_id, time, _ in frames:
        if packed and time != previous_time:
            from .block_steps import prepare_tick
            prepare_tick(graph, model, q, by_time[time], stats)
        previous_time = time
        es = by_frame[batch, time]
        if not es:
            continue
        region = graph.regions[region_id]
        for e in es:
            node = e["node"]
            if "proposal" not in e:
                old = q.states.get((batch, node), model.nodes[node].initial())
                prop, desc = model.nodes[node].prepare(old, e["_content"], time, region.read_mode)
                e.update(proposal_state=prop, proposal=prop.value, descriptor=desc)
                stats["state_steps"] += 1
                stats["read_calls"] = stats.get("read_calls", 0) + 1
        nodes = [e["node"] for e in es]
        active, controls, history = select(graph, model, q, batch, region_id, time,
                                           [(e["node"], e["descriptor"]) for e in es])
        stats["region_steps"] = stats.get("region_steps", 0) + 1
        for e in es:
            node = e["node"]
            old = q.states.get((batch, node), model.nodes[node].initial())
            proposal_slots = e["proposal_state"].slots
            cmp = e.pop("proposal_state") if region.observe_all or node in active else old
            e.pop("proposal_state", None)
            next_state = evaluate_next(model.nodes[node], graph.nodes[node], NextInput(
                old, cmp, time, e["_content"], node in active, controls[node]))
            stats["next_steps"] = stats.get("next_steps", 0) + 1
            q.states[batch, node] = next_state
            e.update(active=node in active, control=controls[node], comparison=cmp.value, next=next_state.value,
                     history=history.fork(), proposal_slots=proposal_slots, comparison_slots=cmp.slots, next_slots=next_state.slots,
                     _comparison_state=cmp)
    for node, es in by_node.items():
        active = [e for e in es if e["active"]]
        if not active:
            continue
        requests = [FullInput(e["_comparison_state"], e["time"], e["_content"], e["control"]) for e in active]
        offsets = graph.port_indexes[1].offsets
        values = evaluate_full(model.nodes[node], requests, offsets[node+1] - offsets[node], mode, zeta,
                               packed=packed, full_autograd=full_autograd)
        stats["max_full_batch"] = max(stats.get("max_full_batch", 0), len(active) if packed else 1)
        stats["full_blocks"] += 1
        stats["full_calls"] = stats.get("full_calls", 0) + (1 if packed else len(active))
        if not model.nodes[node].full_program.joint_batch:
            stats["full_scalar_fallback_steps"] += len(active)
        if torch.is_grad_enabled() and packed:
            key = "batched_full_events" if full_autograd == "batched" else "semantic_full_replays"
            stats[key] = stats.get(key, 0) + len(active)
        for event, value in zip(active, values):
            event["full"], event["emitted"] = value.value, value.emitted
    for event in events:
        event.pop("_comparison_state")
        event.pop("_content")
    stats.update(candidate_events=len(events), selected_events=sum(e["active"] for e in events),
                 source_rows=sum(len(e["fiber"]) for e in events))
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
