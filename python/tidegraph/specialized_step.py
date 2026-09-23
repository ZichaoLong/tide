"""Scalar local frame formulas for the independent fixed-topology schedules."""
from .aggregate import evaluate as aggregate
from .full import FullInput, evaluate as full
from .next import NextInput, evaluate as next_state
from .region import evaluate as select


def frame(graph, model, q, nodes, batch, time, inbox, mode, zeta):
    events = []
    for node in nodes:
        fiber = sorted(inbox.get((node, batch, time), ()), key=lambda a: a.key())
        if not fiber:
            continue
        event = dict(batch=batch, node=node, time=time, fiber=fiber)
        aggregate(graph, model, [event])
        old = q.states.get((batch, node), model.nodes[node].initial())
        region = graph.regions[graph.nodes[node].region]
        proposal, descriptor = model.nodes[node].prepare(old, event["_content"], time, region.read_mode)
        event.update(_old=old, _proposal=proposal, descriptor=descriptor)
        events.append(event)
    if not events:
        return []
    region_id = graph.nodes[nodes[0]].region
    active, controls, history = select(graph, model, q, batch, region_id, time,
                                       [(e["node"], e["descriptor"]) for e in events])
    region = graph.regions[region_id]
    for event in events:
        node = event["node"]
        old, proposal, content = event.pop("_old"), event.pop("_proposal"), event.pop("_content")
        selected, control = node in active, controls[node]
        comparison = proposal if region.observe_all or selected else old
        persistent = next_state(model.nodes[node], graph.nodes[node], NextInput(
            old, comparison, time, content, selected, control))
        q.states[batch, node] = persistent
        if selected:
            offsets = graph.port_indexes[1].offsets
            value = full(model.nodes[node], [FullInput(comparison, time, content, control)],
                         offsets[node+1] - offsets[node], mode, zeta)[0]
            event.update(full=value.value, emitted=value.emitted)
        event.update(proposal=proposal.value, active=selected, control=control,
                     comparison=comparison.value, next=persistent.value, history=history.fork(),
                     proposal_slots=proposal.slots, comparison_slots=comparison.slots, next_slots=persistent.slots)
    return events
