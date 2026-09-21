"""Direct time-major specification oracle; deliberately not event-queue based."""
from .ops import select
from .records import Atom, Result, State
from .validation import validate_window
from .full import FullInput, evaluate as evaluate_full


def run(graph, model, continuation, external, stop, *, sealed_until,
        mode="hard", zeta=1.0, trace=True):
    if mode not in {"hard", "hst", "softp"}:
        raise ValueError("invalid emit mode")
    atoms, ledger = validate_window(graph, model, continuation, external, stop, sealed_until)
    q = continuation.fork()
    q.ledger = ledger
    available = list(q.pending) + atoms
    events, outputs, messages = [], [], []
    visits = 0
    for time in range(q.cut, stop):
        prepared = {}
        for batch in range(q.batch_size):
            for node in range(len(graph.nodes)):
                fiber = sorted((a for a in available if a.batch == batch and a.node == node
                                and a.time == time), key=lambda a: a.key())
                if not fiber:
                    continue
                visits += 1
                old = q.states.get((batch, node), model.nodes[node].initial())
                h = model.aggregate(fiber)
                prop, desc = model.nodes[node].prepare(old, h, time)
                prepared[batch, node] = dict(batch=batch, node=node, time=time, fiber=fiber,
                                            content=h, proposal=prop.value, descriptor=desc,
                                            old=old, proposal_state=prop)
        for batch in range(q.batch_size):
            for r, region in enumerate(graph.regions):
                nodes = sorted(v for b, v in prepared if b == batch and graph.nodes[v].region == r)
                if not nodes:
                    continue
                active, controls, history = select(nodes, {v: prepared[batch, v]["descriptor"] for v in nodes},
                                                   q.history.get((batch, r), {}), region)
                q.history[batch, r] = history
                for v in nodes:
                    event = prepared[batch, v]
                    comparison = event["proposal_state"] if region.observe_all or v in active else event["old"]
                    next_state = model.nodes[v].next(comparison, graph.nodes[v].clear and v in active)
                    q.states[batch, v] = next_state
                    event.update(active=v in active, control=controls[v], comparison=comparison.value,
                                 _comparison_state=comparison,
                                 next=next_state.value, history=dict(history), proposal_slots=event["proposal_state"].slots,
                                 comparison_slots=comparison.slots, next_slots=next_state.slots)
                    event.pop("old")
                    event.pop("proposal_state")
        for (batch, node), event in sorted(prepared.items()):
            if event["active"]:
                offsets = graph.port_indexes[1].offsets
                request = FullInput(event["_comparison_state"], time, event["content"], event["control"])
                result = evaluate_full(model.nodes[node], [request], offsets[node+1] - offsets[node], mode, zeta)[0]
                event["full"], event["emitted"] = result.value, result.emitted
                for e, edge in enumerate(graph.edges):
                    slot = graph.ports.edge_source[e]
                    if edge.source != node or slot not in result.emitted:
                        continue
                    arrival = time + edge.delay
                    if arrival >= 2**63:
                        raise ValueError("logical time overflow")
                    message = Atom(batch, edge.target, arrival, 1, e, time, result.emitted[slot] * model.edge_scale[e])
                    available.append(message)
                    messages.append(message)
                for port, source in enumerate(graph.outputs):
                    slot = graph.ports.output[port]
                    if source == node and slot in result.emitted:
                        outputs.append((batch, time, port, result.emitted[slot] * model.output_scale[port]))
            event.pop("_comparison_state")
            if trace:
                events.append(event)
    q.pending = sorted((a for a in available if a.kind == 1 and a.time >= stop), key=lambda a: a.key())
    q.cut = stop
    return Result(q, events, outputs, messages if trace else [], {"candidate_events": visits})
