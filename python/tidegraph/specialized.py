"""Independent fixed self-loop and chain schedules; no generic scheduler calls."""
from collections import defaultdict
import torch
from .blocks import canonicalize
from .records import Atom, Result, State
from .validation import validate_window
from .full import FullInput, evaluate as evaluate_full
from .aggregate import evaluate as evaluate_aggregate


def validate_topology(graph, topology):
    n = len(graph.nodes)
    if graph.inputs != (0,) or graph.outputs != (n - 1,):
        raise ValueError("specialization requires one chain input/output")
    if len(graph.regions) != n or any(node.region != i for i, node in enumerate(graph.nodes)):
        raise ValueError("specialization requires singleton ordered regions")
    expected = [(0, 0)] if topology == "self_loop" else [(i, i + 1) for i in range(n - 1)]
    if topology not in {"self_loop", "chain"} or (topology == "self_loop" and n != 1):
        raise ValueError("invalid specialization")
    if [(e.source, e.target) for e in graph.edges] != expected:
        raise ValueError("specialization topology mismatch")


def _step(graph, model, q, node, batch, time, atoms, mode, zeta):
    fiber = sorted(atoms, key=lambda a: a.key())
    event = dict(batch=batch, node=node, time=time, fiber=fiber)
    evaluate_aggregate(graph, model, [event])
    h = event["content"]
    old = q.states.get((batch, node), model.nodes[node].initial())
    prop, desc = model.nodes[node].prepare(old, h, time)
    # A singleton softmax retains the generic zero VJP connection to its score.
    control = desc.reshape(1).softmax(0)[0]
    next_state = model.nodes[node].next(prop, graph.nodes[node].clear)
    q.states[batch, node] = next_state
    history = dict(q.history.get((batch, node), {}))
    history[node] = history.get(node, 0) + 1; q.history[batch, node] = history
    offsets = graph.port_indexes[1].offsets
    value = evaluate_full(model.nodes[node], [FullInput(prop, time, h, control)], offsets[node+1] - offsets[node], mode, zeta)[0]
    return dict(event, proposal=prop.value,
                descriptor=desc, control=control, active=True, comparison=prop.value,
                next=next_state.value, history=dict(history), full=value.value, emitted=value.emitted, proposal_slots=prop.slots,
                comparison_slots=prop.slots, next_slots=next_state.slots)


def run(graph, model, initial, external, stop, *, sealed_until, topology, mode="hard", zeta=1.0):
    validate_topology(graph, topology)
    atoms, ledger = validate_window(graph, model, initial, external, stop, sealed_until)
    q = initial.fork(); q.ledger = ledger
    inbox = defaultdict(list)
    for a in list(q.pending) + atoms:
        inbox[a.node, a.batch, a.time].append(a)
    events, messages, outputs = [], [], []
    def apply(node, batch, time):
        fiber = inbox[node, batch, time]
        if not fiber:
            return
        e = _step(graph, model, q, node, batch, time, fiber, mode, zeta); events.append(e)
        if topology == "self_loop" or node < len(graph.nodes) - 1:
            edge = 0 if topology == "self_loop" else node
            target = 0 if topology == "self_loop" else node + 1
            slot = graph.ports.edge_source[edge]
            if slot in e["emitted"]:
                arrival = time + graph.edges[edge].delay
                if arrival >= 2**63:
                    raise ValueError("logical time overflow")
                a = Atom(batch, target, arrival, 1, edge, time, e["emitted"][slot] * model.edge_scale[edge])
                inbox[target, batch, arrival].append(a); messages.append(a)
        if node == len(graph.nodes) - 1 and graph.ports.output[0] in e["emitted"]:
            outputs.append((batch, time, 0, e["emitted"][graph.ports.output[0]] * model.output_scale[0]))
    if topology == "self_loop":
        for time in range(q.cut, stop):
            for batch in range(q.batch_size):
                apply(0, batch, time)
    else:
        for node in range(len(graph.nodes)):
            coordinates = sorted((t, b) for (v, b, t), xs in inbox.items() if v == node and xs and q.cut <= t < stop)
            for time, batch in coordinates:
                apply(node, batch, time)
    q.pending = [a for bucket in inbox.values() for a in bucket if a.kind == 1 and a.time >= stop]
    q.cut = stop
    return canonicalize(graph, Result(q, events, outputs, messages, {"candidate_events": len(events)}))


def settle_chain(spec, model, initial, values, *, mode="hard", zeta=1.0):
    """Fixed-chain SettleGraph anchor; independent token propagation schedule."""
    validate_topology(spec.graph, "chain")
    if initial.cut % spec.stride or initial.pending:
        raise ValueError("SettleGraph needs a complete position cut")
    result = run(spec.graph, model, initial, spec.external(values, initial.cut // spec.stride),
                 initial.cut + values.shape[1] * spec.stride,
                 sealed_until=initial.cut + values.shape[1] * spec.stride,
                 topology="chain", mode=mode, zeta=zeta)
    result.outputs = [(b, t - spec.rank(spec.graph.outputs[0]) + spec.output_rank, p, v)
                      for b, t, p, v in result.outputs]
    return result
