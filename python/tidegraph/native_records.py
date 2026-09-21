"""Tensor-preserving record conversion, separate from native execution ownership."""
from .records import Atom, Continuation, State


def to_continuation(core, graph, compiled, continuation):
    if continuation.identity != graph.identity:
        raise ValueError("continuation graph identity mismatch")
    q = core.Continuation()
    q.identity, q.batch_size, q.cut = compiled.identity, continuation.batch_size, continuation.cut
    q.states = {k: core.State(s.value, s.last_time, s.observations, s.slots) for k, s in continuation.states.items()}
    q.history, q.ledger = continuation.history, continuation.ledger
    q.pending = [core.Atom(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value) for a in continuation.pending]
    return q


def atom(a):
    return Atom(a.batch, a.node, a.time, a.kind, a.source, a.position, a.value)


def from_continuation(graph, q):
    return Continuation(graph.identity, q.batch_size, q.cut,
                        {k: State(s.value, s.last_time, s.observations, s.slots) for k, s in q.states.items()},
                        q.history, [atom(a) for a in q.pending], q.ledger)


def window_records(result):
    events = []
    for e in result.trace:
        event = {k: getattr(e, k) for k in ("batch", "node", "time", "content", "proposal", "descriptor",
                                          "control", "comparison", "next", "active", "history",
                                          "proposal_slots", "comparison_slots", "next_slots", "contributions")}
        event["fiber"] = [atom(a) for a in e.fiber]
        if e.active:
            event["full"] = e.full
            event["emitted"] = e.emitted
        events.append(event)
    return events, [(o.batch, o.time, o.port, o.value) for o in result.outputs], [atom(a) for a in result.messages], result.stats
