"""Exact column-state ↔ graph checkpoint-v5 conversion at complete positions.

Only candidate-computed raw tensors are decoded. No reference or local update
is called. Partial-position cuts and pending work keep the general cursor codec.
"""
import torch
from ..records import Continuation
from ..history import History
from ..settle_ports import validate_alignment


def validate_owner(plan, continuation, samples):
    q, spec = continuation, plan.spec
    if (type(q.batch_size) is not int or q.batch_size < 1 or q.identity != plan.graph.identity or q.batch_size != samples or q.pending
            or type(q.cut) is not int or not 0 <= q.cut < 2**63 or q.cut % spec.stride):
        raise ValueError("column checkpoint requires the matching complete-position owner with no pending work")
    for (b, v) in q.states:
        if type(b) is not int or type(v) is not int or not 0 <= b < samples or not 0 <= v < len(plan.graph.nodes):
            raise ValueError("column checkpoint state owner outside domain")
    for (b, r), h in q.history.items():
        if type(b) is not int or type(r) is not int or not 0 <= b < samples or not 0 <= r < len(plan.graph.regions):
            raise ValueError("column checkpoint history owner outside domain")
        plan.binding.model.regions[r].validate_history(h, plan.graph.region_layouts[r])
        if h.last_time >= q.cut:
            raise ValueError("column checkpoint history is beyond the cut")
    for (b, p), (count, time) in q.ledger.items():
        if (type(b) is not int or type(p) is not int or not 0 <= b < samples or not 0 <= p < len(plan.graph.inputs)
                or type(count) is not int or type(time) is not int or count < 0 or count >= 2**63 or not 0 <= time < q.cut):
            raise ValueError("column checkpoint ledger outside domain")
    validate_alignment(spec, q, [])


def import_state(plan, continuation):
    q = continuation
    validate_owner(plan, q, q.batch_size)
    states = {}
    for v, n in enumerate(plan.binding.model.nodes):
        actual = [q.states.get((b, v), n.initial()) for b in range(q.batch_size)]
        if any(s.last_time >= q.cut for s in actual):
            raise ValueError("column checkpoint state is beyond the cut")
        state = plan.modules[v].state_type.from_states(actual)
        state.validate(n.kernel.config)
        states[v] = state
    return states


def export_state(plan, result, stop_position, previous=None):
    """Typed left state, history and input ledger for existing save/load APIs.

    Input-ledger metadata comes from the actual port rows retained in the result.
    This boundary intentionally materializes only B×nodes terminal states.
    """
    if type(stop_position) is not int or not 0 <= stop_position <= (2**63-1)//plan.spec.stride:
        raise ValueError("invalid column checkpoint stop position")
    if result.graph.identity != plan.graph.identity:
        raise ValueError("column checkpoint graph differs")
    b = result.layout.samples
    q = Continuation(plan.graph.identity, b) if previous is None else previous.fork()
    validate_owner(plan, q, b)
    begin = q.cut//plan.spec.stride
    if stop_position < begin or len(result.layout.times) and ((result.layout.times < begin).any() or (result.layout.times >= stop_position).any()):
        raise ValueError("column checkpoint window lies outside its complete cut")
    owners = result.layout.owners()
    for v, node in result.nodes.items():
        decoded = result.states[v].states()
        for sample in owners[node.rows].unique().tolist():
            q.states[sample, v] = decoded[sample]
            ids = node.rows[owners[node.rows] == sample]
            time = int(result.layout.times[ids[-1]])*plan.spec.stride+plan.spec.rank(v)
            r = plan.graph.nodes[v].region
            q.history[sample, r] = History(last_time=max(time, q.history.get((sample, r), History()).last_time))
    for p, ids in result.input_rows.items():
        result.layout.select(ids)
        for sample in owners[ids].unique().tolist():
            entries = ids[owners[ids] == sample]
            count, before = q.ledger.get((sample, p), (-1, -1))
            first = int(result.layout.times[entries[0]])*plan.spec.stride+plan.spec.rank(plan.graph.inputs[p])
            last = int(result.layout.times[entries[-1]])*plan.spec.stride+plan.spec.rank(plan.graph.inputs[p])
            if first <= before or count+len(entries) >= 2**63:
                raise ValueError("column checkpoint ledger fails monotonicity/capacity")
            q.ledger[sample, p] = (count+len(entries), last)
    q.cut = stop_position*plan.spec.stride
    return q
