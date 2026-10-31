"""Observed eager tensor ownership, using metadata only (no device readback)."""
from collections import Counter, defaultdict


def _counts(values):
    count, size = Counter(), Counter()
    for tensor in values:
        device = str(tensor.device)
        count[device] += 1
        size[device] += tensor.numel()*tensor.element_size()
    return {device: dict(tensors=count[device], logical_bytes=size[device]) for device in sorted(count)}


def _owners(pairs):
    result = defaultdict(set)
    for owner, value in pairs:
        result[str(owner)].add(str(value.device))
    return {owner: sorted(devices) for owner, devices in sorted(result.items())}


def eager_locations(result):
    q = result.continuation
    return dict(scope="actual eager tensors before public return packing; logical bytes count aliases, not allocator peaks",
        state_values=_counts(s.value for s in q.states.values()),
        state_slots=_counts(x for s in q.states.values() for x in s.slots.values()),
        state_node_devices=_owners((node, s.value) for (_, node), s in q.states.items()),
        history_tensors=_counts(x for h in q.history.values() for x in h.tensors.values()),
        messages=_counts(a.value for a in result.messages),
        message_target_node_devices=_owners((a.node, a.value) for a in result.messages),
        pending=_counts(a.value for a in q.pending),
        pending_target_node_devices=_owners((a.node, a.value) for a in q.pending),
        outputs=_counts(x for *_, x in result.outputs),
        output_port_devices=_owners((port, x) for _, _, port, x in result.outputs))
