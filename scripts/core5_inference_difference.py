"""Chronological strict witnesses and literal FP64 source/KV write anchors.

Anchors consume already returned records after both schedules have completed.
They neither supply a candidate trajectory nor change its arithmetic policy.
"""
import torch
from tidegraph.compare import equivalent
from tidegraph.planning.precision import tolerances


def check(actual, expected, *, exact=False):
    try:
        equivalent(actual, expected, **(dict(atol=0, rtol=0) if exact else {}))
    except AssertionError as error:
        return str(error)
    return None


def event_key(event):
    return event["time"], event["batch"], event["node"]


def aligned_events(actual, expected):
    left = {event_key(e): e for e in actual.trace}
    right = {event_key(e): e for e in expected.trace}
    if len(left) != len(actual.trace) or len(right) != len(expected.trace):
        raise ValueError("duplicate physical event identity")
    if left.keys() != right.keys():
        raise ValueError("candidate/reference event identities differ")
    return [(left[key], right[key]) for key in sorted(left)]


def first_strict_trace(actual, expected):
    try:
        events = aligned_events(actual, expected)
    except ValueError as error:
        return dict(reason=str(error))
    for a, b in events:
        error = check(a, b)
        if error is not None:
            return dict(event=list(event_key(a)), error=error,
                        fields={name: check(a.get(name), b.get(name)) for name in sorted(a.keys() | b.keys())})
    return None


def delta(actual, anchor, dtype):
    """Unchanged public dtype threshold; FP64 anchor need not share storage dtype."""
    if actual.shape != anchor.shape:
        return dict(reason="shape mismatch", actual=list(actual.shape), anchor=list(anchor.shape))
    if not bool(torch.isfinite(actual).all() and torch.isfinite(anchor).all()):
        return dict(reason="nonfinite operand")
    atol, rtol = tolerances(dtype)
    error = (actual.detach().double()-anchor.detach().double()).abs()
    ratio = error/(atol+rtol*anchor.detach().double().abs())
    count = int((ratio > 1).sum())
    if not error.numel():
        return dict(maxabs=0., failing_coordinates=0, exact=True, atol=atol, rtol=rtol)
    index = int(ratio.flatten().argmax())
    return dict(maxabs=float(error.max()), failing_coordinates=count, exact=bool((error == 0).all()),
                max_scaled_error=float(ratio.max()), flat_index=index,
                actual=float(actual.flatten()[index]), anchor=float(anchor.flatten()[index]),
                atol=atol, rtol=rtol)


def literal_kv(memory, content):
    """KV §1.2 factored map in CPU FP64, with original parameter values."""
    x = content.detach().cpu().double()
    normal = x/(memory.config.epsilon+x.square().mean()).sqrt()
    base = memory.base.detach().cpu().double()@normal
    low = memory.adapter_v.detach().cpu().double()@normal
    private = memory.adapter_u.detach().cpu().double()@low
    return dict(normal=normal, base=base, adapter_low=low, adapter=private, write=base+private)


def literal_source(program, event):
    """Architecture §2: each physical atom, class map, weighted complete fiber."""
    from tidegraph.weighted.programs import WeightedAggregate
    if not isinstance(program, WeightedAggregate):
        raise ValueError("source anchor requires the public weighted Aggregate")
    masses, values = [], []
    for atom in event["fiber"]:
        payload = atom.value.detach().cpu().double()
        mass, z = payload[0].clamp_min(0), payload[1:]
        if program.alpha:
            transform = program.transforms[str(program.classes.classify(atom))]
            a, b = transform.a.detach().cpu().double(), transform.b.detach().cpu().double()
            if hasattr(transform, "a_left"):
                a = a+transform.a_left.detach().cpu().double()@transform.a_right.detach().cpu().double()
                b = b+transform.b_left.detach().cpu().double()@transform.b_right.detach().cpu().double()
            normal = z/(program.epsilon+z.square().mean()).sqrt()
            hidden = a@normal+transform.embedding.detach().cpu().double()
            z = z+program.alpha*(b@(hidden*torch.sigmoid(hidden)))
        masses.append(mass); values.append(z)
    masses, values = torch.stack(masses), torch.stack(values)
    total = masses.sum()
    content = (masses[:, None]*values).sum(0)/total if total > 0 else torch.zeros_like(values[0])
    return torch.cat((total.reshape(1), content))


def new_write(event):
    slots = event["proposal_slots"]
    ids = torch.nonzero(slots["times"] == event["time"]).flatten()
    if ids.numel() != 1:
        raise ValueError("selected KV event has no unique positive write at its original time")
    i = int(ids[0])
    return torch.cat((slots["keys"][i], slots["values"][i]))


def kv_witness(binding, actual, expected):
    """Separate upstream content differences from each local write's FP64 error."""
    from tidegraph.weighted.kv import KVMemory
    weights = binding.model.nodes[actual["node"]]
    memory = weights.kernel
    if not isinstance(memory, KVMemory):
        return dict(state="inapplicable", reason="selected event is not KV")
    try:
        a, b = new_write(actual), new_write(expected)
    except ValueError as error:
        return dict(state="inapplicable", reason=str(error))
    dtype = a.dtype
    local_a = literal_kv(memory, actual["content"][1:])
    local_b = literal_kv(memory, expected["content"][1:])
    source_a = literal_source(weights.aggregate_program, actual)
    source_b = literal_source(weights.aggregate_program, expected)
    coherent_a = literal_kv(memory, source_a[1:])["write"]
    coherent_b = literal_kv(memory, source_b[1:])["write"]
    return dict(state="observed", event=list(event_key(actual)),
        scope="post-execution CPU FP64 local arithmetic anchor; not exact real arithmetic or a full-graph repair",
        fiber_exact_error=check(actual["fiber"], expected["fiber"], exact=True),
        content=delta(actual["content"], expected["content"], dtype),
        stored_write=delta(a, b, dtype),
        candidate_write_vs_same_content_fp64=delta(a, local_a["write"], dtype),
        reference_write_vs_same_content_fp64=delta(b, local_b["write"], dtype),
        propagated_content_fp64=delta(local_a["write"], local_b["write"], dtype),
        candidate_source_vs_same_fiber_fp64=delta(actual["content"], source_a, dtype),
        reference_source_vs_same_fiber_fp64=delta(expected["content"], source_b, dtype),
        candidate_write_vs_coherent_source_kv_fp64=delta(a, coherent_a, dtype),
        reference_write_vs_coherent_source_kv_fp64=delta(b, coherent_b, dtype),
        coherent_fiber_difference=delta(coherent_a, coherent_b, dtype))


def first_kv_witness(binding, actual, expected):
    try:
        events = aligned_events(actual, expected)
    except ValueError as error:
        return dict(state="inapplicable", reason=str(error))
    first_exact = None
    for a, b in events:
        if "keys" not in a["proposal_slots"] or "keys" not in b["proposal_slots"]:
            continue
        try:
            x, y = new_write(a), new_write(b)
        except ValueError:
            continue
        if first_exact is None and not torch.equal(x, y):
            first_exact = (a, b)
        if check(x, y) is not None:
            return dict(selection="earliest strict new KV key/value write", **kv_witness(binding, a, b))
    if first_exact is not None:
        return dict(selection="earliest exact-only new KV write; no strict new-write difference", **kv_witness(binding, *first_exact))
    return dict(state="identical", selection="all observed new KV writes")
