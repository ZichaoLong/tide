"""Architecture §2: regional source preparation and physical-atom means.

Numerical source reuse never deduplicates physical messages. Whole aggregates
share only when ordered payload roots, row domains and parameter slots coincide.
Caches are regional-call local; no value, parameter or version crosses windows.
"""
import torch
from .column_sources import parameters, source_columns
from .column_sharing import shared_uses
from .mean import weighted_mean
from .packing_observation import bucket


def signature(program, sources, payload_keys=None):
    payload_keys = {} if payload_keys is None else payload_keys
    for p, _ in sources:
        if id(p) not in payload_keys:
            payload_keys[id(p)] = (tuple(p.rows.tolist()), id(p.strength), id(p.content))
    return program.alpha, program.epsilon, tuple((payload_keys[id(p)],
        tuple(map(id, parameters(program.transforms[str(c)]))) if program.alpha else ()) for p, c in sources)


def assemble(sources, values):
    first = sources[0][0].rows
    aligned = all(torch.equal(p.rows, first) for p, _ in sources[1:])
    rows = first if aligned else torch.cat([p.rows for p, _ in sources]).unique(sorted=True)
    masses, aligned_values = [], []
    for (payload, _), z in zip(sources, values):
        if aligned or torch.equal(payload.rows, rows):
            masses.append(payload.strength.clamp_min(0)); aligned_values.append(z)
        else:
            ids = torch.searchsorted(rows, payload.rows).to(z.device)
            masses.append(z.new_zeros(len(rows)).index_copy(0, ids, payload.strength.clamp_min(0)))
            aligned_values.append(z.new_zeros((len(rows), z.shape[-1])).index_copy(0, ids, z))
    mass, value = torch.stack(masses, 1), torch.stack(aligned_values, 1)
    total = mass.sum(1)
    # Existence and clamp/positive branches are unchanged. An entirely zero
    # fiber retains mass roots and has no content/transform dependency.
    positive = (total.detach() > 0).nonzero().flatten()
    if len(positive) == len(rows):
        u = weighted_mean(mass, value)
    elif len(positive):
        u = weighted_mean(mass.index_select(0, positive), value.index_select(0, positive))
        u = value.new_zeros((len(rows), value.shape[-1])).index_copy(0, positive, u)
    else:
        u = value.new_zeros((len(rows), value.shape[-1]))
    return rows, total, u


def aggregate_columns_many(requests, *, max_workspace_bytes=4*1024**3):
    """Ordered (program, sources) requests, independently rooted receiver uses."""
    keys, unique, transform_requests, payload_keys = [], {}, [], {}
    for program, sources in requests:
        sources = [(p, c) for p, c in sources if len(p.rows)]
        key = signature(program, sources, payload_keys) if sources else None
        keys.append(key)
        if key is None or key in unique:
            continue
        indices = []
        for p, label in sources:
            weights = parameters(program.transforms[str(label)]) if program.alpha else ()
            indices.append(len(transform_requests))
            transform_requests.append((p.content, weights, program.alpha, program.epsilon))
        unique[key] = sources, indices
    transformed = source_columns(transform_requests, max_workspace_bytes=max_workspace_bytes)
    from .aggregate_tensor import assemble_many
    entries = [(sources, [transformed[i] for i in ids]) for sources, ids in unique.values()]
    computed = dict(zip(unique, assemble_many(entries, max_workspace_bytes, assemble)))
    received = {i: computed.get(key) for i, key in enumerate(keys)}
    if len(computed) < sum(key is not None for key in keys):
        bucket("aggregate_exact_reuse", len(computed), sum(key is not None for key in keys))
    return list(shared_uses(received).values())
