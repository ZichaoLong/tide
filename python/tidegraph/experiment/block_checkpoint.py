"""Value-only decoding of the public block archive, including opaque padding.

Static canonical order/shapes identify the packed slots. Every numerical field
is copied from the actual owner; no recurrence or optimizer is replayed here.
"""
from ..resident_training_options import GROUP_FIELDS, STATE_FIELDS
from .returns import snapshot


def record(client):
    c = client.checkpoint()
    parameters = dict(c.parameters)
    owners = sorted(parameters)
    offsets, count = [], 0
    for name in owners:
        offsets.append(count)
        count += parameters[name].numel()
    if c.slots.values.numel() != max(1, count) or c.slots.steps.numel() != max(1, len(owners)):
        raise ValueError("block checkpoint canonical owner/slot extent differs")
    trainable = {n for n, x in client.model.named_parameters(remove_duplicate=False) if x.requires_grad}
    raw = lambda rows: [dict(floating=x.floating, integer=x.integer) for x in rows]
    return snapshot(dict(schema="tide-experiment-block-checkpoint-v2", native_schema=c.schema,
        generation=c.generation, next_token=c.next_token, identity=c.identity,
        aliases=[list(a) for a in c.aliases], parameters=parameters,
        trainable=[n for aliases in c.aliases for n in aliases if n.replace(".extra.", ".") in trainable],
        optimizer="sgd" if c.optimizer == client.module.OptimizerKind.sgd else "adamw",
        groups=[{name: getattr(g, name) for name in GROUP_FIELDS} for g in c.groups],
        owner_names=owners, offsets=offsets, state={name: getattr(c.slots, name) for name in STATE_FIELDS},
        raw=raw(c.state), bound_initial=raw(c.bound_initial), last_positions=c.last_positions,
        node_present=c.node_present, region_times=c.region_times,
        input_counts=c.input_counts, input_times=c.input_times))
