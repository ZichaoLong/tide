"""Pure State sharing and independent batching in one legal execution block.

Preparation belongs to the execution graph, not mathematical Content or State.
Programs opt into each capability separately. Only equal sharing keys reuse
differentiable operands; independent batches retain their own input/parameter
identities. Private state is never cached here.
"""


def prepare_shared(model, events, stats):
    groups = {}
    for event in events:
        event.pop("_state_prepared", None)
        kernel = getattr(model.nodes[event["node"]], "kernel", None)
        key_for = getattr(kernel, "preparation_key", None)
        if key_for is None:
            continue
        key = key_for(event["content"])
        if key is not None:
            groups.setdefault((type(kernel), key), []).append(event)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        first = rows[0]
        kernel = model.nodes[first["node"]].kernel
        values = kernel.prepare_shared(first["content"])
        private = getattr(kernel, "prepare_private_batch", None)
        actual = ([values]*len(rows) if private is None else
                  private([model.nodes[e["node"]].kernel for e in rows], values, stats))
        if len(actual) != len(rows):
            raise ValueError("private preparation row count mismatch")
        for event, value in zip(rows, actual):
            event["_state_prepared"] = value
        stats["state_preparation_calls"] = stats.get("state_preparation_calls", 0)+1
        stats["state_preparation_events"] = stats.get("state_preparation_events", 0)+len(rows)
        stats["state_preparation_reused_events"] = stats.get("state_preparation_reused_events", 0)+len(rows)-1
    independent = {}
    for event in events:
        kernel = getattr(model.nodes[event["node"]], "kernel", None)
        if "_state_prepared" not in event and hasattr(kernel, "prepare_independent_batch"):
            independent.setdefault(type(kernel), []).append(event)
    for rows in independent.values():
        if len(rows) < 2:
            continue
        kernels = [model.nodes[e["node"]].kernel for e in rows]
        actual = kernels[0].prepare_independent_batch(kernels, [e["content"] for e in rows], stats)
        if len(actual) != len(rows):
            raise ValueError("independent preparation row count mismatch")
        for event, value in zip(rows, actual):
            if value:
                event["_state_prepared"] = value
