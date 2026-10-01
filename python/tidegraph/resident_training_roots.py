"""Shape-preserving consumer roots; no event lookup or numerical CPU decisions."""

def pair(item, name, value, connected, valid):
    if value is None:
        if connected is not None:
            raise ValueError("a cotangent connection mask requires values")
    else:
        setattr(item, name, value)
        setattr(item, name + "_connected", valid if connected is None else connected)


def caches(module, values, windows):
    values = list(values)
    if len(values) != len(windows):
        raise ValueError("cache cotangents must match the window's cache groups")
    packed = []
    for fields, group in zip(values, windows):
        fields = {} if fields is None else dict(fields)
        if set(fields) - {"key", "value", "log_bias", "key_connected", "value_connected", "log_bias_connected"}:
            raise ValueError("unknown cache cotangent field")
        item = module.CacheCotangents()
        for name in ("key", "value", "log_bias"):
            pair(item, name, fields.get(name), fields.get(name + "_connected"), group.present)
        packed.append(item)
    return packed


def cotangents(module, window, *, outputs=None, outputs_connected=None, pending=None,
               pending_connected=None, final=None, final_connected=None, cache=None, states=None):
    item = module.Cotangents()
    item.token = window.token
    if window.states and (final is not None or final_connected is not None or cache is not None):
        raise ValueError("sharded sessions require owner-local state/cache roots")
    if not window.states and states is not None:
        raise ValueError("single-device sessions use dense state/cache roots")
    for name, value, connected, valid in (
            ("outputs", outputs, outputs_connected, window.outputs.valid),
            ("pending", pending, pending_connected, window.pending_valid),
            ("final", final, final_connected, window.state_present)):
        pair(item, name, value, connected, valid)
    if cache is not None:
        item.cache = caches(module, cache, window.cache)
    if states is not None:
        states = list(states)
        if len(states) != len(window.states):
            raise ValueError("state cotangents must match all owner windows")
        packed = []
        for fields, state in zip(states, window.states):
            fields = {} if fields is None else dict(fields)
            if set(fields) - {"final", "final_connected", "cache"}:
                raise ValueError("unknown state cotangent field")
            root = module.StateCotangents()
            pair(root, "final", fields.get("final"), fields.get("final_connected"), state.present)
            if fields.get("cache") is not None:
                root.cache = caches(module, fields["cache"], state.cache)
            packed.append(root)
        item.states = packed
    return item
