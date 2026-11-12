"""Block-owner capacity identity, separate from the old event queue layout."""
from .resident_policy import BLOCK_LIMITS
from .legacy_topology import integer


def resident_block_capacity(plan, devices, node_devices, limits):
    if len(devices) != 1 or any(x != devices[0] for x in node_devices):
        raise ValueError("resident block capacity requires one device")
    if set(limits)-BLOCK_LIMITS.keys():
        raise ValueError("unknown resources.resident_block_limits field")
    resolved = dict(BLOCK_LIMITS, **limits)
    for key, value in resolved.items():
        integer(value, "resources.resident_block_limits." + key)
    batch, cache, graph = plan["samples"], resolved["cache"], plan["graph"]
    raw, layouts = 0, []
    for v, c in enumerate(plan["memories"]):
        if c.get("memory_family", "kv") != "kv" or c["memory_read"] not in {"sum", "qry"}:
            raise ValueError("block capacity requires KV sum/qry")
        floats = c["width"]+1+cache*(2*c["width"]+c["summary_width"]+1)
        integers = 4+cache
        raw += batch*(4*floats+8*integers)
        layouts.append(dict(node=v, cache_capacity=cache, floating_width=floats, int64_width=integers))
    metadata = batch*(graph["nodes"]+8*graph["regions"]+16*len(graph["inputs"])+8)
    # The C++ total partition includes canonical/role/state/retained tensors
    # plus its optimizer partition. Vendor workspace is separately limited.
    return dict(schema="tide-resident-block-reservation-v2", limits=resolved,
                coordinator=devices[0], node_private_preflight_bytes={devices[0]: 2*raw},
                boundary_metadata_bytes=metadata, raw_layouts=layouts,
                persistent_raw_lower_bound_bytes=2*raw+metadata,
                reserved_total_bytes=resolved["total_bytes"], workspace_bytes=resolved["workspace_bytes"],
                static_rejections=(["resident_block_limits.total_bytes is below initial and bound raw-state storage"]
                    if 2*raw+metadata > resolved["total_bytes"] else []),
                basis="B×L blocks, exact padded initial/bound raw layout; explicit total/workspace reservations",
                extra_unknown=["source role padding and numerical tape peaks", "allocator/vendor context",
                               "device program/task resources; no old queue or stages"])
