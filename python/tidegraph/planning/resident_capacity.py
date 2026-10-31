"""Torch-free mirror of the public resident raw/journal preflight reservation.

The C++ guard deliberately counts raw floating slots as FP32 even with FP16
storage. Numerical tapes, physical role banks and service resources are extra.
"""
from .legacy_topology import integer


def resident_capacity(plan, devices, node_devices, limits):
    resolved = dict(queue=64, stages=128, cache=64, tensor_bytes=512*1024**2,
                    operator_bytes=64*1024**2, timeout_ms=30000, prefill=True)
    allowed = set(resolved) | {"windows", "retained_bytes", "backward_bytes", "optimizer_bytes", "total_bytes"}
    if not isinstance(limits, (dict, type(None))) or set(limits or {})-allowed:
        raise ValueError("resources.resident_limits contains an unknown field")
    resolved.update(limits or {})
    if type(resolved["prefill"]) is not bool:
        raise ValueError("resources.resident_limits.prefill must be bool")
    for name in ("queue", "stages", "cache", "tensor_bytes", "operator_bytes", "timeout_ms"):
        integer(resolved[name], "resources.resident_limits."+name)
    g, b, d = plan["graph"], plan["samples"], plan["width"]+1
    q, stages, cache = (resolved[k] for k in ("queue", "stages", "cache"))
    atom = q*(49+4*d)
    coordinator = devices[0]
    private = dict.fromkeys(devices, 0)
    control = 2*atom+44+16*b*g["regions"]
    per_stage = 2*atom+24+9*b*g["regions"]+8*(5*q+1)
    boundary = 49*q+8
    layouts = []
    for v, c in enumerate(plan["memories"]):
        family, s = c.get("memory_family", "kv"), c.get("summary_width", 2)
        basic = family in {"affine", "delta"}
        cap = 0 if basic else cache
        if family == "kv":
            floating = d+cache*(2*(d-1)+s+1)
        else:
            h, k, value = c.get("heads", 1), c.get("key_width", 2), c.get("value_width", 3)
            conv = 0 if basic else h*(2*k+value) if family == "gated_delta" else h*value+2*c.get("groups", 1)*k
            floating = d+(2 if basic else h)*k*value+2*s+2+cap*conv
        integer_width = 4+cap
        raw = b*(4*floating+8*integer_width)
        private[node_devices[v]] += 3*raw+8*b
        boundary += 8*b*(integer_width+1)
        r = plan["regions"][g["node_regions"][v]]
        regional = r["selection_mode"] in {"regional_total", "regional_total_zero", "regional_threshold", "regional_threshold_hard"}
        descriptors = 3+(r.get("region_control_width", 2) if regional else 0)
        if r.get("region_score") == "att":
            descriptors += 2*r.get("selection_width", 4)
        per_stage += 2*raw+b*(14+8*d+4*(d-1)+4*descriptors)
        layouts.append(dict(node=v, cache_capacity=cap, floating_width=floating, int64_width=integer_width))
    journal = stages*per_stage+boundary
    total = control+sum(private.values())+journal
    return dict(schema="tide-resident-static-reservation-v1", limits=resolved, coordinator=coordinator,
                node_private_preflight_bytes=private, coordinator_queue_control_bytes=control,
                coordinator_journal_bytes=journal, preflight_total_bytes=total,
                exceeds_tensor_bytes=total > resolved["tensor_bytes"],
                physical_local_rows_capacity=stages*b*g["nodes"], raw_layouts=layouts,
                basis="weighted_flow_preflight FP32 reservation; actual raw dtype and logical activity remain separate",
                extra_unknown=["numerical tapes", "physical role banks", "peer packets/workspace", "retained backward snapshots",
                               "allocator/vendor context", "service stream/SQ/CQ and task capacity"])
