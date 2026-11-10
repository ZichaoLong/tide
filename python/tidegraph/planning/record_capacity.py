"""Live event reservations for the audited host component-window schedule.

This estimates event records, not vendor workspaces or a hardware memory cap.
No activity/selection trace is consulted. All possible candidates are counted.
"""


def event_reservations(plan, spec, placement, node_bytes, *, mode, trace_records):
    if type(trace_records) is not bool:
        raise ValueError("trace_records must be an explicit bool")
    selected = (not trace_records and mode == "inference"
                and spec.client in {"pytorch", "python-native"}
                and spec.placement != "resident" and spec.return_mode != "complete"
                and spec.schedule == "prefill" and spec.prefill_schedule != "greedy"
                and plan["graph"]["kind"] == "ranked-local")
    by_device = {}
    for device, owner in placement["per_device"].items():
        complete = sum(node_bytes[v] for v in owner["nodes"])
        if selected:
            regions = {}
            for v in owner["nodes"]:
                region = plan["graph"]["node_regions"][v]
                regions[region] = regions.get(region, 0)+node_bytes[v]
            # Both schedulers process regions in sequence. Python can retain
            # the previous block while evaluating the next RHS; reserve two
            # largest blocks, allowing a repeated region, plus the final output
            # estimate (added by the caller). Raw state and messages are separate.
            live = min(complete, 2*max(regions.values(), default=0))
        else:
            live = complete
        by_device[device] = live
    return by_device, selected, ("selected inference: two largest regional event blocks plus outputs"
                                 if selected else "complete internal event records")
