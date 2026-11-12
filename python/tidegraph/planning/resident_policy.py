"""Allocation-free experiment selection; the actual Model is checked again.

No observed route, reference execution or runtime failure participates in this
choice. Explicit old queue/stage budgets continue to select their old owner.
"""
from .legacy_topology import integer

BLOCK_LIMITS = dict(cache=64, forward_bytes=512 << 20, backward_bytes=1024 << 20,
                    optimizer_bytes=128 << 20, total_bytes=3 << 30,
                    workspace_bytes=512 << 20, windows=4, timeout_ms=30000)


def block_reason(plan, spec, family, task):
    if spec.client not in {"python-native", "libtorch"}:
        return "this experiment client retains its general resident entry"
    if spec.client == "libtorch" and spec.return_mode == "complete":
        return "standalone block requires outputs/metrics; complete records retain the general owner"
    if family != "settle":
        return "requires the validated SettleGraph ports view"
    if spec.schedule != "prefill" or spec.input_interface != "auto":
        return "requires prefill with the tensor input interface"
    if spec.dtype != "float32" or spec.peer_devices:
        return "requires FP32 and one device"
    if task.get("plugin"):
        return "task plugin retains its declared record interface"
    if any(c.get("support_policy", "strict") != "strict" for c in plan["regions"]):
        return "requires strict mathematical support"
    groups = [[] for _ in plan["regions"]]
    for v, c in enumerate(plan["memories"]):
        if c.get("memory_family", "kv") != "kv" or c["memory_read"] not in {"sum", "qry"}:
            return "requires KV sum/query sequence capability for every node"
        groups[plan["graph"]["node_regions"][v]].append(c)
    if any(any(c != rows[0] for c in rows) for rows in groups):
        return "requires compatible KV geometry within each region"
    return None


def select(plan, spec, family, task, resources):
    requested = spec.resident_executor
    old, block = resources.get("resident_limits"), resources.get("resident_block_limits")
    if old is not None and block is not None:
        raise ValueError("resident_limits and resident_block_limits describe different owners; choose one")
    reason = ("explicit general resident owner" if requested == "general" else
              "explicit historical resident_limits retain the general owner" if old is not None else
              block_reason(plan, spec, family, task))
    if reason is not None and (requested == "block-v2" or block is not None):
        raise ValueError("resident block selection refused: " + reason)
    effective = "general" if reason is not None else "block-v2"
    return dict(schema="tide-experiment-resident-selection-v1", requested=requested,
                effective=effective, reason=reason or "compatible complete B×L Settle ports",
                capacity_field="resident_limits" if effective == "general" else "resident_block_limits",
                requested_limits=dict((old if effective == "general" else block) or {}))


def block_limits_defaults(plan, resources):
    result = dict(BLOCK_LIMITS)
    # This is a storage bound, never a limit on cumulative logical time. Empty
    # factory state can add at most one row per position and per node.
    writes = resources["logical_windows"] * plan["positions"]
    needed = max(min(writes, c["window"]) if c["window"] else writes for c in plan["memories"])
    result["cache"] = max(1, needed)
    result.update(resources.get("resident_block_limits", {}))
    for key, value in result.items():
        integer(value, "resources.resident_block_limits." + key)
    return result
