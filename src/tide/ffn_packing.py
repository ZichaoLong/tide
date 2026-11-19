"""Optional row packing confined to the positionwise Full FFN.

Attention retains its node/sample/query layout. A node's selected sample/time
rows may share one FFN matrix, so only that boundary combines those axes.
"""
from dataclasses import dataclass

import torch

from .parameters import select_owners


def validate_ffn_packing(packing):
    if packing not in ("rectangular", "power2"):
        raise ValueError("ffn_packing must be rectangular or power2")
    return packing


@dataclass(frozen=True)
class FFNRowBucket:
    """One disjoint owner set and its rows in the original selected-row order."""
    nodes: torch.Tensor
    rows: torch.Tensor
    local_node: torch.Tensor
    slot: torch.Tensor
    width: int


def power2_buckets(queries):
    """Group positive per-node row counts without reading counts on the host.

    Widths are 1, 2, 4, ... with the last width capped at the original B*Q.
    Every selected owner appears once, so executed rows never exceed the old
    rectangle and are less than twice the selected rows. The nonzero operations
    still have dynamic-size/synchronization costs; this is not a resident kernel.
    """
    nodes, batch, length = queries.valid.shape
    limit = batch * length
    valid = queries.valid.reshape(nodes, limit)
    counts = valid.sum(-1)
    source = queries.batch * length + queries.slot
    slots = (valid.to(torch.int64).cumsum(-1) - 1)[queries.node, source]
    lower, upper = 0, 1
    buckets = []
    while lower < limit:
        upper = min(upper, limit)
        members = (counts > lower) & (counts <= upper)
        selected_nodes = members.nonzero().flatten()
        if selected_nodes.numel():
            rows = members[queries.node].nonzero().flatten()
            local = (members.to(torch.int64).cumsum(0) - 1)[queries.node[rows]]
            buckets.append(FFNRowBucket(selected_nodes, rows, local, slots[rows], upper))
        lower, upper = upper, upper * 2
    return buckets


def selected_ffn(ffn, x, queries, owners, *, packing="rectangular"):
    """Apply one FFN with a shared packing map for gate, up and down.

    Return the existing rectangle for Queries.publish. Its unused rows are zero
    in the compact path; only published rows are observable. No history, control,
    original time, parameter ownership or state update is changed here.
    """
    validate_ffn_packing(packing)
    if packing == "rectangular" or queries.node.numel() == queries.valid.numel():
        # Fully selected rectangles, including B1/L1, gain nothing from packing.
        return ffn(x, owners)
    values = x[queries.node, queries.batch, queries.slot]
    output_rows, output_values = [], []
    for bucket in power2_buckets(queries):
        compact = x.new_zeros((bucket.nodes.numel(), bucket.width, x.shape[-1]))
        compact = compact.index_put((bucket.local_node, bucket.slot), values[bucket.rows])
        result = ffn(compact, select_owners(owners, bucket.nodes))
        output_rows.append(bucket.rows)
        output_values.append(result[bucket.local_node, bucket.slot])
    rows = torch.cat(output_rows)
    result = torch.cat(output_values)
    return x.new_zeros(x.shape).index_put(
        (queries.node[rows], queries.batch[rows], queries.slot[rows]), result)
