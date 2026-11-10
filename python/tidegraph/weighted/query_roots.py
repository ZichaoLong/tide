"""Transpose complete disjoint view partitions before entering private autograd.

This only composes linear view maps. A partial/overlapping/strided partition
keeps its original roots, so absent cotangents and independent fields survive.
"""
from collections import defaultdict
import torch
from . import packing_observation as observation


def coalesce(roots, bars):
    groups = defaultdict(list)
    for i, root in enumerate(roots):
        base = root._base
        if (base is not None and base.requires_grad and base.is_contiguous()
                and root.is_contiguous() and root.numel() and not root.is_conj()
                and not root.is_neg() and root.dtype == base.dtype):
            groups[id(base)].append(i)
    replacements, removed = {}, set()
    for ids in groups.values():
        if len(ids) < 2:
            continue
        base = roots[ids[0]]._base
        ordered = sorted(ids, key=lambda i: roots[i].storage_offset())
        cursor = base.storage_offset()
        for i in ordered:
            if roots[i].storage_offset() != cursor:
                break
            cursor += roots[i].numel()
        else:
            if cursor == base.storage_offset()+base.numel():
                replacements[ids[0]] = (base, torch.cat([bars[i].reshape(-1) for i in ordered]).reshape_as(base))
                removed.update(ids[1:])
                observation.projection("query_vjp_root_partition", len(ids))
    result, gradients = [], []
    for i, (root, bar) in enumerate(zip(roots, bars)):
        if i not in removed:
            root, bar = replacements.get(i, (root, bar))
            result.append(root); gradients.append(bar)
    return result, gradients
