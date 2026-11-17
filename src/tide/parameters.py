"""Batched parameters with the instance's exact base/adapter ownership."""
import math

import torch
from torch import nn


def norm(x, eps=1e-5):
    work = x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    return (work * torch.rsqrt(work.square().mean(-1,keepdim=True)+eps)).to(x.dtype)


class NodeLinear(nn.Module):
    """One mathematical matrix per node; optional shared base plus private LR.

    The node axis is first. All intervening axes are batched, never dispatched
    row by row. Adapters remain two projections, not materialized dense weights.
    """

    def __init__(self, nodes, in_features, out_features, groups=None, rank=8, bias=False):
        super().__init__()
        self.nodes, self.in_features, self.out_features = nodes, in_features, out_features
        self.shared = groups is not None
        indices = torch.arange(nodes) if groups is None else torch.as_tensor(groups, dtype=torch.long)
        self.register_buffer("groups", indices)
        count = int(indices.max()) + 1 if nodes else 0
        self.weight = nn.Parameter(torch.empty(count, out_features, in_features))
        nn.init.normal_(self.weight, std=1 / math.sqrt(in_features))
        if self.shared:
            r = min(rank, in_features, out_features)
            self.left = nn.Parameter(torch.empty(nodes, out_features, r))
            self.right = nn.Parameter(torch.empty(nodes, r, in_features))
            nn.init.normal_(self.left, std=0.02 / math.sqrt(r))
            nn.init.normal_(self.right, std=1 / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(count, out_features)) if bias else None

    def forward(self, x, owners=None):
        if owners is None:
            owners = torch.arange(self.nodes, device=self.groups.device)
        groups = self.groups[owners]
        shape = x.shape
        flat = x.reshape(len(owners), -1, self.in_features)
        out = torch.bmm(flat, self.weight[groups].transpose(1, 2))
        if self.shared:
            out = out + torch.bmm(torch.bmm(flat, self.right[owners].transpose(1, 2)),
                                  self.left[owners].transpose(1, 2))
        if self.bias is not None:
            out = out + self.bias[groups, None, :]
        return out.reshape(*shape[:-1], self.out_features)

    def selected(self, x, owners):
        """Convenience for irregular rows; pack once outside repeated projections."""
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        unique, inverse, counts = owners.unique(sorted=True, return_inverse=True, return_counts=True)
        order = inverse.argsort(stable=True)
        starts = counts.cumsum(0) - counts
        slots_sorted = torch.arange(len(owners), device=owners.device) - starts[inverse[order]]
        slots = torch.empty_like(slots_sorted).scatter(0, order, slots_sorted)
        packed = x.new_zeros((len(unique), int(counts.max()), x.shape[-1]))
        packed = packed.index_put((inverse, slots), x)
        out = self(packed, unique)
        return out[inverse, slots]


class NodeVector(nn.Module):
    def __init__(self, nodes, width, groups=None, value=0.0):
        super().__init__()
        indices = torch.arange(nodes) if groups is None else torch.as_tensor(groups, dtype=torch.long)
        self.register_buffer("groups", indices)
        count = int(indices.max()) + 1 if nodes else 0
        self.value = nn.Parameter(torch.full((count, width), float(value)))

    def forward(self, owners=None):
        return self.value[self.groups if owners is None else self.groups[owners]]

    def selected(self, owners):
        return self(owners)


class HeadQuery(nn.Module):
    """KV query adapters belong to each mathematical head, not the packed matrix."""
    def __init__(self, nodes, d, groups=None, rank=8):
        super().__init__()
        self.heads = nn.ModuleList(NodeLinear(nodes, d, 32, groups, rank) for _ in range(d // 32))

    def forward(self, x, owners=None):
        return torch.cat([head(x, owners) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, nodes, d, groups=None, rank=8):
        super().__init__()
        self.gate = NodeLinear(nodes, d, 4*d, groups, rank)
        self.up = NodeLinear(nodes, d, 4*d, groups, rank)
        self.down = NodeLinear(nodes, 4*d, d, groups, rank)

    def forward(self, x, owners=None):
        z = norm(x)
        return x + self.down(torch.nn.functional.silu(self.gate(z, owners)) * self.up(z, owners), owners)
