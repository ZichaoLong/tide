"""Batched parameters with the instance's exact base/adapter ownership."""
from dataclasses import dataclass
import math

import torch
from torch import nn


def norm(x, eps=1e-5):
    work = x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    return (work * torch.rsqrt(work.square().mean(-1,keepdim=True)+eps)).to(x.dtype)


@dataclass(frozen=True, slots=True)
class NodeOwners:
    """Device IDs with a trusted static superset of possible host owner IDs.

    The model already knows this superset from its fixed region membership.
    Dynamic selection keeps it unchanged; it is never reconstructed from device
    values. Callers must ensure every actual ID belongs to possible_ids.
    """
    ids: torch.Tensor
    possible_ids: tuple[int, ...]

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64:
            raise ValueError("owner IDs must be an int64 vector")
        if not isinstance(self.possible_ids, tuple) or any(type(i) is not int or i < 0 for i in self.possible_ids):
            raise ValueError("possible owner IDs must be a tuple of nonnegative Python integers")
        if self.ids.numel() and not self.possible_ids:
            raise ValueError("nonempty owners require a nonempty possible-ID superset")


def owner_ids(owners):
    return owners.ids if isinstance(owners, NodeOwners) else owners


def select_owners(owners, selected):
    ids = owner_ids(owners)[selected]
    return NodeOwners(ids, owners.possible_ids) if isinstance(owners, NodeOwners) else ids


class _OwnerGroups(nn.Module):
    def _set_groups(self, indices):
        self.register_buffer("groups", indices)
        self._remember_groups()

    def _remember_groups(self):
        # Construction/checkpoint loading may read host metadata; forward never
        # inspects device IDs to select the shared-base implementation.
        self._host_groups = tuple(self.groups.detach().cpu().tolist())
        self._uniform_groups = {}

    def _uniform_group(self, owners):
        if not isinstance(owners, NodeOwners):
            return None
        possible = owners.possible_ids
        if possible not in self._uniform_groups:
            groups = {self._host_groups[i] for i in possible}
            self._uniform_groups[possible] = next(iter(groups)) if len(groups) == 1 else None
        return self._uniform_groups[possible]

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._remember_groups()


class NodeLinear(_OwnerGroups):
    """One mathematical matrix per node; optional shared base plus private LR.

    The node axis is first. All intervening axes are batched, never dispatched
    row by row. Adapters remain two projections, not materialized dense weights.
    """

    def __init__(self, nodes, in_features, out_features, groups=None, rank=8, bias=False):
        super().__init__()
        self.nodes, self.in_features, self.out_features = nodes, in_features, out_features
        self.shared = groups is not None
        indices = torch.arange(nodes) if groups is None else torch.as_tensor(groups, dtype=torch.long)
        self._set_groups(indices)
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
        ids, uniform = owner_ids(owners), self._uniform_group(owners)
        shape = x.shape
        flat = x.reshape(len(ids), -1, self.in_features)
        if uniform is None:
            groups = self.groups[ids]
            out = torch.bmm(flat, self.weight[groups].transpose(1, 2))
        else:
            out = torch.mm(flat.reshape(-1, self.in_features), self.weight[uniform].T)
            out = out.reshape(len(ids), -1, self.out_features)
        if self.shared:
            out = out + torch.bmm(torch.bmm(flat, self.right[ids].transpose(1, 2)),
                                  self.left[ids].transpose(1, 2))
        if self.bias is not None:
            out = out + (self.bias[groups, None, :] if uniform is None else self.bias[uniform])
        return out.reshape(*shape[:-1], self.out_features)

    def selected(self, x, owners):
        """Convenience for irregular rows; pack once outside repeated projections."""
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        ids = owner_ids(owners)
        unique, inverse, counts = ids.unique(sorted=True, return_inverse=True, return_counts=True)
        order = inverse.argsort(stable=True)
        starts = counts.cumsum(0) - counts
        slots_sorted = torch.arange(len(ids), device=ids.device) - starts[inverse[order]]
        slots = torch.empty_like(slots_sorted).scatter(0, order, slots_sorted)
        packed = x.new_zeros((len(unique), int(counts.max()), x.shape[-1]))
        packed = packed.index_put((inverse, slots), x)
        selected = NodeOwners(unique, owners.possible_ids) if isinstance(owners, NodeOwners) else unique
        out = self(packed, selected)
        return out[inverse, slots]


class NodeVector(_OwnerGroups):
    def __init__(self, nodes, width, groups=None, value=0.0):
        super().__init__()
        indices = torch.arange(nodes) if groups is None else torch.as_tensor(groups, dtype=torch.long)
        self._set_groups(indices)
        count = int(indices.max()) + 1 if nodes else 0
        self.value = nn.Parameter(torch.full((count, width), float(value)))

    def forward(self, owners=None):
        uniform = self._uniform_group(owners)
        if uniform is not None:
            return self.value[uniform].expand(len(owner_ids(owners)), -1)
        return self.value[self.groups if owners is None else self.groups[owner_ids(owners)]]

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
