"""Batched parameters with the instance's exact base/adapter ownership."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from .grouped_linear import GroupedPlan, check_grouped_policy, make_grouped_plan


_INHERIT_GROUPS = object()


def norm(x, eps=1e-5):
    work = x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    return (work * torch.rsqrt(work.square().mean(-1,keepdim=True)+eps)).to(x.dtype)


@dataclass(frozen=True, slots=True)
class NodeOwners:
    """Device IDs with a trusted static superset of possible host owner IDs.

    The model already knows this superset from its fixed region membership.
    Dynamic selection keeps it unchanged; it is never reconstructed from device
    values. Callers must ensure every actual ID belongs to possible_ids.
    ``exact_ids`` is a stronger, ordered identity, supplied only when actual
    IDs are constructed from known host metadata. It is never inferred from a
    superset or from a dynamic selection's shape. Use ``from_static`` to build
    both representations together. Direct exact annotations are trusted input;
    validation intentionally never reads device IDs back to the host.
    """
    ids: torch.Tensor
    possible_ids: tuple[int, ...]
    exact_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.ids.ndim != 1 or self.ids.dtype != torch.int64:
            raise ValueError("owner IDs must be an int64 vector")
        if not isinstance(self.possible_ids, tuple) or any(type(i) is not int or i < 0 for i in self.possible_ids):
            raise ValueError("possible owner IDs must be a tuple of nonnegative Python integers")
        if self.ids.numel() and not self.possible_ids:
            raise ValueError("nonempty owners require a nonempty possible-ID superset")
        if self.exact_ids is not None:
            if (not isinstance(self.exact_ids, tuple) or len(self.exact_ids) != self.ids.numel()
                    or any(type(i) is not int or i < 0 for i in self.exact_ids)
                    or not set(self.exact_ids) <= set(self.possible_ids)):
                raise ValueError("exact owner order must match the ID shape and belong to the possible superset")

    @classmethod
    def from_static(cls, ordered_ids, *, device, possible_ids=None):
        """Construct actual device IDs from their exact, known host order."""
        ordered = tuple(ordered_ids)
        possible = ordered if possible_ids is None else tuple(possible_ids)
        return cls(torch.tensor(ordered, dtype=torch.int64, device=device), possible, ordered)


@dataclass(frozen=True, slots=True)
class _GroupedOwners(NodeOwners):
    """A single FFN invocation reuses only its discrete owner plan."""
    parameter_groups: tuple[int, ...] = ()
    base_plan: GroupedPlan | None = None
    private_plan: GroupedPlan | None = None
    rows_per_owner: int = 0


def owner_ids(owners):
    return owners.ids if isinstance(owners, NodeOwners) else owners


def select_owners(owners, selected):
    ids = owner_ids(owners)[selected]
    # The dynamic indices may reorder/repeat/drop owners, even at equal length.
    # Keep only the superset; Full selection must continue using real gathers.
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
        self._view_indices = {}

    def _uniform_group(self, owners):
        if not isinstance(owners, NodeOwners):
            return None
        possible = owners.possible_ids
        if possible not in self._uniform_groups:
            groups = {self._host_groups[i] for i in possible}
            self._uniform_groups[possible] = next(iter(groups)) if len(groups) == 1 else None
        return self._uniform_groups[possible]

    def _parameter_index(self, owners, *, grouped):
        """A view only for an explicitly exact consecutive parameter row order.

        Uniform bases keep their existing MM path. Repeated groups, duplicate
        owners, permutations and arbitrary dynamic Full retain tensor gathers.
        """
        exact = owners.exact_ids if isinstance(owners, NodeOwners) else None
        if exact is not None:
            key = (exact, grouped)
            if key not in self._view_indices:
                rows = tuple(self._host_groups[i] for i in exact) if grouped else exact
                start = rows[0] if rows else 0
                consecutive = all(row == start+offset for offset, row in enumerate(rows))
                self._view_indices[key] = slice(start, start+len(rows)) if consecutive else None
            span = self._view_indices[key]
            if span is not None:
                return span
        ids = owner_ids(owners)
        return self.groups[ids] if grouped else ids

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._remember_groups()


class NodeLinear(_OwnerGroups):
    """One mathematical matrix per node; optional shared base plus private LR.

    The node axis is first. All intervening axes are batched, never dispatched
    row by row. Adapters remain two projections, not materialized dense weights.
    """

    def __init__(self, nodes, in_features, out_features, groups=None, rank=8, bias=False, *, weight_access="gather"):
        super().__init__()
        if weight_access not in {"gather", "grouped"}:
            raise ValueError("weight_access must be gather or grouped")
        self.weight_access = weight_access
        self.nodes, self.in_features, self.out_features = nodes, in_features, out_features
        self.shared = groups is not None
        # Ownership is construction metadata, even during a meta-device capacity
        # plan. Do not call item()/tolist() on a meta tensor to discover it.
        indices = (torch.arange(nodes, device="cpu") if groups is None else
                   torch.as_tensor(groups, dtype=torch.long, device="cpu"))
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
        if self.weight_access == "grouped":
            return self._grouped_forward(x, self._grouped_owners(x, owners))
        ids, uniform = owner_ids(owners), self._uniform_group(owners)
        shape = x.shape
        flat = x.reshape(len(ids), -1, self.in_features)
        if uniform is None:
            groups = self._parameter_index(owners, grouped=True)
            out = torch.bmm(flat, self.weight[groups].transpose(1, 2))
        else:
            out = torch.mm(flat.reshape(-1, self.in_features), self.weight[uniform].T)
            out = out.reshape(len(ids), -1, self.out_features)
        if self.shared:
            private = self._parameter_index(owners, grouped=False)
            out = out + torch.bmm(torch.bmm(flat, self.right[private].transpose(1, 2)),
                                  self.left[private].transpose(1, 2))
        if self.bias is not None:
            out = out + (self.bias[groups, None, :] if uniform is None else self.bias[uniform])
        return out.reshape(*shape[:-1], self.out_features)

    def _grouped_owners(self, x, owners):
        """Only local per-call plans are reused; parameter reloads cannot stale them."""
        check_grouped_policy(x, self.weight)
        rows = math.prod(x.shape[1:-1])
        if (isinstance(owners, _GroupedOwners) and owners.parameter_groups == self._host_groups
                and owners.rows_per_owner == rows and self.shared == (owners.private_plan is not None)):
            return owners
        ids = owner_ids(owners)
        possible = owners.possible_ids if isinstance(owners, NodeOwners) else tuple(range(self.nodes))
        exact = owners.exact_ids if isinstance(owners, NodeOwners) else None
        annotated = NodeOwners(ids, possible, exact)
        base = (None if self._uniform_group(annotated) is not None else
                make_grouped_plan(ids, self.groups, self._host_groups, possible, rows, self.weight.shape[0]))
        private = (make_grouped_plan(ids, None, (), possible, rows, self.nodes) if self.shared else None)
        return _GroupedOwners(ids, possible, exact, self._host_groups, base, private, rows)

    def _grouped_forward(self, x, owners):
        ids = owners.ids
        flat = x.reshape(len(ids), owners.rows_per_owner, self.in_features)
        if owners.base_plan is None:
            uniform = self._uniform_group(owners)
            out = torch.mm(flat.reshape(-1, self.in_features), self.weight[uniform].T)
            out = out.reshape(len(ids), owners.rows_per_owner, self.out_features)
        else:
            out = owners.base_plan.apply(flat, self.weight)
        if self.shared:
            low = owners.private_plan.apply(flat, self.right)
            out = out + owners.private_plan.apply(low, self.left)
        if self.bias is not None:
            out = out + self.bias[self.groups[ids], None, :]
        return out.reshape(*x.shape[:-1], self.out_features)

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
        self.shared = groups is not None
        indices = (torch.arange(nodes, device="cpu") if groups is None else
                   torch.as_tensor(groups, dtype=torch.long, device="cpu"))
        self._set_groups(indices)
        count = int(indices.max()) + 1 if nodes else 0
        self.value = nn.Parameter(torch.full((count, width), float(value)))

    def forward(self, owners=None):
        uniform = self._uniform_group(owners)
        if uniform is not None:
            return self.value[uniform].expand(len(owner_ids(owners)), -1)
        return self.value[self.groups if owners is None else self._parameter_index(owners, grouped=True)]

    def selected(self, owners):
        return self(owners)


def fused_node_linears(layers, x, owners=None):
    """Concatenate compatible projections, preserving all private adapters.

    The head/branch axis is batched only for the private adapter's second
    contraction. No adapter is shared or collapsed into a dense base matrix.
    """
    first = layers[0]
    if any(layer.weight_access != "gather" for layer in layers):
        raise ValueError("fused node projections currently require gather weight access")
    if any(layer.in_features != first.in_features or layer.out_features != first.out_features or
           layer._host_groups != first._host_groups or layer.shared != first.shared or
           (layer.bias is None) != (first.bias is None) for layer in layers):
        raise ValueError("fused node projections require compatible dimensions and ownership")
    if owners is None:
        owners = torch.arange(first.nodes, device=first.groups.device)
    ids, uniform = owner_ids(owners), first._uniform_group(owners)
    n, branches, width = len(ids), len(layers), first.out_features
    flat = x.reshape(n, -1, first.in_features)
    if uniform is None:
        groups = first._parameter_index(owners, grouped=True)
        weights = torch.cat([layer.weight[groups] for layer in layers], 1)
        output = torch.bmm(flat, weights.transpose(1, 2))
    else:
        weights = torch.cat([layer.weight[uniform] for layer in layers], 0)
        output = torch.mm(flat.reshape(-1, first.in_features), weights.T)
        output = output.reshape(n, -1, branches * width)
    if first.shared:
        rank = first.right.shape[1]
        if any(layer.right.shape[1] != rank for layer in layers):
            raise ValueError("adapter ranks must agree")
        private_index = first._parameter_index(owners, grouped=False)
        right = torch.cat([layer.right[private_index] for layer in layers], 1)
        low = torch.bmm(flat, right.transpose(1, 2))
        low = low.reshape(n, -1, branches, rank).transpose(1, 2).reshape(n * branches, -1, rank)
        left = torch.stack([layer.left[private_index] for layer in layers], 1).reshape(n * branches, width, rank)
        private = torch.bmm(low, left.transpose(1, 2))
        private = private.reshape(n, branches, -1, width).transpose(1, 2).reshape(n, -1, branches * width)
        output = output + private
    if first.bias is not None:
        bias = (torch.cat([layer.bias[groups] for layer in layers], 1)[:, None] if uniform is None else
                torch.cat([layer.bias[uniform] for layer in layers], 0))
        output = output + bias
    return output.reshape(*x.shape[:-1], branches * width)


class HeadQuery(nn.Module):
    """KV query adapters belong to each mathematical head, not the packed matrix."""
    def __init__(self, nodes, d, groups=None, rank=8, *, out_features=None):
        super().__init__()
        width = d if out_features is None else out_features
        if type(width) is not int or width < 1 or width % 32:
            raise ValueError("query width must be a positive multiple of 32")
        self.heads = nn.ModuleList(NodeLinear(nodes, d, 32, groups, rank) for _ in range(width // 32))

    def uses_fused_projection(self, x):
        # Fixed NPU witnesses showed fewer launches and 3–6% complete training
        # gains. CPU and optional sharded wrappers keep their measured paths.
        return x.device.type == "npu" and len(self.heads) >= 4 and all(
            isinstance(head, NodeLinear) for head in self.heads)

    def forward(self, x, owners=None):
        if self.uses_fused_projection(x):
            return fused_node_linears(self.heads, x, owners)
        return torch.cat([head(x, owners) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, nodes, d, groups=None, rank=8, *, hidden_features=None, weight_access="gather"):
        super().__init__()
        width = 4*d if hidden_features is None else hidden_features
        if type(width) is not int or width < 1 or width % 16:
            raise ValueError("FFN width must be a positive multiple of 16")
        self.weight_access = weight_access
        self.gate = NodeLinear(nodes, d, width, groups, rank, weight_access=weight_access)
        self.up = NodeLinear(nodes, d, width, groups, rank, weight_access=weight_access)
        self.down = NodeLinear(nodes, width, d, groups, rank, weight_access=weight_access)

    def forward(self, x, owners=None):
        z = norm(x)
        if self.weight_access == "grouped" and all(isinstance(layer, NodeLinear) for layer in (self.gate, self.up, self.down)):
            if owners is None:
                owners = torch.arange(self.gate.nodes, device=x.device)
            owners = self.gate._grouped_owners(z, owners)
        return x + self.down(torch.nn.functional.silu(self.gate(z, owners)) * self.up(z, owners), owners)
