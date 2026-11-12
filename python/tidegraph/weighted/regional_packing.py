"""Host int64 incidence packing for complete regional candidate rows.

Rows with the same real candidate count can share one R×N kernel even when
their members differ. Missing candidates are never padded with numeric zeros.
This module owns linear layout operations, not regional numerical semantics.
"""
from dataclasses import dataclass
import torch


def _ids(values, device):
    return torch.tensor(values, dtype=torch.int64, device=device)


@dataclass(frozen=True)
class PackedRegion:
    fields: dict
    nodes: torch.Tensor
    group: object
    sources: dict

    def pullback(self, bars, support=None):
        """Linear transpose with explicit structural support, never zero tests.

        support is bool R or R×N; None means every packed coordinate. A source
        column with no supported coordinates returns None, while an explicitly
        zero cotangent on supported coordinates remains a connected zero.
        Ordinary autograd on a whole packed Tensor uses whole-column roots;
        independent event-root adapters must carry their support explicitly.
        """
        if set(bars)-set(self.fields):
            raise ValueError("unknown regional packed field adjoint")
        g = self.group
        shape = (len(g.rows), g.candidates)
        if support is None:
            enabled = [True]*(shape[0]*shape[1])
        else:
            if support.dtype != torch.bool or tuple(support.shape) not in (shape, shape[:1]):
                raise ValueError("regional support must be bool R or R,N")
            if support.ndim == 1:
                support = support[:, None].expand(shape)
            enabled = support.detach().cpu().flatten().tolist()
        result = {node: {name: None for name in self.fields} for node in g.participants}
        for name, packed in self.fields.items():
            bar = bars.get(name)
            if bar is None:
                continue
            if bar.shape != packed.shape or (bar.dtype, bar.device) != (packed.dtype, packed.device):
                raise ValueError("regional field adjoint metadata mismatch")
            flat = bar.reshape(shape[0]*shape[1], *bar.shape[2:])
            for node in g.participants:
                positions = [i for i, owner in enumerate(g.node_ids) if owner == node and enabled[i]]
                if not positions:
                    continue
                target = [g.local_rows[i] for i in positions]
                result[node][name] = torch.zeros_like(self.sources[node][name]).index_copy(
                    0, _ids(target, bar.device), flat.index_select(0, _ids(positions, bar.device)))
        return result


@dataclass(frozen=True)
class RegionalGroup:
    rows: tuple
    candidates: int
    participants: tuple
    domains: dict
    node_ids: tuple
    local_rows: tuple
    gather: tuple

    def pack(self, sources):
        """Each field uses whole node columns and one complete gather.

        The aligned case uses a direct stack (or view for one node). Tensor
        identities may repeat across different nodes; their uses are retained.
        """
        if any(node not in sources for node in self.participants):
            raise ValueError("regional descriptor source missing")
        names = tuple(sorted(sources[self.participants[0]]))
        if not names:
            raise ValueError("regional descriptors require at least one field")
        ref = sources[self.participants[0]][names[0]]
        if not ref.is_floating_point():
            raise ValueError("regional descriptors must be floating tensors")
        for node in self.participants:
            if tuple(sorted(sources[node])) != names:
                raise ValueError("regional descriptor fields differ")
            for name in names:
                value, expected = sources[node][name], sources[self.participants[0]][name]
                if (value.ndim < 1 or len(value) != len(self.domains[node]) or value.shape[1:] != expected.shape[1:]
                        or (value.dtype, value.device) != (ref.dtype, ref.device)):
                    raise ValueError("regional descriptor metadata mismatch")
        aligned = len(self.participants) == self.candidates and all(self.domains[n] == self.rows for n in self.participants)
        indices = None if aligned else _ids(self.gather, ref.device)
        fields = {}
        for name in names:
            values = [sources[node][name] for node in self.participants]
            if aligned:
                fields[name] = values[0].unsqueeze(1) if len(values) == 1 else torch.stack(values, 1)
            else:
                fields[name] = torch.cat(values).index_select(0, indices).reshape(
                    len(self.rows), self.candidates, *values[0].shape[1:])
        nodes = _ids(self.node_ids, ref.device).reshape(len(self.rows), self.candidates)
        return PackedRegion(fields, nodes, self, {n: sources[n] for n in self.participants})


class RegionalPacking:
    """A complete incidence plan with stable ascending node IDs in each row."""
    def __init__(self, rows_by_node):
        self.domains = {}
        by_row = {}
        for node in sorted(rows_by_node):
            if type(node) is not int or not 0 <= node < 2**63:
                raise ValueError("regional node ID must be a nonnegative int64")
            rows = rows_by_node[node]
            if rows.ndim != 1 or rows.dtype != torch.int64 or rows.device.type != "cpu":
                raise ValueError("regional row domains must be host int64 vectors")
            domain = tuple(rows.tolist())
            if any(row < 0 for row in domain) or any(a >= b for a, b in zip(domain, domain[1:])):
                raise ValueError("regional row domains must be sorted, unique and nonnegative")
            self.domains[node] = domain
            for local, row in enumerate(domain):
                by_row.setdefault(row, []).append((node, local))
        cardinalities = {}
        for row, candidates in sorted(by_row.items()):
            cardinalities.setdefault(len(candidates), []).append(row)
        self.groups = []
        for count, rows in sorted(cardinalities.items()):
            pairs = [pair for row in rows for pair in by_row[row]]
            participants = tuple(sorted({node for node, _ in pairs}))
            offsets, offset = {}, 0
            for node in participants:
                offsets[node] = offset
                offset += len(self.domains[node])
            self.groups.append(RegionalGroup(tuple(rows), count, participants,
                {n: self.domains[n] for n in participants}, tuple(n for n, _ in pairs),
                tuple(i for _, i in pairs), tuple(offsets[n]+i for n, i in pairs)))
        # Every node/domain coordinate appears exactly once in this inverse.
        inverse = {node: [None]*len(domain) for node, domain in self.domains.items()}
        offset = 0
        for group in self.groups:
            for position, (node, local) in enumerate(zip(group.node_ids, group.local_rows)):
                inverse[node][local] = offset+position
            offset += len(group.node_ids)
        self.inverse = {node: tuple(ids) for node, ids in inverse.items()}

    def _validate_controls(self, values, like=None):
        if len(values) != len(self.groups):
            raise ValueError("regional control group count differs")
        if not values:
            if self.domains and like is None:
                raise ValueError("empty regional controls require a reference tensor")
            return like
        ref = values[0]
        if not ref.is_floating_point():
            raise ValueError("regional controls must be floating")
        for group, value in zip(self.groups, values):
            if value.shape != (len(group.rows), group.candidates) or (value.dtype, value.device) != (ref.dtype, ref.device):
                raise ValueError("regional control metadata mismatch")
        return ref

    def unpack(self, values, *, like=None):
        """Return each node's complete control column, in its original row order."""
        ref = self._validate_controls(values, like)
        if not self.groups:
            return {node: ref.new_empty(0) for node in self.domains}
        group = self.groups[0]
        if (len(self.groups) == 1 and len(self.domains) == len(group.participants) == group.candidates
                and all(group.domains[n] == group.rows for n in group.participants)):
            return dict(zip(group.participants, values[0].unbind(1)))
        result = _Unpack.apply(self, *values)
        return dict(zip(self.domains, result))

    def controls_pullback(self, bars, like):
        """Transpose node columns to groups plus bool R dependency support.

        Missing/None node bars are disconnected. Numeric zeros are never used
        as a support predicate. This is linear publication support; a regional
        kernel's own dependency rules still govern descriptor/parameter VJPs.
        """
        if set(bars)-set(self.domains):
            raise ValueError("unknown regional node adjoint")
        active = {node: bar for node, bar in bars.items() if bar is not None}
        for node, bar in active.items():
            if bar.shape != (len(self.domains[node]),) or (bar.dtype, bar.device) != (like.dtype, like.device):
                raise ValueError("regional node adjoint metadata mismatch")
        offsets, offset = {}, 0
        for node in sorted(active):
            offsets[node] = offset
            offset += len(self.domains[node])
        bank = torch.cat([active[n] for n in sorted(active)]) if active else None
        outputs, supports = [], []
        for group in self.groups:
            positions = [i for i, node in enumerate(group.node_ids) if node in active]
            supported = [any(node in active for node in group.node_ids[start:start+group.candidates])
                         for start in range(0, len(group.node_ids), group.candidates)]
            supports.append(torch.tensor(supported, dtype=torch.bool, device=like.device))
            if not positions:
                outputs.append(None)
                continue
            source = [offsets[group.node_ids[i]]+group.local_rows[i] for i in positions]
            value = like.new_zeros(len(group.node_ids)).index_copy(
                0, _ids(positions, like.device), bank.index_select(0, _ids(source, like.device)))
            outputs.append(value.reshape(len(group.rows), group.candidates))
        return outputs, supports


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packing, *values):
        ctx.set_materialize_grads(False)
        ctx.packing = packing
        ctx.prototype = values[0].new_empty(0)
        group = packing.groups[0]
        aligned = (len(values) == 1 and len(packing.domains) == group.candidates
                   and all(domain == group.rows for domain in packing.domains.values()))
        if aligned:
            outputs = values[0].unbind(1)
        else:
            bank = values[0].reshape(-1) if len(values) == 1 else torch.cat([value.reshape(-1) for value in values])
            outputs = tuple(bank.index_select(0, _ids(ids, bank.device)) for ids in packing.inverse.values())
        differentiable = {node for group, value in zip(packing.groups, values) if value.requires_grad
                          for node in group.participants}
        ctx.mark_non_differentiable(*(value for node, value in zip(packing.domains, outputs) if node not in differentiable))
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        result, _ = ctx.packing.controls_pullback(dict(zip(ctx.packing.domains, bars)), ctx.prototype)
        return None, *result
