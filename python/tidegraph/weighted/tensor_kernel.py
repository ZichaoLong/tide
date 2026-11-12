"""Typed node blocks: pack once, compute once, publish independent field roots.

The private formula is block diagonal in the node axis. Every role is an
independent leaf inside the block; genuine aliases are summed by outer autograd.
This boundary declares first order only. No scalar/reference replay is used.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Role:
    values: tuple
    indices: torch.Tensor | None = None  # [node, B, Q/K], global concatenation ids
    shared: bool = False

    def pack(self):
        if self.shared:
            return self.values[0]
        if self.indices is None:
            return torch.stack(self.values)
        flat = torch.cat(self.values)
        sentinel = flat.new_zeros((1, *flat.shape[1:]))
        return torch.cat((flat, sentinel)).index_select(0, self.indices.flatten()).reshape(
            *self.indices.shape, *flat.shape[1:])

    def transpose(self, bar):
        if self.indices is None:
            return tuple(bar.unbind())
        sizes = [len(x) for x in self.values]
        flat = bar.reshape(self.indices.numel(), *bar.shape[self.indices.ndim:])
        total = bar.new_zeros((sum(sizes)+1, *flat.shape[1:]))
        total.index_add_(0, self.indices.flatten(), flat)
        return total[:-1].split(sizes)


def mapped_role(values, local_indices):
    """A separate sentinel is permitted per node, never a real numerical zero."""
    total = sum(len(x) for x in values)
    offset, mapped = 0, []
    for x, ids in zip(values, local_indices):
        mapped.append(torch.where(ids < len(x), ids+offset, total))
        offset += len(x)
    return Role(tuple(values), torch.stack(mapped).to(values[0].device))


def _dependencies(value, leaves):
    # The inner graph must have no multi-output custom Function with a union of
    # unrelated dependencies. Single-output ATen and Attention maps satisfy this.
    ids = {id(x): i for i, x in enumerate(leaves)}
    if value.is_leaf:
        return {ids[id(value)]} if id(value) in ids else set()
    pending, seen, result = [value.grad_fn], set(), set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if variable is not None and id(variable) in ids:
            result.add(ids[id(variable)])
        pending.extend(edge for edge, _ in node.next_functions)
    return result


@dataclass
class Kernel:
    roles: dict
    output_indices: tuple  # one index into flattened B×Q per node
    evaluate: object
    record_grad: bool


class _Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, *originals):
        ctx.set_materialize_grads(False)
        roles, names = tuple(kernel.roles.values()), tuple(kernel.roles)
        leaves = [r.pack().detach().requires_grad_(kernel.record_grad and any(x.requires_grad for x in r.values))
                  for r in roles]
        with torch.enable_grad() if kernel.record_grad else torch.no_grad():
            fields = kernel.evaluate(dict(zip(names, leaves)))
        ctx.roles, ctx.names, ctx.field_names = roles, names, tuple(fields)
        if ctx.field_names != tuple(kernel.evaluate.fields):
            raise ValueError("tensor kernel field order differs from its declared schema")
        ctx.kernel, ctx.leaves, ctx.fields = kernel, leaves, tuple(fields.values())
        span = ctx.fields[0].shape[1]*ctx.fields[0].shape[2]
        ctx.publish = tuple(index+g*span for g, index in enumerate(kernel.output_indices))
        from ..autograd_census import register_private_graph
        register_private_graph("tensor_kernel", ctx.fields)
        ctx.deps = [_dependencies(y, leaves) for y in ctx.fields]
        ctx.save_for_backward(*originals)
        outputs, disconnected = [], []
        for node, index in enumerate(kernel.output_indices):
            for f, y in enumerate(ctx.fields):
                value = y[node].flatten(0, 1).index_select(0, index)
                outputs.append(value)
                if not any(roles[r].values[node].requires_grad for r in ctx.deps[f]):
                    disconnected.append(value)
        ctx.mark_non_differentiable(*disconnected)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("tensor kernels declare first-order VJP only")
        ctx.saved_tensors  # Check versions of every borrowed original.
        count, width = len(ctx.kernel.output_indices), len(ctx.fields)
        outputs, cotangents = [], []
        connected = [[False]*count for _ in ctx.roles]
        for f, output in enumerate(ctx.fields):
            used = [g for g in range(count) if bars[g*width+f] is not None]
            if not used or not output.requires_grad:
                continue
            bar = output.new_zeros(output.shape)  # contiguous even for transposed formula fields
            indices = torch.cat([ctx.publish[g] for g in used])
            values = torch.cat([bars[g*width+f] for g in used])
            bar.flatten(0, 2).index_copy_(0, indices, values)
            for g in used:
                for r in ctx.deps[f]:
                    connected[r][g] = True
            outputs.append(output); cotangents.append(bar)
        live = [i for i, leaf in enumerate(ctx.leaves) if leaf.requires_grad]
        gradient = {}
        if outputs and live:
            with torch.enable_grad():
                gradient = dict(zip(live, torch.autograd.grad(outputs, [ctx.leaves[i] for i in live],
                    cotangents, allow_unused=True, retain_graph=True)))
        result = []
        for r, role in enumerate(ctx.roles):
            bar = gradient.get(r)
            if role.shared:
                live_nodes = [g for g in range(count) if connected[r][g] and role.values[g].requires_grad]
                first = live_nodes[0] if live_nodes else -1
                result.extend(bar if g == first else None for g in range(count))
                continue
            pieces = role.transpose(bar) if bar is not None else [None]*count
            result.extend(piece if connected[r][g] and role.values[g].requires_grad else None
                          for g, piece in enumerate(pieces))
        return None, *result


def execute(roles, output_indices, evaluate):
    """Fields have [node,B,Q,*field]. All node math stays inside this block.

    Nodes in one call have identical structural dependencies. In particular,
    query-memory presence branches must be partitioned before this call.
    Node-local missing bars remain None; connected numerical zero stays zero.
    """
    for role in roles.values():
        if len(role.values) != len(output_indices):
            raise ValueError("tensor role node domain mismatch")
        if role.shared and (role.indices is not None or any(x is not role.values[0] for x in role.values)):
            raise ValueError("shared tensor role requires identical original roots")
    if hasattr(evaluate, "shared"):
        evaluate.shared = {name for name, role in roles.items() if role.shared}
    kernel = Kernel(roles, output_indices, evaluate, torch.is_grad_enabled())
    outputs = _Kernel.apply(kernel, *(x for r in roles.values() for x in r.values))
    count = len(output_indices)
    width = len(outputs)//count
    # Evaluation declares stable field ordering without requiring a second run.
    names = tuple(evaluate.fields)
    if width != len(names):
        raise RuntimeError("tensor kernel field schema mismatch")
    return [dict(zip(names, outputs[i*width:(i+1)*width])) for i in range(count)]
