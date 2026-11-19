"""Direct bank reads for owner-grouped FP32 projections.

CPU executes a literal segmented-MM reference. NPU uses native grouped matmul
for forward and both adjoints; it never indexes or makes contiguous a weight
bank. Only activations and small owner metadata are reordered. Other devices
and autocast are explicitly unsupported by this optional execution candidate.
"""
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


def check_grouped_policy(x, weight):
    if x.device.type not in {"cpu", "npu"}:
        raise ValueError("grouped weight access supports CPU reference and NPU only")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("grouped weight access requires explicit FP32; autocast is unsupported")
    allowed = {torch.float32, torch.float64} if x.device.type == "cpu" else {torch.float32}
    if x.dtype not in allowed or weight.dtype != x.dtype:
        raise ValueError("grouped weight access requires matching FP32 tensors (CPU FP64 diagnostic allowed)")


def _validate(x, weight, ends):
    check_grouped_policy(x, weight)
    if x.ndim != 2 or weight.ndim != 3 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("grouped projection expects x[rows,input], weight[groups,output,input]")
    if ends.ndim != 1 or ends.dtype != torch.int64 or ends.numel() != weight.shape[0]:
        raise ValueError("grouped projection requires one int64 segment end per weight group")
    if weight.device != x.device or ends.device != x.device:
        raise ValueError("grouped inputs, weight bank and segment ends must share a device")


def _native(x, weight, ends, group_type):
    # Backend admission/import belongs to the application. This module never
    # imports torch_npu, and an explicit request never falls back to gather.
    try:
        operation = torch.ops.npu.npu_grouped_matmul.default
    except AttributeError as error:
        raise RuntimeError("grouped NPU weight access requires registered torch_npu.npu_grouped_matmul") from error
    return operation([x], [weight], group_list=ends, split_item=3,
                     group_type=group_type, group_list_type=0)[0]


class _GroupedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, ends):
        _validate(x, weight, ends)
        ctx.save_for_backward(x, weight, ends)
        if not x.shape[0]:
            return x.new_empty((0, weight.shape[1]))
        if x.device.type == "npu":
            return _native(x, weight.transpose(-1, -2), ends, 0)
        output = x.new_empty((x.shape[0], weight.shape[1]))
        start = 0
        for group, stop in enumerate(ends.tolist()):
            if stop > start:
                torch.mm(x[start:stop], weight[group].T, out=output[start:stop])
            start = stop
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        x, weight, ends = ctx.saved_tensors
        need_x, need_weight, _ = ctx.needs_input_grad
        # This is an activation layout copy when necessary, never a bank copy.
        dy = incoming.contiguous()
        dx = dw = None
        if not x.shape[0]:
            return (torch.zeros_like(x) if need_x else None,
                    torch.zeros_like(weight) if need_weight else None, None)
        if x.device.type == "npu":
            if need_x:
                dx = _native(dy, weight, ends, 0)
            if need_weight:
                dw = _native(dy.T, x, ends, 2)
        else:
            dx = torch.empty_like(x) if need_x else None
            dw = torch.zeros_like(weight) if need_weight else None
            start = 0
            for group, stop in enumerate(ends.tolist()):
                if stop > start:
                    if need_x:
                        torch.mm(dy[start:stop], weight[group], out=dx[start:stop])
                    if need_weight:
                        torch.mm(dy[start:stop].T, x[start:stop], out=dw[group])
                start = stop
        return dx, dw, None


@dataclass(frozen=True, slots=True)
class GroupedPlan:
    """Per-call discrete plan; contains no parameters or cached activations."""
    order: torch.Tensor
    inverse: torch.Tensor
    ends: torch.Tensor
    start: int
    stop: int
    rows_per_owner: int

    def apply(self, x, weight):
        if x.ndim != 3 or x.shape[:2] != (self.order.numel(), self.rows_per_owner):
            raise ValueError("grouped owner plan does not match this activation layout")
        ordered = x.index_select(0, self.order).reshape(-1, x.shape[-1])
        # A host-metadata slice of the original bank, including absent groups;
        # actual dynamic owner IDs are never read on the host.
        bank = weight[self.start:self.stop]
        output = _GroupedProjection.apply(ordered, bank, self.ends)
        output = output.reshape(x.shape[0], x.shape[1], weight.shape[1])
        return output.index_select(0, self.inverse)


def make_grouped_plan(ids, groups, host_groups, possible_ids, rows_per_owner, bank_groups):
    """Sort owner blocks; merge repeated/shared bases into one segment.

    Device counts include zero-length groups in the statically bounded bank.
    Host possible_ids is only a trusted superset, never an asserted actual order.
    groups=None denotes private, node-owned adapter banks.
    """
    if ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError("grouped owners must be an int64 vector")
    possible = tuple(possible_ids)
    possible_groups = tuple(host_groups[owner] for owner in possible) if groups is not None else possible
    start = min(possible_groups) if possible_groups else 0
    stop = max(possible_groups) + 1 if possible_groups else bank_groups
    actual = (groups[ids] if groups is not None else ids) - start
    order = actual.argsort(stable=True)
    inverse = torch.empty_like(order).scatter(0, order, torch.arange(len(ids), device=ids.device))
    counts = ids.new_zeros(stop - start).scatter_add(0, actual, torch.full_like(actual, rows_per_owner))
    return GroupedPlan(order, inverse, counts.cumsum(0), start, stop, rows_per_owner)
