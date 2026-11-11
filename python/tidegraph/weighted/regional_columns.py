"""Allocation §2–6: tensor descriptors → scores/evidence → controls/support.

R is the flattened sample/time group, N the complete candidate set. Layout,
int64 coordinates and state ownership belong to the caller; no event objects
or history reconstruction enter this mathematical boundary.
"""
from dataclasses import dataclass
import math
import torch
from .allocation_columns import allocate_columns
from .allocation_batch import ZERO, CAPS, LOWER, EQUALITY, NAMES
from .config import TOTAL, ATTENTION, HARD
from .sending_batch import attention, proportional, sorted_send


@dataclass(frozen=True)
class RegionColumns:
    controls: torch.Tensor
    allocation: object | None
    mode: str

    @property
    def active(self):
        return self.controls.detach() > 0

    def dependencies(self):
        if self.allocation is None:
            return [self.mode]*len(self.controls)
        return [NAMES[b] for b in self.allocation.numeric.branch.detach().cpu().tolist()]

    def differentiable_rows(self):
        if self.allocation is None:
            return [self.controls.requires_grad]*len(self.controls)
        a = self.allocation
        result = []
        for b in a.numeric.branch.detach().cpu().tolist():
            used = ({1} if b == CAPS else set() if b == ZERO else
                    {0, 1, 2 if b in {LOWER, EQUALITY} else 3, 4, 5})
            result.append(any(a.inputs[i] is not None and a.inputs[i].requires_grad for i in used))
        return result

    def vjp_sources(self, grad, rows):
        """Sources/cotangents for a row-root adapter, preserving branch support."""
        if self.allocation is None:
            return [(self.controls, grad)] if self.controls.requires_grad else []
        return [(x, bar) for x, bar in zip(self.allocation.inputs, self.allocation.pullback(grad, rows))
                if x is not None and x.requires_grad and bar is not None]


def region_columns(program, fields, nodes=None, *, selection=None):
    """Public column kernel; fields are R×N scalars or R×N×D descriptors.

    Different descriptor widths and candidate counts form separate real groups.
    Stable-sorted sending requires int64 R×N node IDs; other functions ignore IDs.
    Selection placement moves whole columns and returns its actual device.
    """
    c, mode = program.config, program.config.selection_mode
    recv, caps, scores = (fields[n] for n in ("w", "mu", "score"))
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("region columns require nonempty R,N")
    for name, value in fields.items():
        if value.shape[:2] != scores.shape or (value.dtype, value.device) != (scores.dtype, scores.device):
            raise ValueError(f"region column {name} metadata mismatch")
    if recv.ndim != 2 or caps.ndim != 2:
        raise ValueError("region w/mu columns must be scalar fields")
    if mode == "regional_total_zero" or mode in HARD:
        if nodes is None or nodes.shape != scores.shape or nodes.dtype != torch.int64 or nodes.device != scores.device:
            raise ValueError("stable sending requires int64 R,N node IDs on the descriptor device")
    if c.region_score == "att":
        weights = fields["w" if c.query_weight == "received" else "mu"]
        x = (weights[..., None]*fields["x"]).sum(1)/weights.sum(1).clamp_min(1)[:, None]
        query = program.query(x)
        scores = c.alpha_loc*scores+c.alpha_sel*(fields["key"]*query[:, None]).sum(-1)/math.sqrt(c.selection_width)
    received, mass = recv.sum(1), caps.sum(1)
    context = None
    if c.region_control_mode == "context":
        z = (recv[..., None]*fields["evidence"]).sum(1)/received.clamp_min(1)[:, None]
        context = program.head(torch.cat((z, received.log1p()[:, None], mass.log1p()[:, None]), 1))
    shift, low, high = None, None, None
    if mode in TOTAL:
        low, high = received.clamp_max(1), mass.clamp_max(c.k)
        rho = context.sigmoid() if context is not None else received.new_full(received.shape, c.total_fraction)
        low = high = low+rho*(high-low)
    elif mode not in ATTENTION and mode != "proportional":
        shift = context if context is not None else received.new_full(received.shape, c.eta)
        low = (received if c.regional else mass).clamp_max(1)
        high = received.new_full(received.shape, c.k)
    selection = selection if selection is not None else getattr(program, "_selection_device", None)
    if selection is not None:
        from .placement_tensors import move
        recv, caps, scores, nodes, received, low, high, shift = (
            move(t, selection) for t in (recv, caps, scores, nodes, received, low, high, shift))
    allocation = None
    if mode == "regional_total_zero":
        values = sorted_send(scores, caps, nodes, target=low)
    elif mode in ATTENTION:
        values = attention(scores, caps, recv, mode, c.allocation_base,
                           received.new_full(received.shape, c.allocation_temperature))
    elif mode == "proportional":
        values = proportional(caps, c.k)
    elif mode in HARD:
        values = sorted_send(scores-shift[:, None], caps, nodes, k=c.k)
    else:
        allocation = allocate_columns(scores, caps, low, high, c.allocation_temperature,
            regularizer=c.regularizer, ent_alpha=c.ent_alpha, score_shift=shift, support_policy=c.support_policy)
        values = allocation.controls
    return RegionColumns(values, allocation, mode)
