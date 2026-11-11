"""Allocation §6/S.2: column roots and an explicit, support-aware exact-map VJP.

R flattens any declared (sample, logical-time) layout. Candidate count N is real,
not padding. Solver iterations never participate in autograd. The explicit
pullback accepts structural row support separately from numeric zero adjoints.
"""
from dataclasses import dataclass, field
import torch
from .allocation_batch import solve, BatchAllocation, ZERO, CAPS, SLACK, LOWER, EQUALITY
from .regularizers import Regularizer
from . import support_policy as support


def row_scalar(value, scores, name):
    if isinstance(value, torch.Tensor):
        if (value.dtype, value.device) != (scores.dtype, scores.device):
            raise ValueError(f"allocation column {name} metadata mismatch")
    else:
        value = scores.new_tensor(value)
    if value.ndim == 0:
        return value.expand(scores.shape[0])
    if value.shape != scores.shape[:1]:
        raise ValueError(f"allocation column {name} must be scalar or R")
    return value


@dataclass(frozen=True)
class AllocationColumns:
    controls: torch.Tensor
    numeric: BatchAllocation
    inputs: tuple
    rule: Regularizer
    policy: str
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        tensors = (*self.inputs, self.numeric.controls, self.numeric.response_input, self.numeric.branch)
        object.__setattr__(self, "_versions", tuple(None if t is None or t.is_inference() else t._version for t in tensors))

    @property
    def backward(self):
        return self.rule.kind+"-exact-map-v1" if self.policy == "strict" else "ent-positive-floor-v1"

    def pullback(self, grad, rows=None):
        """First-order column adjoints; omitted rows are disconnected, not zero.

        None is reported per tensor column. An adapter exposing independent row
        roots must scatter only the declared rows, grouped by numeric.branch.
        """
        if torch.is_grad_enabled():
            raise ValueError("allocation columns support first-order exact-map VJP only")
        tensors = (*self.inputs, self.numeric.controls, self.numeric.response_input, self.numeric.branch)
        for t, version in zip(tensors, self._versions):
            if t is not None and (version is None or t._version != version):
                raise RuntimeError("allocation column input changed or has no retained version counter")
        return pullback(self.inputs, self.numeric, self.rule, self.policy, grad, rows)


def pullback(inputs, numeric, rule, policy, grad, rows=None):
    scores, caps, lower, upper, temperature, shift = inputs
    if grad.shape != scores.shape or (grad.dtype, grad.device) != (scores.dtype, scores.device):
        raise ValueError("allocation column adjoint metadata mismatch")
    branch = numeric.branch
    if rows is None:
        rows = torch.ones_like(branch, dtype=torch.bool)
    if rows.shape != branch.shape or rows.dtype != torch.bool or rows.device != branch.device:
        raise ValueError("allocation column adjoint support must be bool R on the same device")
    result = [None]*6
    capped = rows & (branch == CAPS)
    normal = rows & (branch != ZERO) & (branch != CAPS)
    if capped.any():
        result[1] = torch.where(capped[:, None], grad, 0)
    if normal.any():
        ids = normal.nonzero().flatten()
        c, tau, bar = numeric.controls[ids], temperature[ids, None], grad[ids]
        x, mu, b = numeric.response_input[ids], caps[ids], branch[ids]
        if policy != "strict":
            ds, dc, mean, dt = support.shannon_vjp(x, mu, tau, bar, b != SLACK)
        else:
            a, e, dtau = rule.derivatives(x, mu, tau, c)
            total = a.sum(1)
            constrained = (b != SLACK) & (total > 0)
            anchor = bar.gather(1, a.argmax(1)[:, None]).squeeze(1)
            relative = bar-anchor[:, None]
            offset = (a*relative).sum(1)/torch.where(total > 0, total, 1)
            mean = torch.where(constrained, anchor+offset, 0)
            centered = torch.where(constrained[:, None], relative-offset[:, None], bar)
            ds, dc, dt = a*centered, e*centered, (dtau*centered).sum(1)
        result[0] = torch.zeros_like(scores).index_copy_(0, ids, ds)
        cap_bar = torch.zeros_like(caps).index_copy_(0, ids, dc)
        result[1] = cap_bar if result[1] is None else result[1]+cap_bar
        for index, mask in ((2, (b == LOWER) | (b == EQUALITY)), (3, (b != LOWER) & (b != EQUALITY))):
            if mask.any():
                result[index] = torch.zeros_like(lower).index_copy_(0, ids[mask], mean[mask])
        result[4] = torch.zeros_like(temperature).index_copy_(0, ids, dt)
        if shift is not None:
            dshift = torch.where(b != SLACK, 0, -ds.sum(1))
            result[5] = torch.zeros_like(shift).index_copy_(0, ids, dshift)
    return tuple(result)


class _Columns(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rule, policy, numeric, *inputs):
        ctx.set_materialize_grads(False)
        ctx.rule, ctx.policy = rule, policy
        ctx.shifted = inputs[-1] is not None
        ctx.save_for_backward(*inputs, numeric.controls, numeric.multiplier, numeric.branch,
                              numeric.response_input, numeric.free, numeric.residual)
        out = numeric.controls.view_as(numeric.controls)
        # Zero and all-caps maps have a narrower dependency than the general KKT
        # map. Preserve a genuinely constant result, including inference mode.
        branches = set(numeric.branch.detach().cpu().tolist())
        used = {1} if CAPS in branches else set()
        for b in branches-{ZERO, CAPS}:
            used.update((0, 1, 2 if b in {LOWER, EQUALITY} else 3, 4))
            if ctx.shifted:
                used.add(5)
        if not any(inputs[i].requires_grad for i in used):
            ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise ValueError("allocation columns support first-order exact-map VJP only")
        saved = ctx.saved_tensors
        if grad is None:
            return (None,)*9
        numeric = BatchAllocation(*saved[6:])
        bars = pullback(saved[:6], numeric, ctx.rule, ctx.policy, grad)
        return None, None, None, *bars


def allocate_columns(scores, caps, lower, upper, temperature=1., *, regularizer="quad",
                     ent_alpha=None, score_shift=None, support_policy="strict"):
    """Complete R×N score/cap tensors, scalar or R bounds/temperature/shift.

    Returned numerical diagnostics are detached; controls use the declared exact
    solution VJP. Shared scalar inputs accumulate through broadcast once.
    """
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError("allocation columns require nonempty R,N")
    rule = Regularizer(regularizer, ent_alpha)
    lo, hi, tau = (row_scalar(v, scores, n) for v, n in
                   ((lower, "lower"), (upper, "upper"), (temperature, "temperature")))
    shift = None if score_shift is None else row_scalar(score_shift, scores, "shift")
    inputs = scores, caps, lo, hi, tau, shift
    numeric = solve(scores, caps, lo, hi, tau, rule, shift, support_policy)
    controls = _Columns.apply(rule, support_policy, numeric, *inputs)
    return AllocationColumns(controls, numeric, inputs, rule, support_policy)
