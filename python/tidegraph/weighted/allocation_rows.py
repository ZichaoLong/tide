"""Independent row roots for the exact allocation map, never solver derivatives.

Allocation §6/S.2: selected KKT branch, capped/zero boundaries, explicit common
shift derivative. Ragged callers bucket only by the real candidate count.
"""
from collections import defaultdict
import torch
from .allocation import Allocation, _validate, _scalar
from .allocation_batch import solve, ZERO, CAPS, SLACK, LOWER, EQUALITY, NAMES
from .regularizers import Regularizer


class _Rows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rule, shifted, numeric, *inputs):
        ctx.set_materialize_grads(False)
        ctx.rule, ctx.shifted, ctx.width = rule, shifted, 6 if shifted else 5
        ctx.count = len(inputs)
        ctx.save_for_backward(*inputs, numeric.controls, numeric.multiplier, numeric.branch, numeric.response_input)
        outputs = tuple(c.clone() for c in numeric.controls.unbind())
        constants = []
        for i, c in enumerate(outputs):
            row = inputs[i*ctx.width:(i+1)*ctx.width]
            branch = int(numeric.branch[i])
            if branch == ZERO:
                differentiable = False
            elif branch == CAPS:
                differentiable = row[1].requires_grad
            else:
                target = 2 if branch in {LOWER, EQUALITY} else 3
                differentiable = any(row[j].requires_grad for j in (0, 1, target, 4)) or shifted and row[5].requires_grad
            if not differentiable:
                constants.append(c)
        ctx.mark_non_differentiable(*constants)
        return outputs

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("batched allocation supports first-order exact-map VJP only")
        saved = ctx.saved_tensors
        inputs, (controls, multiplier, branch, response) = saved[:ctx.count], saved[ctx.count:]
        result, normal = [None]*ctx.count, []
        for i, grad in enumerate(bars):
            if grad is None:
                continue
            if int(branch[i]) == CAPS:
                result[i*ctx.width+1] = grad
            elif int(branch[i]) != ZERO:
                normal.append(i)
        if normal:
            fields = [torch.stack([inputs[i*ctx.width+j] for i in normal]) for j in range(ctx.width)]
            scores, caps, _, _, temperature = fields[:5]
            c, tau, grad = controls[normal], temperature[:, None], torch.stack([bars[i] for i in normal])
            if ctx.rule.kind == "quad":
                x = scores-multiplier[normal, None]
                if ctx.shifted:
                    x = scores-fields[5][:, None]-multiplier[normal, None]
            else:
                x = response[normal]
            a, e, dtau = ctx.rule.derivatives(x, caps, tau, c)
            total = a.sum(1)
            constrained = (branch[normal] != SLACK) & (total > 0)
            anchor = grad.gather(1, a.argmax(1)[:, None]).squeeze(1)
            relative = grad-anchor[:, None]
            offset = (a*relative).sum(1)/torch.where(total > 0, total, torch.ones_like(total))
            mean = torch.where(constrained, anchor+offset, torch.zeros_like(total))
            centered = torch.where(constrained[:, None], relative-offset[:, None], grad)
            scores_bar, caps_bar = a*centered, e*centered
            tau_bar = (dtau*centered).sum(1)
            shift_bar = torch.where(branch[normal] != SLACK, torch.zeros_like(total), -scores_bar.sum(1))
            for k, i in enumerate(normal):
                base = i*ctx.width
                target = 2 if int(branch[i]) in {LOWER, EQUALITY} else 3
                result[base], result[base+1], result[base+target], result[base+4] = scores_bar[k], caps_bar[k], mean[k], tau_bar[k]
                if ctx.shifted:
                    result[base+5] = shift_bar[k]
        return None, None, None, *result


def allocate_batch(scores, caps, lower, upper, temperature=1., *, regularizer="quad", ent_alpha=None, score_shift=None):
    """Lists of complete score/cap vectors; bounds may be scalar or row lists.

    A scalar Tensor bound/temperature is a shared root, whose uses accumulate.
    Results retain each row's complete controls, branch, multiplier and residual.
    """
    if len(scores) != len(caps):
        raise ValueError("allocation batch row domain mismatch")
    n = len(scores)
    def rows(value):
        value = list(value) if isinstance(value, (tuple, list)) else [value]*n
        if len(value) != n:
            raise ValueError("allocation batch row domain mismatch")
        return value
    lo, hi, tau, shifts = map(rows, (lower, upper, temperature, score_shift))
    inputs, groups, result = [], defaultdict(list), [None]*n
    rule = Regularizer(regularizer, ent_alpha)
    for i, (t, c) in enumerate(zip(scores, caps)):
        l, h, temp = _validate(t, c, lo[i], hi[i], tau[i])
        shift = None if shifts[i] is None else _scalar(shifts[i], t, "score_shift")
        inputs.append((t, c, l, h, temp)+(tuple() if shift is None else (shift,)))
        groups[(len(t), t.dtype, t.device, shift is not None)].append(i)
    for (_, _, _, shifted), ids in groups.items():
        fields = [torch.stack([inputs[i][j] for i in ids]) for j in range(6 if shifted else 5)]
        numeric = solve(*fields[:5], rule, fields[5] if shifted else None)
        flat = [t for i in ids for t in inputs[i]]
        controls = _Rows.apply(rule, shifted, numeric, *flat)
        identity = regularizer+"-exact-map-v1"
        for k, i in enumerate(ids):
            result[i] = Allocation(controls[k], numeric.multiplier[k], NAMES[int(numeric.branch[k])],
                                   numeric.free[k], numeric.residual[k], identity)
    return result
