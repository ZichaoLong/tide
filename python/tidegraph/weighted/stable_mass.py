"""Scaled Aggregate and fused mass-weighted KV read for positive-floor-v1.

These operations preserve the declared real formulas at stored inputs. A
nonrepresentable public adjoint is rejected, never clipped or silently dropped.
"""
import torch
from .support_policy import signed_exp_product


def finite(*values):
    if not all(torch.isfinite(value).all() for value in values):
        raise ArithmeticError("positive-floor-v1 weighted reduction exceeds requested dtype range")


class _ScaledMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mass, values):
        ctx.set_materialize_grads(False)
        peak = mass.max(-1).values
        scaled = mass/torch.where(peak > 0, peak, torch.ones_like(peak)).unsqueeze(-1)
        total = scaled.sum(-1)
        probabilities = scaled/torch.where(total > 0, total, torch.ones_like(total)).unsqueeze(-1)
        mean = (probabilities.unsqueeze(-1)*values).sum(-2)
        finite(mean, mass.sum(-1))
        ctx.save_for_backward(mass, values, probabilities, mean)
        return mean

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("positive-floor-v1 supports first-order VJP only")
        if bar is None:
            return None, None
        mass, values, probabilities, mean = ctx.saved_tensors
        total = mass.sum(-1)
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        dm = ((values-mean.unsqueeze(-2))*bar.unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1)
        if mass.shape[-1] == 1:
            dm = torch.zeros_like(mass)
        dv = probabilities.unsqueeze(-1)*bar.unsqueeze(-2)
        finite(dm, dv)
        return dm, dv


class _MassAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, values, weights, total):
        ctx.set_materialize_grads(False)
        log_weights = weights.log()
        scores = logits+log_weights
        probabilities = scores.softmax(-1)
        mean = (probabilities.unsqueeze(-1)*values).sum(-2)
        factor = total.clamp(max=1)
        result = factor.unsqueeze(-1)*mean
        finite(scores, result)
        ctx.save_for_backward(logits, values, weights, total, probabilities, mean)
        return result

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("positive-floor-v1 supports first-order VJP only")
        if bar is None:
            return (None,)*4
        logits, values, weights, total, probabilities, mean = ctx.saved_tensors
        factor = total.clamp(max=1)
        delta = values-values[..., :1, :]
        centered = delta-(probabilities.unsqueeze(-1)*delta).sum(-2, keepdim=True)
        root = (centered*bar.unsqueeze(-2)).sum(-1)
        # Fuse factor*p/weight in log space: the intermediate log derivative
        # can overflow, or its upstream cotangent underflow, separately.
        log_scale = factor.log().unsqueeze(-1)+logits-torch.logsumexp(logits+weights.log(), -1, keepdim=True)
        dw = signed_exp_product(log_scale, root)
        log_mass = log_scale+weights.log()
        dl = signed_exp_product(log_mass, root)
        dv = signed_exp_product(log_mass.unsqueeze(-1), bar.unsqueeze(-2))
        dt = torch.where(total <= 1, (mean*bar).sum(-1), torch.zeros_like(total))
        finite(dw, dl, dv, dt)
        return dl, dv, dw, dt


scaled_mean = _ScaledMean.apply
mass_attention = _MassAttention.apply
