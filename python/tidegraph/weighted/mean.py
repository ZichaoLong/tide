"""Architecture §2: weighted mean with its centered, exact first-order VJP.

The forward quotient is unchanged. The mass derivative contracts z_i-u before
division, avoiding subtraction of large independently rounded quotient terms.
Zero-total branch ownership remains with the Aggregate program.
"""
import torch


class _Mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mass, values):
        ctx.set_materialize_grads(False)
        total = mass.sum(-1)
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        mean = (mass[..., None]*values).sum(-2)/denominator[..., None]
        ctx.save_for_backward(mass, values, total, mean)
        return mean

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted mean supports first-order VJP only")
        if bar is None:
            return None, None
        mass, values, total, mean = ctx.saved_tensors
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        dm = ((values-mean.unsqueeze(-2))*bar.unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1)
        # The mathematical singleton mean is independent of its positive mass.
        # This is a structural identity, not a gradient-value threshold.
        if mass.shape[-1] == 1:
            dm = torch.zeros_like(mass)
        dv = (mass/denominator.unsqueeze(-1)).unsqueeze(-1)*bar.unsqueeze(-2)
        return dm, dv


weighted_mean = _Mean.apply
