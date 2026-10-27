"""Architecture §1: epsilon RMS normalization and its normalized first-order VJP.

The forward x/sqrt(epsilon+mean(x²)) is unchanged. Contract the adjoint with
the normalized vector before dividing, instead of forming two inverse powers.
"""
import torch


class _Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, epsilon):
        ctx.set_materialize_grads(False)
        operand = value.float() if value.dtype == torch.float16 else value
        denominator = torch.sqrt(epsilon+operand.square().mean(-1, keepdim=True))
        normalized = operand/denominator
        ctx.save_for_backward(value, normalized, denominator)
        return normalized.to(value.dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted normalization supports first-order VJP only")
        if bar is None:
            return None, None
        value, normalized, denominator = ctx.saved_tensors
        if value.dtype == torch.float16:
            bar, denominator = bar.float(), denominator.float()
            normalized = value.float()/denominator
        gradient = (bar-normalized*(bar*normalized).mean(-1, keepdim=True))/denominator
        return gradient.to(value.dtype), None


norm = _Normalize.apply


class _HalfScaledRMS(torch.autograd.Function):
    """Half operands/output with FP32 normalization and first-order contraction."""
    @staticmethod
    def forward(ctx, value, scale, epsilon):
        ctx.set_materialize_grads(False)
        denominator = torch.sqrt(epsilon+value.float().square().mean(-1, keepdim=True))
        ctx.save_for_backward(value, scale, denominator)
        return (scale.float()*value.float()/denominator).to(value.dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted scaled RMS supports first-order VJP only")
        if bar is None:
            return None, None, None
        value, scale, denominator = ctx.saved_tensors
        n = value.float()/denominator.float()
        weighted = bar.float()*scale.float()
        dx = (weighted-n*(weighted*n).mean(-1, keepdim=True))/denominator.float()
        ds = (bar.float()*n).sum_to_size(scale.shape)
        return dx.to(value.dtype), ds.to(scale.dtype), None


def scaled_rms(value, scale, epsilon):
    if value.dtype == torch.float16:
        return _HalfScaledRMS.apply(value, scale, epsilon)
    return scale*value/torch.sqrt(epsilon+value.square().mean(-1, keepdim=True))


class _HalfL2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, epsilon):
        ctx.set_materialize_grads(False)
        denominator = torch.sqrt(epsilon+value.float().square().sum(-1, keepdim=True))
        ctx.save_for_backward(value, denominator)
        return (value.float()/denominator).to(value.dtype)

    @staticmethod
    def backward(ctx, bar):
        if torch.is_grad_enabled():
            raise ValueError("weighted epsilon L2 supports first-order VJP only")
        if bar is None:
            return None, None
        value, denominator = ctx.saved_tensors
        denominator = denominator.float()
        n, bar = value.float()/denominator, bar.float()
        gradient = (bar-n*(bar*n).sum(-1, keepdim=True))/denominator
        return gradient.to(value.dtype), None


def epsilon_l2(value, epsilon):
    if value.dtype == torch.float16:
        return _HalfL2.apply(value, epsilon)
    return value/torch.sqrt(epsilon+value.square().sum(-1, keepdim=True))
