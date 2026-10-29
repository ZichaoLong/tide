"""CPU diagnostic: unchanged half forward, FP64 analytic normalization VJP.

This tests the local cancellation boundary before half gradients are accumulated.
It is not a replacement precision policy or evidence for accelerator FP64.
"""
from contextlib import ExitStack
from unittest.mock import patch
import torch


class HalfNormalizationProbe:
    def __init__(self, enabled=False):
        self.enabled, self.counts, self.stack = enabled, {}, ExitStack()

    def __enter__(self):
        if not self.enabled:
            return self
        from tidegraph.weighted.normalization import _Normalize, _FP32L2, _FP32ScaledRMS
        for cls, kind in ((_Normalize, "rms"), (_FP32L2, "l2"), (_FP32ScaledRMS, "scaled_rms")):
            forward, backward = cls.forward, cls.backward
            def make(old_forward, old_backward, mode):
                def new_forward(ctx, *args):
                    value = old_forward(ctx, *args)
                    ctx.probe_epsilon = args[-1]
                    return value
                def new_backward(ctx, bar):
                    value = ctx.saved_tensors[0]
                    if value.dtype != torch.float16 or bar is None:
                        return old_backward(ctx, bar)
                    if torch.is_grad_enabled():
                        raise ValueError("half diagnostic supports first-order VJP only")
                    if value.device.type != "cpu":
                        raise ValueError("half normalization diagnostic requires CPU")
                    self.counts[mode] = self.counts.get(mode, 0)+1
                    x, b = value.double(), bar.double()
                    norm = x.square().sum(-1, keepdim=True) if mode == "l2" else x.square().mean(-1, keepdim=True)
                    denominator = (norm+ctx.probe_epsilon).sqrt()
                    n = x/denominator
                    scale = ctx.saved_tensors[1] if mode == "scaled_rms" else None
                    weighted = b if scale is None else b*scale.double()
                    product = weighted*n
                    dot = product.sum(-1, keepdim=True) if mode == "l2" else product.mean(-1, keepdim=True)
                    dx = ((weighted-n*dot)/denominator).to(value.dtype)
                    if scale is None:
                        return dx, None
                    return dx, (b*n).sum_to_size(scale.shape).to(scale.dtype), None
                return new_forward, new_backward
            f, b = make(forward, backward, kind)
            self.stack.enter_context(patch.multiple(cls, forward=staticmethod(f), backward=staticmethod(b)))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def record(self):
        return dict(policy="normalization-vjp-fp64" if self.enabled else "public",
                    forward="unchanged", output_and_gradient_dtype="unchanged",
                    counts=self.counts, scope="diagnostic only; CPU half, local VJP before half accumulation")
