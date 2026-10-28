"""CPU arithmetic interventions for diagnosis, never a production precision policy.

Each selected ATen result is rounded back to FP32. This is different from a
coherent FP64 graph, and from differentiating a replacement Python expression.
Both independent schedules receive the same intervention. Optimizers stay out.
"""
from collections import Counter
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


LINEAR = {"aten.mm.default", "aten.bmm.default", "aten.mv.default", "aten.dot.default", "aten.addmm.default"}
REDUCE = {"aten.sum.default", "aten.sum.dim_IntList", "aten.mean.default", "aten.mean.dim"}
NONLINEAR = {"aten.silu.default", "aten.silu_backward.default", "aten.sigmoid.default",
             "aten.sigmoid_backward.default", "aten.exp.default", "aten.log.default", "aten.sqrt.default"}
VARIANTS = {
    "none": (set(), False, False),
    "linear64": (LINEAR, False, False),
    "reduce64": (REDUCE, False, False),
    "linear_reduce64": (LINEAR | REDUCE, False, False),
    "norm64": (set(), True, False),
    "linear_reduce_norm64": (LINEAR | REDUCE, True, False),
    "nonlinear64": (LINEAR | REDUCE | NONLINEAR, True, False),
    "backward_linear_reduce64": (LINEAR | REDUCE, False, True),
}


class _Arithmetic(TorchDispatchMode):
    def __init__(self, names):
        super().__init__()
        self.names, self.phase, self.counts = names, "forward", Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        x = args[0] if args else None
        if (str(func) not in self.names or not isinstance(x, torch.Tensor) or x.dtype != torch.float32
                or kwargs.get("dtype") not in (None, torch.float32)):
            return func(*args, **kwargs)
        if x.device.type != "cpu":
            raise ValueError("arithmetic diagnostics require CPU")
        def cast(t):
            return t.double() if isinstance(t, torch.Tensor) and t.dtype == torch.float32 else t
        widened = dict(kwargs)
        if "dtype" in widened:
            widened["dtype"] = torch.float64
        result = func(*tree_map(cast, args), **tree_map(cast, widened))
        self.counts[self.phase+":"+str(func)] += 1
        return tree_map(lambda t: t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.float64 else t, result)


def _normal_forward(ctx, value, epsilon):
    ctx.set_materialize_grads(False)
    if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64):
        raise ValueError("normalization diagnostic requires CPU FP32/FP64")
    operand = value.double()
    denominator = torch.sqrt(epsilon+operand.square().mean(-1, keepdim=True))
    normalized = operand/denominator
    ctx.save_for_backward(value, normalized, denominator)
    return normalized.to(value.dtype)


def _normal_backward(ctx, bar):
    if torch.is_grad_enabled():
        raise ValueError("normalization diagnostic supports first-order VJP only")
    if bar is None:
        return None, None
    value, normalized, denominator = ctx.saved_tensors
    b = bar.double()
    dx = (b-normalized*(b*normalized).mean(-1, keepdim=True))/denominator
    return dx.to(value.dtype), None


class ArithmeticProbe:
    """Single-process, single-thread diagnostic contexts; restore all patches.

    Use the matching context around both forward and backward. No C++/device
    behavior is inferred from Python dispatch interception. Counts are actual
    intercepted calls, not total FLOPs or profiling/throughput measurements.
    """
    def __init__(self, variant, *, dtype, implementation):
        if variant not in VARIANTS:
            raise ValueError("unknown arithmetic diagnostic")
        if variant != "none" and (dtype != "float32" or implementation != "python"):
            raise ValueError("arithmetic intervention requires Python CPU FP32")
        self.variant = variant
        names, self.norm64, self.backward_only = VARIANTS[variant]
        self.mode = _Arithmetic(names)

    @contextmanager
    def phase(self, name):
        if name not in ("forward", "backward"):
            raise ValueError("arithmetic intervention cannot cover optimizer or checkpoint")
        if self.variant == "none" or (self.backward_only and name == "forward"):
            yield
            return
        from tidegraph.weighted.normalization import _Normalize
        self.mode.phase = name
        norm_patch = patch.multiple(_Normalize, forward=staticmethod(_normal_forward),
                                    backward=staticmethod(_normal_backward)) if self.norm64 else nullcontext()
        with norm_patch, self.mode:
            yield

    def record(self):
        return dict(variant=self.variant, widened_ops=sorted(self.mode.names),
                    normalization_fp64=self.norm64, backward_only=self.backward_only,
                    operation_counts=dict(self.mode.counts),
                    result_boundary="selected operator result cast to FP32; norm internal FP64 with FP32 result/VJP",
                    optimizer="outside intervention, unchanged",
                    scope="diagnostic only; not public implementation, native/device qualification or throughput")
