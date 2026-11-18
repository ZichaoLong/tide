"""Explicit mixed precision for the instance library.

BF16 means BF16 matrix arithmetic with FP32 parameters and optimizer state.
Callers keep floating inputs and parameters in FP32, put forward/loss in this
context, and run backward/AdamW outside it. FP64 is a separate CPU diagnostic.
"""
from contextlib import contextmanager
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def precision_policy(device, precision="fp32"):
    """Serializable requested policy; qualification still requires execution."""
    device = torch.device(device)
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if device.type not in {"cpu", "cuda", "npu"}:
        raise ValueError("precision policy supports CPU, CUDA and NPU")
    return {
        "name": precision,
        "device": str(device),
        "parameters": "float32",
        "input": "float32",
        "matrix_arithmetic": "bfloat16_autocast" if precision == "bf16" else "float32",
        "normalization_mass_allocation": "float32",
        "kv_keys_values": "bfloat16" if precision == "bf16" else "float32",
        "kv_summary_and_ssd_state": "float32",
        "logical_time_indices": "int64",
        "parameter_gradients": "float32",
        "master_weights": "parameters_are_fp32_masters; no_separate_copy",
        "adamw_moments": "float32",
        "attention": "native_by_shape_and_prior; math/tiled_accumulate_float32",
        "loss": "float32",
        "autocast_cache": False,
    }


@contextmanager
def precision_context(device, precision="fp32"):
    """Select arithmetic without changing parameter/storage ownership.

    Disabling the cast cache avoids stale casts when the same context spans
    optimizer updates. An explicit FP32 context also overrides outer autocast.
    No device plugin is imported here: resolve the backend before constructing
    the model. CPU FP64 diagnostics should call the ordinary model directly.
    """
    policy = precision_policy(device, precision)
    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                        enabled=policy["name"] == "bf16", cache_enabled=False):
        yield


class ArithmeticAudit(TorchDispatchMode):
    """Intrusive one-call dtype/operator witness, separate from timed samples.

    This records dispatch-visible tensor dtypes and device placement; it does
    not infer a hardware accumulation dtype or certify a fused implementation.
    """
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        if any(part in name for part in ("mm.", "bmm.", "addmm.", "attention", "convolution")):
            inputs = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
            outputs = [x for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor)]
            ins = tuple(sorted({f"{x.device.type}:{str(x.dtype).removeprefix('torch.')}" for x in inputs}))
            outs = tuple(sorted({f"{x.device.type}:{str(x.dtype).removeprefix('torch.')}" for x in outputs}))
            self.counts[(name, ins, outs)] += 1
        return result

    def report(self):
        return [{"operator": op, "inputs": list(inputs), "outputs": list(outputs), "calls": count}
                for (op, inputs, outputs), count in sorted(self.counts.items())]
