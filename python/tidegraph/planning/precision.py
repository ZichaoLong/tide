"""Explicit execution precision for the same seeded FP32 benchmark material.

This changes execution dtype, not the saved plan, parameter count or topology.
Promotion happens before either schedule and preserves actual input aliases.
"""
from dataclasses import replace
import torch


def promote(binding, inputs, dtype):
    if dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("benchmark precision must be float16, float32 or float64")
    binding.model.to(dtype=dtype)
    copies = {}
    for x in inputs:
        if id(x.value) not in copies:
            copies[id(x.value)] = x.value.to(dtype=dtype)
    return binding, [replace(x, value=copies[id(x.value)]) for x in inputs]


def tolerances(dtype):
    return (1e-10, 1e-8) if dtype == torch.float64 else (1e-3, 2e-2) if dtype == torch.float16 else (1e-6, 1e-5)


def validate(dtype, device, implementation):
    if dtype == "float64" and (device.type not in {"cpu", "cuda"} or implementation == "resident"):
        raise ValueError("FP64 benchmark requires CPU or single-card CUDA eager execution; no implicit substitution")
