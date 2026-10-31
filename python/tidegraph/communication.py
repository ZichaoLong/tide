"""Explicit eager transport policy, fixed for each experiment process.

Ordinary copies make no P2P claim. Blocking synchronizes both endpoints.
CPU bridge is an explicit, blocking accelerator -> CPU -> accelerator route.
Autograd edges save their chosen policy so backward threads cannot change it.
"""
import os
import threading
import torch

POLICIES = {"ordinary", "blocking", "cpu_bridge"}
_configured = None
_lock = threading.Lock()
_counts = dict(copies=0, logical_bytes=0, tensor_leg_bytes=0, endpoint_synchronizations=0, cpu_bridge_copies=0)


def policy():
    value = os.environ.get("TIDE_COMMUNICATION_POLICY", "ordinary")
    if value not in POLICIES:
        raise ValueError("TIDE_COMMUNICATION_POLICY: choose ordinary, blocking or cpu_bridge")
    return value


def configure(value):
    global _configured
    if value not in POLICIES:
        raise ValueError("execution.communication: choose ordinary, blocking or cpu_bridge")
    if _configured is not None and _configured != value:
        raise ValueError("communication policy is fixed per process; start a separate experiment process")
    _configured = value
    os.environ["TIDE_COMMUNICATION_POLICY"] = value


def statistics(*, reset=False):
    with _lock:
        result = dict(_counts, requested=policy(), effective=policy(),
                      scope="public Python placement/message transport only; excludes native/internal allocator copies")
        if reset:
            for key in _counts:
                _counts[key] = 0
    return result


def copy(value, device, dtype=None, *, selected=None):
    selected = policy() if selected is None else selected
    if selected not in POLICIES:
        raise ValueError("unknown explicit communication policy")
    device = torch.device(device)
    dtype = value.dtype if dtype is None else dtype
    if value.device == device:
        return value.to(dtype=dtype)
    bridge = selected == "cpu_bridge" and value.device.type != "cpu" and device.type != "cpu"
    endpoints = [d for d in (value.device, device) if d.type != "cpu"]
    fences = 0
    if selected != "ordinary":
        from .runtime import synchronize
        for endpoint in endpoints:
            synchronize(endpoint)
            fences += 1
    copied = value.to(device="cpu") if bridge else value
    copied = copied.to(device=device, dtype=dtype)
    if selected != "ordinary":
        for endpoint in endpoints:
            synchronize(endpoint)
            fences += 1
    with _lock:
        _counts["copies"] += 1
        _counts["logical_bytes"] += value.numel()*value.element_size()
        _counts["tensor_leg_bytes"] += value.numel()*value.element_size()*(2 if bridge else 1)
        _counts["endpoint_synchronizations"] += fences
        _counts["cpu_bridge_copies"] += int(bridge)
    return copied
