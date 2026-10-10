"""Explicit accelerator targets shared by resident and complete-consumer gates."""
import os
import re
import pytest
import torch


def target_device(variable="TIDE_RESIDENT_DEVICE", *, cards=1, resident=True):
    spec = os.environ.get(variable)
    if spec is None:
        pytest.skip("optional accelerator target not requested: " + variable)
    assert re.fullmatch(r"(?:npu|cuda):[0-9]+", spec), "explicit indexed CUDA/NPU target required"
    from tidegraph.runtime import resolve_device
    device, _ = resolve_device(spec)  # An explicitly absent device must fail.
    api = device_api(device)
    assert api.is_available() and api.device_count() >= device.index + cards, "requested accelerator owners unavailable"
    if resident:
        assert os.environ.get("TIDE_RESIDENT_LIBRARY"), "explicit target requires its backend build"
    return str(device)


def device_api(device):
    return getattr(torch, torch.device(device).type)


def owner_devices(device, count):
    first = torch.device(device)
    assert first.index is not None and first.type in {"cuda", "npu"}
    return tuple(f"{first.type}:{first.index+i}" for i in range(count))
