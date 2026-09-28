"""Small runtime boundary shared by Python entry points and NPU probes.

The graph implementation itself remains device neutral.  This module is the
only place that imports TorchNPU or selects a logical accelerator index.
"""
from __future__ import annotations

import platform
from typing import Any

import torch


def _split_device(spec: str, index: int | None) -> tuple[str, int | None]:
    if index is not None and (type(index) is not int or not 0 <= index < 2**15):
        raise ValueError("device index must be a nonnegative integer below 32768")
    if not isinstance(spec, str) or not spec:
        raise ValueError("device must be auto, cpu, cuda[:N] or npu[:N]")
    if ":" in spec:
        backend, suffix = spec.split(":", 1)
        if index is not None:
            raise ValueError("specify a logical index either in device or device-index, not both")
        if not suffix.isdigit():
            raise ValueError("device index must be a nonnegative integer")
        index = int(suffix)
        if index >= 2**15:
            raise ValueError("device index must be below 32768")
        spec = backend
    if spec not in {"auto", "cpu", "cuda", "npu"}:
        raise ValueError("device must be auto, cpu, cuda[:N] or npu[:N]")
    return spec, index


def _load_npu() -> Any:
    try:
        import torch_npu  # noqa: F401  # explicit registration is required
    except Exception as error:  # pragma: no cover - depends on the site stack
        raise RuntimeError(
            "NPU requested but torch_npu could not be imported; load the documented CANN/TorchNPU stack"
        ) from error
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available() or npu.device_count() < 1:
        raise RuntimeError("NPU requested but no usable logical NPU is visible")
    return npu


def resolve_device(spec: str, index: int | None = None) -> tuple[torch.device, str]:
    """Resolve one logical device and return it with a stable reason."""
    backend, index = _split_device(spec, index)
    if backend == "auto":
        cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        npu = False
        try:
            npu_api = _load_npu()
            npu = npu_api.device_count() > 0
        except RuntimeError:
            npu_api = None
        if cuda and npu:
            raise RuntimeError("auto device selection is ambiguous: both CUDA and NPU are visible")
        if npu:
            if index is not None:
                raise ValueError("device-index is not valid with device=auto")
            npu_api.set_device("npu:0")
            return torch.device("npu:0"), "auto:single-visible-npu"
        if cuda:
            if index is not None:
                raise ValueError("device-index is not valid with device=auto")
            torch.cuda.set_device(0)
            return torch.device("cuda:0"), "auto:single-visible-cuda"
        if index is not None:
            raise ValueError("device-index is not valid with device=auto")
        return torch.device("cpu"), "auto:no-accelerator-selected-cpu"
    if backend == "cpu":
        if index not in (None, 0):
            raise ValueError("CPU accepts only logical device index 0")
        return torch.device("cpu"), "explicit:cpu"
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but this Torch build has no usable CUDA runtime")
        count = torch.cuda.device_count()
        logical = 0 if index is None else index
        if logical < 0:
            raise ValueError("device index must be nonnegative")
        if logical >= count:
            raise RuntimeError(f"CUDA logical device {logical} is unavailable (count={count})")
        torch.cuda.set_device(logical)
        return torch.device("cuda", logical), "explicit:cuda"
    npu_api = _load_npu()
    count = npu_api.device_count()
    logical = 0 if index is None else index
    if logical < 0:
        raise ValueError("device index must be nonnegative")
    if logical >= count:
        raise RuntimeError(f"NPU logical device {logical} is unavailable (count={count})")
    npu_api.set_device(f"npu:{logical}")
    return torch.device("npu", logical), "explicit:npu"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        _load_npu().synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "float64": torch.float64,
              "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def manifest(device: torch.device, reason: str, dtype: torch.dtype) -> dict[str, Any]:
    """Return a small sanitized runtime record suitable for project evidence."""
    record: dict[str, Any] = {
        "resolved_device": str(device),
        "backend": device.type,
        "logical_device_index": device.index,
        "resolution_reason": reason,
        "dtype": str(dtype).split(".")[-1],
        "host_architecture": platform.machine(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "python": platform.python_version(),
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
    }
    if device.type == "npu":
        npu = _load_npu()
        record.update(torch_npu=getattr(__import__("torch_npu"), "__version__", "unknown"),
                      logical_device_count=npu.device_count(), device_name=npu.get_device_name(device))
        import os
        record["cann_version"] = os.environ.get("ASCEND_CANN_VERSION", "unknown")
        record["matmul_allow_hf32"] = getattr(getattr(npu, "matmul", None), "allow_hf32", None)
    elif device.type == "cuda":
        record["logical_device_count"] = torch.cuda.device_count()
        record.update(cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      device_name=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)),
                      compiled_architectures=torch.cuda.get_arch_list(),
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    else:
        record["logical_device_count"] = 1
    return record
