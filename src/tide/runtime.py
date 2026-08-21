"""Small runtime boundary shared by CPU, CUDA, and Ascend NPU entry points.

This module deliberately has no module-level Torch or vendor import.  Static
commands such as ``tide --help`` therefore remain usable without an
accelerator stack.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import subprocess
import sys
from typing import Any


BACKENDS = {"auto", "cpu", "cuda", "npu"}
DTYPES = {"auto", "float32", "float16", "bfloat16"}


class RuntimeConfigurationError(ValueError):
    """The requested runtime configuration is invalid."""


class RuntimeUnavailableError(RuntimeError):
    """The requested runtime cannot execute the required operator probe."""


@dataclasses.dataclass(frozen=True)
class RuntimeRequest:
    backend: str
    device_index: int | None = None
    dtype: str = "auto"
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class RuntimeInfo:
    backend: str
    resolution_reason: str
    device_index: int
    device: str
    dtype: str
    seed: int
    torch_version: str
    plugin_version: str | None
    device_count: int
    device_name: str
    probe_result: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["probe_result"] = list(self.probe_result)
        return result


class Runtime:
    def __init__(self, info: RuntimeInfo, torch_module: Any) -> None:
        self.info = info
        self.torch = torch_module
        self.device = torch_module.device(info.device)
        self.dtype = getattr(torch_module, info.dtype)

    def synchronize(self) -> None:
        if self.info.backend == "cuda":
            self.torch.cuda.synchronize(self.info.device_index)
        elif self.info.backend == "npu":
            self.torch.npu.synchronize(self.info.device_index)

    def memory_allocated(self) -> int | None:
        if self.info.backend == "cuda":
            return int(self.torch.cuda.max_memory_allocated(self.info.device_index))
        if self.info.backend == "npu":
            return int(self.torch.npu.max_memory_allocated(self.info.device_index))
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.info.to_dict()


def normalize_device(value: str, explicit_index: int | None) -> tuple[str, int | None]:
    value = value.strip().lower()
    suffix_index: int | None = None
    if ":" in value:
        backend, suffix = value.split(":", 1)
        if backend not in {"cuda", "npu"}:
            raise RuntimeConfigurationError("only cuda and npu accept a logical index suffix")
        try:
            suffix_index = int(suffix)
        except ValueError as exc:
            raise RuntimeConfigurationError("device suffix must be an integer") from exc
        value = backend
    if value not in BACKENDS:
        raise RuntimeConfigurationError("device must be auto, cpu, cuda, npu, cuda:N, or npu:N")
    if suffix_index is not None and explicit_index is not None:
        raise RuntimeConfigurationError("specify the logical index only once")
    index = suffix_index if suffix_index is not None else explicit_index
    if index is not None and index < 0:
        raise RuntimeConfigurationError("device index must be nonnegative")
    if value == "auto" and index is not None:
        raise RuntimeConfigurationError("auto cannot be combined with a device index")
    if value == "cpu" and index not in {None, 0}:
        raise RuntimeConfigurationError("CPU accepts only logical index 0")
    return value, index


def _isolated_availability(backend: str) -> dict[str, Any]:
    code = r'''
import importlib
import json
import os
result = {"available": False, "count": 0, "error": None}
try:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    torch = importlib.import_module("torch")
    backend = os.environ["TIDE_PROBE_BACKEND"]
    if backend == "npu":
        importlib.import_module("torch_npu")
    api = getattr(torch, backend)
    result["available"] = bool(api.is_available())
    result["count"] = int(api.device_count())
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print("TIDE_PROBE=" + json.dumps(result, sort_keys=True))
'''
    environment = os.environ.copy()
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    environment["TIDE_PROBE_BACKEND"] = backend
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "count": 0, "fatal": True, "error": str(exc)}
    marker = "TIDE_PROBE="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            result = json.loads(line[len(marker) :])
            result["fatal"] = completed.returncode != 0
            return result
    return {
        "available": False,
        "count": 0,
        "fatal": True,
        "error": "probe process did not return structured output",
        "stderr": completed.stderr[-1000:],
    }


def _select_backend(requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, f"explicit:{requested}"
    probes = {name: _isolated_availability(name) for name in ("cuda", "npu")}
    if any(result.get("fatal") for result in probes.values()):
        raise RuntimeUnavailableError(
            "automatic accelerator probing failed; select a backend explicitly; "
            f"probes={json.dumps(probes, sort_keys=True)}"
        )
    available = [name for name, result in probes.items() if result["available"]]
    if len(available) > 1:
        raise RuntimeUnavailableError("both CUDA and NPU are visible; select one explicitly")
    if available:
        return available[0], f"auto:single-visible-{available[0]}"
    return "cpu", "auto:no-accelerator-selected-cpu"


def resolve_runtime(request: RuntimeRequest) -> Runtime:
    if request.backend not in BACKENDS:
        raise RuntimeConfigurationError(f"invalid backend: {request.backend}")
    if request.dtype not in DTYPES:
        raise RuntimeConfigurationError(f"invalid dtype: {request.dtype}")
    if request.seed < 0:
        raise RuntimeConfigurationError("seed must be nonnegative")

    backend, reason = _select_backend(request.backend)
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeUnavailableError(f"cannot import torch: {type(exc).__name__}: {exc}") from exc

    plugin_version: str | None = None
    if backend == "npu":
        try:
            plugin = importlib.import_module("torch_npu")
            plugin_version = str(getattr(plugin, "__version__", "unknown"))
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"NPU was requested but torch_npu import failed: {type(exc).__name__}: {exc}"
            ) from exc

    index = 0 if request.device_index is None else request.device_index
    if backend == "cpu":
        if index != 0:
            raise RuntimeConfigurationError("CPU accepts only logical index 0")
        device, count, name = "cpu", 1, "CPU"
    else:
        api = getattr(torch, backend, None)
        if api is None:
            raise RuntimeUnavailableError(f"this Torch runtime has no torch.{backend}")
        try:
            available, count = bool(api.is_available()), int(api.device_count())
        except Exception as exc:
            raise RuntimeUnavailableError(f"{backend} availability probe failed: {exc}") from exc
        if not available:
            raise RuntimeUnavailableError(f"{backend} was requested but is_available() is false")
        if index >= count:
            raise RuntimeUnavailableError(f"logical {backend} index {index} is outside {count} visible devices")
        try:
            api.set_device(index)
            name = str(api.get_device_name(index))
        except Exception as exc:
            raise RuntimeUnavailableError(f"cannot select {backend}:{index}: {exc}") from exc
        device = f"{backend}:{index}"

    dtype_name = "float32" if request.dtype == "auto" else request.dtype
    if not hasattr(torch, dtype_name):
        raise RuntimeUnavailableError(f"Torch does not expose dtype {dtype_name}")
    random.seed(request.seed)
    torch.manual_seed(request.seed)
    if backend == "cuda":
        torch.cuda.manual_seed(request.seed)
    elif backend == "npu":
        torch.npu.manual_seed(request.seed)

    try:
        x = torch.arange(4, device=device, dtype=getattr(torch, dtype_name))
        y = x * x
        if backend == "cuda":
            torch.cuda.synchronize(index)
        elif backend == "npu":
            torch.npu.synchronize(index)
        probe = tuple(float(item) for item in y.float().cpu().tolist())
    except Exception as exc:
        raise RuntimeUnavailableError(
            f"required operator probe failed on {device}/{dtype_name}: {type(exc).__name__}: {exc}"
        ) from exc

    return Runtime(
        RuntimeInfo(
            backend=backend,
            resolution_reason=reason,
            device_index=index,
            device=device,
            dtype=dtype_name,
            seed=request.seed,
            torch_version=str(torch.__version__),
            plugin_version=plugin_version,
            device_count=count,
            device_name=name,
            probe_result=probe,
        ),
        torch,
    )
