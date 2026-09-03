"""Project-owned runtime boundary for CPU, CUDA, and Ascend NPU.

Only the selected framework stack is imported, and all Torch imports are lazy.
This keeps ``python -m tide.runtime --help`` usable before Torch is installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence, Tuple


_BACKENDS = {"auto", "cpu", "cuda", "npu"}
_DTYPES = {"auto", "float64", "float32", "float16", "bfloat16"}
_ACCELERATOR_BACKENDS = {"cuda", "npu"}
_PROBE_TIMEOUT_SECONDS = 60
_PROBE_NAME = "elementwise-square-forward-backward"


class RuntimeConfigurationError(ValueError):
    """The requested runtime syntax or project policy is invalid."""


class RuntimeUnavailableError(RuntimeError):
    """The selected runtime cannot pass its minimum capability probe."""


@dataclasses.dataclass(frozen=True)
class RuntimeRequest:
    """A normalized runtime request.

    ``backend`` may also use the convenience forms ``cuda:N`` and ``npu:N``;
    :func:`resolve_runtime` normalizes them before importing Torch.
    """

    backend: str = "auto"
    device_index: Optional[int] = None
    dtype: str = "auto"
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class RuntimeInfo:
    """Serializable facts for a successfully resolved runtime."""

    requested_backend: str
    backend: str
    resolution_reason: str
    device_index: int
    device: str
    requested_dtype: str
    dtype: str
    dtype_resolution_reason: str
    seed: int
    torch_version: str
    plugin_version: Optional[str]
    device_count: int
    device_name: str
    device_architecture: Optional[str]
    library_architectures: Tuple[str, ...]
    capability_probe: str
    probe_forward: Tuple[float, ...]
    probe_gradient: Tuple[float, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-compatible runtime metadata without module objects."""

        result = dataclasses.asdict(self)
        result["library_architectures"] = list(self.library_architectures)
        result["probe_forward"] = list(self.probe_forward)
        result["probe_gradient"] = list(self.probe_gradient)
        return result


class Runtime:
    """Resolved runtime plus the selected Torch module and options."""

    def __init__(self, info: RuntimeInfo, torch_module: Any) -> None:
        self.info = info
        self.torch = torch_module
        self.device = torch_module.device(info.device)
        self.dtype = getattr(torch_module, info.dtype)

    def synchronize(self) -> None:
        """Synchronize only the selected accelerator family."""

        try:
            _synchronize_backend(
                self.torch, self.info.backend, self.info.device_index
            )
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"failed to synchronize {self.info.device}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def to_dict(self) -> Dict[str, Any]:
        return self.info.to_dict()


def add_runtime_arguments(
    parser: argparse.ArgumentParser, *, require_explicit_device: bool = False
) -> None:
    """Add the shared runtime flags to an application parser.

    Training and benchmark entry points must set ``require_explicit_device``;
    diagnostics may retain the documented ``auto`` default.
    """

    parser.add_argument(
        "--device",
        required=require_explicit_device,
        default=None if require_explicit_device else "auto",
        help="auto, cpu, cuda, npu, cuda:LOGICAL_INDEX, or npu:LOGICAL_INDEX",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help=(
            "logical index after launcher visibility remapping; conflicts with "
            "a device suffix"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="auto",
        help=(
            "auto, float64, float32, float16, or bfloat16; auto resolves to "
            "float32 and float64 is the CPU oracle only"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)


def request_from_args(args: argparse.Namespace) -> RuntimeRequest:
    backend, index = normalize_device(args.device, args.device_index)
    return RuntimeRequest(backend, index, args.dtype, args.seed)


def normalize_device(
    device: str, device_index: Optional[int]
) -> Tuple[str, Optional[int]]:
    """Normalize a backend and logical index without importing Torch."""

    if not isinstance(device, str):
        raise RuntimeConfigurationError("device must be a string")
    value = device.strip().lower()
    suffix_index: Optional[int] = None
    if ":" in value:
        backend, suffix = value.split(":", 1)
        if backend not in _ACCELERATOR_BACKENDS:
            raise RuntimeConfigurationError(
                "only cuda and npu accept a ':LOGICAL_INDEX' suffix"
            )
        if not suffix:
            raise RuntimeConfigurationError("device suffix must be an integer")
        try:
            suffix_index = int(suffix)
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "device suffix must be an integer"
            ) from exc
        value = backend
    if value not in _BACKENDS:
        raise RuntimeConfigurationError(
            "device must be one of auto, cpu, cuda, npu, cuda:N, or npu:N"
        )
    if suffix_index is not None and device_index is not None:
        raise RuntimeConfigurationError(
            "specify a logical index either in --device or --device-index, not both"
        )
    index = suffix_index if suffix_index is not None else device_index
    if index is not None and index < 0:
        raise RuntimeConfigurationError("device index must be nonnegative")
    if value == "cpu" and index not in {None, 0}:
        raise RuntimeConfigurationError("CPU accepts only logical device index 0")
    if value == "auto" and index is not None:
        raise RuntimeConfigurationError(
            "an index with auto is ambiguous; request cuda or npu explicitly"
        )
    return value, index


def _isolated_availability(backend: str) -> Dict[str, Any]:
    """Probe one accelerator family in a fresh interpreter."""

    if backend not in _ACCELERATOR_BACKENDS:
        raise AssertionError("isolated probes are only for accelerator families")
    code = r"""
import importlib
import json
import os

result = {"available": False, "count": 0, "error": None}
try:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    torch = importlib.import_module("torch")
    backend = os.environ["TIDE_RUNTIME_PROBE_BACKEND"]
    if backend == "npu":
        importlib.import_module("torch_npu")
    api = getattr(torch, backend)
    result["available"] = bool(api.is_available())
    result["count"] = int(api.device_count())
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print("TIDE_RUNTIME_PROBE=" + json.dumps(result, sort_keys=True))
"""
    probe_env = os.environ.copy()
    probe_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    probe_env["TIDE_RUNTIME_PROBE_BACKEND"] = backend
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            env=probe_env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "count": 0,
            "error": f"probe exceeded {_PROBE_TIMEOUT_SECONDS} seconds",
            "fatal": True,
            "diagnostic_stdout": str(exc.stdout or "")[-2000:],
            "diagnostic_stderr": str(exc.stderr or "")[-2000:],
        }
    except OSError as exc:
        return {
            "available": False,
            "count": 0,
            "error": f"cannot start probe: {type(exc).__name__}: {exc}",
            "fatal": True,
        }

    marker = "TIDE_RUNTIME_PROBE="
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            result = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            continue
        result["returncode"] = completed.returncode
        result["fatal"] = completed.returncode != 0
        if completed.stderr:
            result["diagnostic_stderr"] = completed.stderr[-2000:]
        return result
    return {
        "available": False,
        "count": 0,
        "error": "probe process did not emit structured output",
        "fatal": True,
        "returncode": completed.returncode,
        "diagnostic_stdout": completed.stdout[-2000:],
        "diagnostic_stderr": completed.stderr[-2000:],
    }


def _select_backend_with_reason(requested: str) -> Tuple[str, str]:
    if requested != "auto":
        return requested, f"explicit:{requested}"
    probes = {
        name: _isolated_availability(name) for name in ("cuda", "npu")
    }
    fatal = {name: result for name, result in probes.items() if result.get("fatal")}
    if fatal:
        raise RuntimeUnavailableError(
            "auto backend probing failed; select a backend explicitly after "
            f"inspecting the environment; probes={json.dumps(fatal, sort_keys=True)}"
        )
    available = [name for name, result in probes.items() if result.get("available")]
    if len(available) > 1:
        raise RuntimeUnavailableError(
            "auto detected both CUDA and NPU; select one explicitly to avoid "
            f"mixing vendor stacks; probes={json.dumps(probes, sort_keys=True)}"
        )
    if available:
        selected = available[0]
        return selected, f"auto:single-visible-{selected}"
    return "cpu", "auto:no-accelerator-selected-cpu"


def _package_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _import_selected_torch(backend: str) -> Tuple[Any, Optional[str]]:
    """Import Torch and, only for NPU, register TorchNPU immediately."""

    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeUnavailableError(
            f"cannot import torch: {type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(torch, "__version__") or not hasattr(torch, "device"):
        raise RuntimeUnavailableError(
            "the imported 'torch' namespace is not a usable PyTorch runtime"
        )

    plugin_version: Optional[str] = None
    if backend == "npu":
        try:
            plugin = importlib.import_module("torch_npu")
        except Exception as exc:
            raise RuntimeUnavailableError(
                "NPU was selected but torch_npu import failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        plugin_version = getattr(plugin, "__version__", None)
        if plugin_version is None:
            plugin_version = _package_version("torch-npu", "torch_npu")
    return torch, plugin_version


def _resolve_dtype(requested: str, backend: str) -> Tuple[str, str]:
    if requested not in _DTYPES:
        raise RuntimeConfigurationError(f"invalid dtype: {requested}")
    dtype = "float32" if requested == "auto" else requested
    if dtype == "float64" and backend != "cpu":
        raise RuntimeConfigurationError(
            "float64 is reserved for the CPU oracle; CUDA and NPU requests must "
            "use float32, float16, bfloat16, or auto"
        )
    reason = "auto:project-default-float32" if requested == "auto" else f"explicit:{dtype}"
    return dtype, reason


def _seed_selected_runtime(torch: Any, backend: str, seed: int) -> None:
    """Seed Python, Torch CPU, and only the selected accelerator family."""

    random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed(seed)
    elif backend == "npu":
        torch.npu.manual_seed(seed)


def _synchronize_backend(torch: Any, backend: str, index: int) -> None:
    if backend == "cuda":
        torch.cuda.synchronize(index)
    elif backend == "npu":
        torch.npu.synchronize(index)


def _minimum_operator_probe(
    torch: Any, backend: str, index: int, device: str, dtype: Any
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Run a tiny dtype-specific forward/backward capability probe."""

    x = torch.tensor(
        [0.0, 1.0, 2.0, 3.0],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    y = x * x
    y.sum().backward()
    _synchronize_backend(torch, backend, index)
    if x.grad is None:
        raise RuntimeError("minimum operator probe produced no input gradient")
    forward = tuple(float(value) for value in y.detach().float().cpu().tolist())
    gradient = tuple(
        float(value) for value in x.grad.detach().float().cpu().tolist()
    )
    if forward != (0.0, 1.0, 4.0, 9.0):
        raise RuntimeError(f"unexpected forward probe result: {forward}")
    if gradient != (0.0, 2.0, 4.0, 6.0):
        raise RuntimeError(f"unexpected gradient probe result: {gradient}")
    return forward, gradient


def resolve_runtime(request: RuntimeRequest) -> Runtime:
    """Resolve, seed, and capability-probe exactly one backend family."""

    if request.seed < 0:
        raise RuntimeConfigurationError("seed must be nonnegative")
    backend_request, requested_index = normalize_device(
        request.backend, request.device_index
    )
    if request.dtype not in _DTYPES:
        raise RuntimeConfigurationError(f"invalid dtype: {request.dtype}")

    backend, resolution_reason = _select_backend_with_reason(backend_request)
    dtype_name, dtype_resolution_reason = _resolve_dtype(request.dtype, backend)
    torch, plugin_version = _import_selected_torch(backend)

    index = requested_index if requested_index is not None else 0
    if backend == "cpu":
        if index != 0:
            raise RuntimeConfigurationError("CPU accepts only logical device index 0")
        device_count = 1
        device_name = "CPU"
        device_architecture = None
        library_architectures: Tuple[str, ...] = ()
        device = "cpu"
    else:
        api = getattr(torch, backend, None)
        if api is None:
            raise RuntimeUnavailableError(
                f"{backend} was selected but this Torch runtime has no torch.{backend}"
            )
        try:
            available = bool(api.is_available())
            device_count = int(api.device_count())
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"{backend} availability probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not available:
            raise RuntimeUnavailableError(
                f"{backend} was selected but is_available() is false; no CPU fallback "
                "was attempted"
            )
        if index >= device_count:
            raise RuntimeUnavailableError(
                f"logical {backend} index {index} is out of range for "
                f"{device_count} visible devices"
            )
        device = f"{backend}:{index}"
        try:
            api.set_device(index)
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"failed to select {device}: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            device_name = str(api.get_device_name(index))
        except Exception:
            device_name = device
        device_architecture = None
        library_architectures = ()
        if backend == "cuda":
            try:
                major, minor = torch.cuda.get_device_capability(index)
                device_architecture = f"sm_{major}{minor}"
            except Exception:
                device_architecture = None
            try:
                library_architectures = tuple(torch.cuda.get_arch_list())
            except Exception:
                library_architectures = ()

    try:
        torch_dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        raise RuntimeUnavailableError(
            f"PyTorch {torch.__version__} does not expose dtype {dtype_name}"
        ) from exc

    try:
        _seed_selected_runtime(torch, backend, request.seed)
    except Exception as exc:
        raise RuntimeUnavailableError(
            f"failed to seed {device}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        probe_forward, probe_gradient = _minimum_operator_probe(
            torch, backend, index, device, torch_dtype
        )
    except Exception as exc:
        raise RuntimeUnavailableError(
            f"minimum {_PROBE_NAME} probe failed for {device}/{dtype_name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    info = RuntimeInfo(
        requested_backend=backend_request,
        backend=backend,
        resolution_reason=resolution_reason,
        device_index=index,
        device=device,
        requested_dtype=request.dtype,
        dtype=dtype_name,
        dtype_resolution_reason=dtype_resolution_reason,
        seed=request.seed,
        torch_version=str(torch.__version__),
        plugin_version=plugin_version,
        device_count=device_count,
        device_name=device_name,
        device_architecture=device_architecture,
        library_architectures=library_architectures,
        capability_probe=_PROBE_NAME,
        probe_forward=probe_forward,
        probe_gradient=probe_gradient,
    )
    return Runtime(info, torch)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and probe the TIDE Torch runtime"
    )
    add_runtime_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the low-risk runtime diagnostic CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        runtime = resolve_runtime(request_from_args(args))
    except (RuntimeConfigurationError, RuntimeUnavailableError) as exc:
        parser.exit(2, f"runtime error: {exc}\n")
    print(json.dumps(runtime.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
