"""Installed environment inspection; metadata never implies operator qualification."""
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from .identity import capture
from .records import now
from .site import load as load_site


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tool(command):
    path = shutil.which(command[0])
    if path is None:
        return dict(state="absent")
    try:
        proc = subprocess.run([path, *command[1:]], capture_output=True, text=True, timeout=10)
        return dict(path=path, exit_code=proc.returncode, version=(proc.stdout or proc.stderr).strip()[:2000])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(path=path, state="failed", error=str(error))


def inspect(*, site=None, device="cpu", probe="metadata", dtype="float32", disk_path="."):
    if not re.fullmatch(r"cpu|(?:cuda|npu):[0-9]+", device):
        raise ValueError("doctor --device: specify cpu, cuda:N or npu:N")
    if probe not in {"metadata", "operator"} or dtype not in {"float16", "float32", "float64"}:
        raise ValueError("doctor: unknown explicit probe/dtype")
    selected = load_site(site)
    # Caller-selected site variables are explicit inputs, not auto-discovered
    # substitutions. Apply them before the selected runtime's lazy import.
    os.environ.update(selected["environment"])
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    usage = shutil.disk_usage(disk_path)
    record = dict(schema="tide-doctor-v1", observed=now(), state="metadata-only", probe=probe,
        device=device, dtype=dtype, site=selected, system=dict(platform=platform.platform(),
            machine=platform.machine(), python=sys.version.split()[0], executable=sys.executable,
            affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
            cpu_max=_read("/sys/fs/cgroup/cpu.max"),
            cpu_quota_us=_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            cpu_period_us=_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            memory_max=_read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            memory_current=_read("/sys/fs/cgroup/memory.current") or _read("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            disk_total=usage.total, disk_free=usage.free),
        packages={key: _version(key) for key in ("tidegraph", "torch", "torch-npu", "pytest", "ninja")},
        tools={key: _tool(command) for key, command in (
            ("cmake", ["cmake", "--version"]), ("compiler", ["c++", "--version"]))},
        install_identity=capture(selected["paths"]), operator={"state": "unchecked"},
        qualification="metadata and selected operator smoke only; graph/client/training/resident qualification is separate")
    if device.startswith("cuda:"):
        record["tools"]["toolkit"] = _tool(["nvcc", "--version"])
        record["tools"]["driver"] = _tool(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"])
    if probe == "operator":
        record["operator"] = _operator(device, dtype)
        record["state"] = "passed"
    return record


def _operator(device, dtype):
    # Import only the explicitly selected runtime and never fall back to CPU.
    import torch
    from ..runtime import resolve_device, synchronize, manifest
    target, reason = resolve_device(device)
    payload_dtype = getattr(torch, dtype)
    if target.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    x = torch.arange(16, dtype=payload_dtype).reshape(4, 4).div(16).to(target).requires_grad_(True)
    w = torch.eye(4, dtype=payload_dtype, device=target, requires_grad=True)
    result = x@w
    result.square().sum().backward()
    synchronize(target)
    expected = torch.arange(16, dtype=payload_dtype).reshape(4, 4).div(16)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad.cpu(), 2*expected, rtol=0, atol=0)
    if not torch.isfinite(w.grad).all().item():
        raise RuntimeError("doctor operator produced a non-finite parameter gradient")
    runtime = manifest(target, reason, payload_dtype)
    if target.type == "cuda":
        runtime["compute_capability"] = list(torch.cuda.get_device_capability(target))
        runtime["compiled_architectures"] = torch.cuda.get_arch_list()
        runtime["memory_free_bytes"], runtime["memory_total_bytes"] = torch.cuda.mem_get_info(target)
        runtime["tf32"] = False
    return dict(state="passed", operation="4x4 matmul; exact representable forward/input VJP; finite parameter VJP",
                runtime=runtime, checkpoint="unchecked", full_graph="unchecked", clients="Python Torch only")
