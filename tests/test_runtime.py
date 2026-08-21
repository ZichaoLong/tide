from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest

from tide.runtime import RuntimeConfigurationError, RuntimeRequest, normalize_device, resolve_runtime


def test_normalize_device_suffix() -> None:
    assert normalize_device("NPU:0", None) == ("npu", 0)
    assert normalize_device("cpu", None) == ("cpu", None)


def test_reject_conflicting_index() -> None:
    with pytest.raises(RuntimeConfigurationError):
        normalize_device("npu:0", 1)


def test_cpu_fp32_operator_probe() -> None:
    runtime = resolve_runtime(RuntimeRequest("cpu", 0, "float32", 7))
    assert runtime.info.backend == "cpu"
    assert runtime.info.resolution_reason == "explicit:cpu"
    assert runtime.info.probe_result == (0.0, 1.0, 4.0, 9.0)
