"""Resident CUDA build must not inherit wheel/default hardware architecture flags."""
import os
from pathlib import Path
import subprocess

import pytest


CONFIG = Path(__file__).resolve().parents[1] / "tools/device_online/CudaArchitectures.cmake"


def configure(tmp_path, request):
    script = tmp_path / "architectures.cmake"
    script.write_text(
        f'set(TIDE_CUDA_ARCHITECTURES "{request}")\n'
        f'include("{CONFIG}")\n'
        'message(STATUS "effective=$ENV{TORCH_CUDA_ARCH_LIST}")\n'
    )
    return subprocess.run(
        ["cmake", "-P", str(script)], capture_output=True, text=True,
        env={**os.environ, "TORCH_CUDA_ARCH_LIST": "5.0;6.0;7.0"},
    )


def test_explicit_architectures_override_torch_defaults(tmp_path):
    result = configure(tmp_path, "80;90;100;90")
    assert result.returncode == 0, result.stderr
    assert "effective=8.0;9.0;10.0" in result.stdout


@pytest.mark.parametrize("architectures", ["", "50", "80;75", "native", "all", "90-virtual", "90-real"])
def test_ambiguous_or_incompatible_architectures_rejected(tmp_path, architectures):
    result = configure(tmp_path, architectures)
    assert result.returncode != 0
    assert "Resident CUDA" in result.stderr
