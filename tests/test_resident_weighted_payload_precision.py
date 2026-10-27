"""Storage support guard: positive mass, numeric underflow and absence differ.

The host adapter checks only the kernel decision; device casts and complete
online execution are independently checked by weighted-flow on each target.
"""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    work = tmp_path_factory.mktemp("weighted_payload_precision")
    source, binary = work / "adapter.cpp", work / "adapter.so"
    source.write_text('#define TIDE_KERNEL_CPU\n#include "tide_weighted_payload_guard.cpp"\n')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I" + str(root / "cuda"), "-I" + str(root / "ascendc"),
                    "-I" + str(Path(torch.__file__).parent / "include"),
                    str(source), "-o", str(binary)], check=True)
    lib = ctypes.CDLL(str(binary))
    lib.tide_weighted_payload_guard.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
    return lib


@pytest.mark.parametrize("mass,content,present,refused", [
    (2**-24, 2**-25, True, False),  # Positive mass survives; G may round to zero.
    (2**-25, 1., True, True),      # Losing positive sending support is refused.
    (0., 2**-25, True, False),     # A present zero is a distinct valid payload.
    (1., 1e5, True, True),         # Finite arithmetic can overflow storage.
    (float("nan"), float("inf"), False, False),
    (-1., 1., True, True),
])
def test_storage_preserves_sending_support(library, mass, content, present, refused):
    source = torch.tensor([[mass, content]], dtype=torch.float32)
    rounded = source.half().float()
    active = torch.tensor([present]); error = torch.zeros(1, dtype=torch.int32)
    call = lambda: library.tide_weighted_payload_guard(
        source.data_ptr(), rounded.data_ptr(), active.data_ptr(), error.data_ptr(), 1, 2)
    call()
    assert error.item() == (24 if refused else 0)
    error.fill_(7); call()
    assert error.item() == 7  # An earlier transaction failure is never cleared.
