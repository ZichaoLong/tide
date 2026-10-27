"""Exact replica identity precedes any distributed optimizer commit."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def guard(tmp_path_factory):
    source = Path(__file__).resolve().parents[1]/"tools/device_online"
    work = tmp_path_factory.mktemp("weighted_peer_identity")
    adapter = work/"check.cpp"
    adapter.write_text('#define TIDE_KERNEL_CPU\n#include "tide_weighted_peer_equal.cpp"\n')
    library = work/"check.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-I"+str(source/"cuda"),
                    "-I"+str(source/"ascendc"), "-I"+str(Path(torch.__file__).parent/"include"),
                    str(adapter), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.tide_weighted_peer_equal.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int64]
    return lib.tide_weighted_peer_equal


@pytest.mark.parametrize("left,right", [
    (torch.tensor([2**55+17]), torch.tensor([2**55+18])),
    (torch.tensor([0.]), torch.tensor([-0.])),
    (torch.tensor([.3]), torch.tensor([.30001])),
    (torch.tensor([2**-24], dtype=torch.float16), torch.tensor([0.], dtype=torch.float16)),
])
def test_replica_difference_is_exact_and_sticky(guard, left, right):
    error = torch.zeros(1, dtype=torch.int32)
    copy = left.clone()
    guard(left.data_ptr(), copy.data_ptr(), error.data_ptr(), left.nbytes)
    assert error == 0
    guard(left.data_ptr(), right.data_ptr(), error.data_ptr(), left.nbytes)
    assert error == 25
    error.fill_(20)
    guard(left.data_ptr(), right.data_ptr(), error.data_ptr(), left.nbytes)
    assert error == 20
