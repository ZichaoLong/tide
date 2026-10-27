"""KV §2/4.1: selected Full ownership and int64 distance, before softmax.

Complete independent numerical/gradient checks: weighted_kv_full_check.cpp.
"""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    directory = tmp_path_factory.mktemp("kv-full")
    source, output = directory / "adapter.cpp", directory / "adapter.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_kv_full_math.h"
extern "C" void plan(uint8_t* active,int64_t* owners,int64_t* output,int32_t* error,int64_t rows,int64_t banks){
 tide_weighted_full_device::plan(active,owners,output,error,rows,banks);
}
extern "C" void bias(int64_t* ages,float* output,int32_t* error,int64_t count,float scale,float slope){
 tide_weighted_full_device::bias(ages,output,error,count,scale,slope);
}
''')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I"+str(root/"cuda"), "-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"), str(source), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.plan.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int64]*2
    lib.bias.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64, ctypes.c_float, ctypes.c_float]
    return lib


def test_only_formally_selected_full_reads_parameter_identity(library):
    active = torch.tensor([True, False, True, False])
    owners = torch.tensor([1, -99, 0, 999])
    output = torch.zeros_like(owners); error = torch.zeros(1, dtype=torch.int32)
    library.plan(*(x.data_ptr() for x in (active, owners, output, error)), 4, 2)
    assert error.item() == 0 and output.tolist() == [1, -1, 0, -1]
    active[1] = True
    library.plan(*(x.data_ptr() for x in (active, owners, output, error)), 4, 2)
    assert error.item() == 2


@pytest.mark.parametrize("scale,slope", [(1., 0.), (7., .13), (3., .07)])
def test_distance_uses_actual_int64_age(library, scale, slope):
    age = torch.tensor([0, 1, 2**24+1, 2**54+3, 2**63-1], dtype=torch.int64)
    output = torch.zeros(age.numel()); error = torch.zeros(1, dtype=torch.int32)
    library.bias(*(x.data_ptr() for x in (age, output, error)), age.numel(), scale, slope)
    assert not error.item()
    torch.testing.assert_close(output, -age.float()/scale*slope, rtol=0, atol=0)


def test_nonfinite_numerical_distance_refused(library):
    age = torch.tensor([2**63-1]); output = torch.zeros(1); error = torch.zeros(1, dtype=torch.int32)
    library.bias(*(x.data_ptr() for x in (age, output, error)), 1, 1., torch.finfo(torch.float32).max)
    assert error.item() == 33
