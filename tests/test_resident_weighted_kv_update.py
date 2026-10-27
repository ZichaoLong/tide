"""KV §1 positive reception and independent-field VJP metadata boundaries."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    directory = tmp_path_factory.mktemp("kv-update")
    cpp, output = directory / "adapter.cpp", directory / "adapter.so"
    cpp.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_kv_update_math.h"
extern "C" void prepare(float* h,int64_t* owners,uint8_t* present,float* u,float* m,
 int64_t* selected,int64_t* indices,int64_t* row_ids,uint8_t* positive,int32_t* error,int64_t rows,int64_t d,int64_t b){
 tide_weighted_kv_device::prepare(h,owners,present,u,m,selected,indices,row_ids,positive,error,rows,d,b);
}
extern "C" void roots(uint8_t* raw,uint8_t* kv,uint8_t* summary,int32_t* error,int64_t rows){
 tide_weighted_kv_device::roots(raw,kv,summary,error,rows);
}
extern "C" void merge(float* mass,uint8_t* mr,float* kv,uint8_t* kr,float* summary,uint8_t* sr,
 float* output,uint8_t* connected,int32_t* error,int64_t rows,int64_t width){
 tide_weighted_kv_device::merge(mass,mr,kv,kr,summary,sr,output,connected,error,rows,width);
}
''')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-ffp-contract=off",
                    "-I" + str(root / "cuda"), "-I" + str(root / "ascendc"),
                    "-I" + str(Path(torch.__file__).parent / "include"), str(cpp), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.prepare.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int64]*3
    lib.roots.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int64]
    lib.merge.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int64]*2
    return lib


def inputs():
    h = torch.tensor([[.1, 2., -1.], [1., 0., 3.], [2., 1., 1.],
                      [0., float("nan"), float("nan")], [-1., float("nan"), float("nan")],
                      [float("nan"), float("nan"), float("nan")]])
    ids = torch.tensor([1, 0, 1, -99, -99, -99])
    present = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    u, mass = torch.zeros(6, 2), torch.zeros(6)
    selected, indices, row_ids = [torch.zeros(6, dtype=torch.int64) for _ in range(3)]
    positive, error = torch.zeros(6, dtype=torch.bool), torch.zeros(1, dtype=torch.int32)
    return h, ids, present, u, mass, selected, indices, row_ids, positive, error


def test_receive_branch_and_ignored_poison(library):
    args = inputs()
    library.prepare(*(t.data_ptr() for t in args), 6, 2, 2)
    h, _, _, u, mass, selected, indices, row_ids, positive, error = args
    assert error.item() == 0
    assert positive.tolist() == [True, True, True, False, False, False]
    assert selected.tolist() == [1, 0, 1, -1, -1, -1]
    assert indices.tolist() == [1, 0, 1, 0, 0, 0]
    assert row_ids.tolist() == [0, 1, 2, -1, -1, -1]
    torch.testing.assert_close(u[:3], h[:3, 1:])
    assert u[3:].eq(0).all() and mass[-1] == 0
    torch.testing.assert_close(mass[:-1], h[:-1, 0])


@pytest.mark.parametrize("failure", ["owner", "mass", "content", "sticky"])
def test_invalid_actual_roles(library, failure):
    args = list(inputs())
    if failure == "owner": args[1][0] = 2
    if failure == "mass": args[0][0, 0] = float("nan")
    if failure == "content": args[0][0, 1] = float("inf")
    if failure == "sticky": args[-1][0] = 41
    library.prepare(*(t.data_ptr() for t in args), 6, 2, 2)
    assert args[-1].item() == {"owner": 2, "mass": 20, "content": 20, "sticky": 41}[failure]


def test_independent_field_and_complete_content_roots(library):
    raw = torch.eye(5, dtype=torch.bool)
    kv, summary = torch.zeros(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool)
    error = torch.zeros(1, dtype=torch.int32)
    library.roots(*(t.data_ptr() for t in (raw, kv, summary, error)), 5)
    assert kv.tolist() == [False, True, True, False, False]
    assert summary.tolist() == [False, False, False, True, False]
    mass_roots = raw[:, 4].contiguous()
    mass = torch.full((5,), float("nan"));mass[4] = 0  # saturated but connected
    a = torch.full((5, 2), float("nan"));a[1:3] = torch.tensor([[2., -1.], [0., 0.]])
    b = torch.full((5, 2), float("nan"));b[3] = 4
    output, connected = torch.zeros(5, 3), torch.zeros(5, dtype=torch.bool)
    args = (mass, mass_roots, a, kv, b, summary, output, connected, error)
    library.merge(*(t.data_ptr() for t in args), 5, 2)
    assert error.item() == 0 and connected.tolist() == [False, True, True, True, True]
    torch.testing.assert_close(output, torch.tensor([[0.,0.,0.], [0.,2.,-1.], [0.,0.,0.], [0.,4.,4.], [0.,0.,0.]]))
    kv[3] = True;a[3] = torch.finfo(torch.float32).max;b[3] = torch.finfo(torch.float32).max
    library.merge(*(t.data_ptr() for t in args), 5, 2)
    assert error.item() == 23
