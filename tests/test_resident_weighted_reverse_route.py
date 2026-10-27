"""PDG finite-cut ancestry: physical message VJPs, exact clocks and pending.

CPU execution of device metadata source, not a resident runtime fallback.
"""
import ctypes
from pathlib import Path
import subprocess

import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device = Path(__file__).resolve().parents[1] / "tools/device_online"
    out = tmp_path_factory.mktemp("weighted-reverse-route")
    source = out / "adapter.cpp"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_reverse_route_math.h"
extern "C" void route(int64_t* coordinates,uint8_t* valid_atoms,int64_t* indices,int64_t* lengths,float* gradients,uint8_t* roots,int64_t* edges,
 int64_t* times,uint8_t* present,float* bars,uint8_t* flags,int64_t* boundary,uint8_t* valid,float* boundary_bars,uint8_t* boundary_flags,
 int64_t* stage,int64_t* cut,int32_t* error,int64_t rows,int64_t slots,int64_t atoms,int64_t width,int64_t edge_count,int64_t stages,int64_t nodes,int64_t samples,int64_t boundary_count){
 tide_weighted_reverse_device::route(coordinates,valid_atoms,indices,lengths,gradients,roots,edges,times,present,bars,flags,boundary,valid,boundary_bars,boundary_flags,
 stage,cut,error,rows,slots,atoms,width,edge_count,stages,nodes,samples,boundary_count);}
''')
    target = out / "route.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off",
                    "-I"+str(device/"cuda"), "-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"), str(source), "-o", str(target)], check=True)
    lib = ctypes.CDLL(str(target))
    lib.route.argtypes = [ctypes.c_void_p]*18 + [ctypes.c_int64]*9
    return lib


def fixture():
    # Three event stages, two nodes, two independent samples. Repeated physical
    # edges 0/1 share a producer but remain two separate adjoint contributions.
    huge = 2**54+3
    coords = torch.tensor([[0,1,6,1,0,4], [0,1,6,1,1,4], [0,0,9,1,2,8],
                           [1,0,huge,0,0,0], [0,1,4,1,0,1], [1,1,9,1,2,8]])
    times = torch.zeros(3,2,2, dtype=torch.int64)
    present = torch.zeros_like(times, dtype=torch.bool)
    times[0,0,0], present[0,0,0] = 4, True
    times[1,1,0], present[1,1,0] = 8, True
    times[1,1,1], present[1,1,1] = 8, True
    return dict(coordinates=coords, valid_atoms=torch.ones(6,dtype=torch.bool), indices=torch.arange(6).reshape(2,3), lengths=torch.tensor([3,3]),
                gradients=torch.tensor([[[1.,2.,3.],[-1.,4.,0.],[.5,0.,1.]],
                                        [[0.,0.,0.],[2.,-1.,.5],[float("nan")]*3]]),
                roots=torch.tensor([[True,True,True],[True,True,False]]), edges=torch.tensor([0,0,1]),
                times=times, present=present, bars=torch.zeros(3,2,2,3), flags=torch.zeros(3,2,2,dtype=torch.bool),
                boundary=torch.stack((coords[4],coords[3])), valid=torch.ones(2,dtype=torch.bool),
                boundary_bars=torch.zeros(2,3), boundary_flags=torch.zeros(2,dtype=torch.bool),
                stage=torch.tensor([2]), cut=torch.tensor([3]), error=torch.zeros(1,dtype=torch.int32))


def run(lib, x):
    lib.route(*(t.data_ptr() for t in x.values()), 2,3,6,3,3,3,2,2,2)
    return x["error"].item()


def test_parallel_edges_private_samples_zero_roots_and_exact_boundary(library):
    x = fixture()
    assert run(library, x) == 0
    torch.testing.assert_close(x["bars"][0,0,0], torch.tensor([0.,6.,3.]))
    torch.testing.assert_close(x["bars"][1,1,0], torch.tensor([.5,0.,1.]))
    assert x["flags"].sum() == 2 and not x["flags"][1,1,1]
    torch.testing.assert_close(x["boundary_bars"], torch.tensor([[2.,-1.,.5],[0.,0.,0.]]))
    assert x["boundary_flags"].all()  # Connected zero external root stays present.
    # Repeated caller uses add to existing roots; the kernel never clears them.
    assert run(library, x) == 0
    torch.testing.assert_close(x["bars"][0,0,0], torch.tensor([0.,12.,6.]))


def test_absent_destination_values_are_not_read(library):
    x = fixture()
    x["bars"].fill_(float("nan"))
    x["boundary_bars"].fill_(float("nan"))
    assert run(library, x) == 0
    torch.testing.assert_close(x["bars"][0,0,0], torch.tensor([0.,6.,3.]))
    assert x["boundary_flags"].all() and x["boundary_bars"][1].eq(0).all()


def test_distinguish_adjacent_large_integer_producer_times(library):
    x = fixture()
    huge = 2**54+3
    x["coordinates"][:2,2], x["coordinates"][:2,5] = huge+2, huge
    x["times"][0,0,0] = huge
    x["present"][1,0,0], x["times"][1,0,0] = True, huge-1
    assert run(library, x) == 0
    torch.testing.assert_close(x["bars"][0,0,0], torch.tensor([0.,6.,3.]))
    assert not x["flags"][1,0,0]


def test_accumulation_overflow_refuses_whole_result_and_is_sticky(library):
    x = fixture()
    x["bars"][0,0,0,0] = torch.finfo(torch.float32).max
    x["flags"][0,0,0] = True
    x["gradients"][0,0,0] = torch.finfo(torch.float32).max
    assert run(library, x) == 37
    saved = x["bars"].clone()
    x["gradients"].zero_()
    assert run(library, x) == 37
    assert torch.equal(saved, x["bars"])


@pytest.mark.parametrize("fault", ["missing_producer", "ambiguous_producer", "future_stage",
                                  "missing_pending", "duplicate_boundary", "edge", "absent_atom", "nonfinite"])
def test_refuse_invalid_ancestry_and_connected_nonfinite(library, fault):
    x = fixture()
    if fault == "missing_producer":
        x["present"][0,0,0] = False
    elif fault == "ambiguous_producer":
        x["present"][1,0,0], x["times"][1,0,0] = True, 4
    elif fault == "future_stage":
        x["stage"][0] = 0
    elif fault == "missing_pending":
        x["valid"][0] = False
    elif fault == "duplicate_boundary":
        x["boundary"][0] = x["boundary"][1]
    elif fault == "edge":
        x["coordinates"][0,4] = 3
    elif fault == "absent_atom":
        x["valid_atoms"][0] = False
    else:
        x["gradients"][0,0,0] = float("nan")
    assert run(library, x) == (37 if fault == "nonfinite" else 41)
