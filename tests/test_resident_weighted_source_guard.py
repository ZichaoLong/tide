"""0770593 source adaptation: connection masks precede vendor adjoint products."""
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("source-guard")
    generated=directory/"generated"
    subprocess.run([sys.executable,str(device/"cuda/generate_kernels.py"),"--source",str(device/"ascendc"),
                    "--output",str(generated),"--backend","cpu"],check=True)
    path=directory/"guard.so"
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),
                    str(generated/"tide_weighted_source_guard.cpp"),"-o",str(path)],check=True)
    lib=ctypes.CDLL(str(path));lib.tide_cuda_launch_tide_weighted_source_guard.argtypes=(
        [ctypes.c_uint32,ctypes.c_void_p]+[ctypes.c_void_p]*5+[ctypes.c_int64]*3)
    return lib


def call(lib,x,roots,out,errors,error,stage):
    assert lib.tide_cuda_launch_tide_weighted_source_guard(1,None,
        *(t.data_ptr() for t in (x,roots,out,errors,error)),x.shape[0],x.numel()//x.shape[0],stage)==0


def test_absent_poison_and_connected_zero(library):
    x=torch.tensor([[0.,0.],[float("nan"),float("inf")],[.3,-.2]])
    roots=torch.tensor([True,False,True]);out=torch.full_like(x,42)
    errors=torch.zeros(3,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    call(library,x,roots,out,errors,error,1)
    assert errors.eq(0).all() and out[:2].eq(0).all() and torch.equal(out[2],x[2])
    assert roots.tolist()==[True,False,True]


@pytest.mark.parametrize("stage,code",[(0,20),(1,23)])
def test_real_nonfinite_rejected_and_sticky_error(library,stage,code):
    x=torch.tensor([[float("nan"),0.]]);roots=torch.tensor([True]);out=torch.zeros_like(x)
    errors=torch.zeros(1,dtype=torch.int32);error=torch.zeros_like(errors)
    call(library,x,roots,out,errors,error,stage);assert errors.item()==code
    error.fill_(code);out.fill_(17);call(library,x,roots,out,errors,error,stage);assert out.eq(17).all()
