"""Allocation §3/5.1 scalar chain and independent structural row roots.

Complete heads/fields/parameter VJPs: weighted_describe_check.cpp.
"""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    directory = tmp_path_factory.mktemp("describe")
    source, output = directory/"adapter.cpp", directory/"adapter.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_row_graph_math.h"
extern "C" void bind(int64_t* owner,float* input,float* output,int32_t* error,int64_t rows,int64_t width,int64_t banks,int64_t positive){
 tide_cuda::reset_arena();tide_weighted_row_device::bind(owner,input,output,error,rows,width,banks,positive);
}
extern "C" void accumulate(int64_t* owner,float* input,uint8_t* roots,float* output,uint8_t* connected,int32_t* error,int64_t rows,int64_t width){
 tide_cuda::reset_arena();tide_weighted_row_device::accumulate(owner,input,roots,output,connected,error,rows,width);
}
extern "C" void derivative(float* input,float* value,float* gradient,uint8_t* roots,float* output,int32_t* error,int64_t rows,int64_t width,int64_t mode,float scalar){
 tide_cuda::reset_arena();tide_weighted_row_device::derivative(input,value,gradient,roots,output,error,rows,width,mode,scalar);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(source),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output))
    lib.bind.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]*4
    lib.accumulate.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*2
    lib.derivative.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*3+[ctypes.c_float]
    return lib


@pytest.mark.parametrize("mode", [0,1,2,3,4,5])
def test_declared_scalar_derivative_including_cap_equality(library,mode):
    x=torch.tensor([[.1,.2,1.,21.], [.3,.5,.9,1.2]],requires_grad=True)
    value=(x.sigmoid() if mode==0 else x.log1p() if mode==1 else x.clamp_max(1)
           if mode==2 else x.exp() if mode==3 else x.sqrt() if mode==4
           else torch.nn.functional.softplus(x))
    bar=torch.tensor([[.2,-.4,.6,.7], [.1,.2,.3,.4]])
    expected,=torch.autograd.grad(value,x,bar)
    flags=torch.tensor([True,False]);bar[1]=float("nan")
    result=torch.full_like(x,77);error=torch.zeros(1,dtype=torch.int32)
    library.derivative(*(v.data_ptr() for v in (x.detach(),value.detach(),bar,flags,result,error)),2,4,mode,1.)
    assert not error.item() and result[1].eq(0).all()
    torch.testing.assert_close(result[0],expected[0],rtol=1e-6,atol=1e-7)


def test_shared_contributions_keep_zero_connection_and_ignore_absent_poison(library):
    owner=torch.tensor([0,1,-1]);first=torch.tensor([[1.,2.],[0.,0.],[float("nan")]*2])
    second=torch.tensor([[-1.,-2.],[float("nan")]*2,[float("nan")]*2])
    out=torch.zeros(3,2);connected=torch.zeros(3,dtype=torch.bool);error=torch.zeros(1,dtype=torch.int32)
    for value,roots in ((first,torch.tensor([True,True,True])),(second,torch.tensor([True,False,True]))):
        library.accumulate(*(t.data_ptr() for t in (owner,value,roots,out,connected,error)),3,2)
    assert not error.item() and out.eq(0).all() and connected.tolist()==[True,True,False]


def test_nonnegative_statistics_and_missing_owner_binding(library):
    owner=torch.tensor([1,-1]);x=torch.tensor([[.2,0.],[float("nan")]*2]);out=torch.zeros_like(x);error=torch.zeros(1,dtype=torch.int32)
    library.bind(*(t.data_ptr() for t in (owner,x,out,error)),2,2,2,True)
    assert not error.item() and out[1].eq(0).all()
    x[0,0]=-.1
    library.bind(*(t.data_ptr() for t in (owner,x,out,error)),2,2,2,True)
    assert error.item()==36
