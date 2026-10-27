"""KV §2 relative rotations, exact-age phase conversion and saved linear VJP."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("rotary")
    cpp,output=directory/"adapter.cpp",directory/"adapter.so"
    cpp.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_rotary_math.h"
#include "weighted_rotary_reduce.h"
extern "C" void reduce(float* input,float* output,int32_t* quadrant,int32_t* error,int64_t count){
 tide_weighted_rotary_device::reduce(input,output,quadrant,error,count);
}
extern "C" void unfold(float* cosine,float* sine,int32_t* quadrant,int32_t* error,int64_t count){
 tide_weighted_rotary_device::unfold(cosine,sine,quadrant,error,count);
}
extern "C" void numeric_time(int64_t* input,float* output,int64_t count){
 for(int64_t i=0;i<count;++i)output[i]=tide_weighted_time_device::as_float(input[i]);
}
extern "C" void phase(int64_t* ages,float* f,float* out,int32_t* error,int64_t count,int64_t pairs,float scale){
 tide_cuda::reset_arena();tide_weighted_rotary_device::phase(ages,f,out,error,count,pairs,scale);
}
extern "C" void apply(float* x,float* c,float* s,uint8_t* roots,float* out,int32_t* error,int64_t rows,int64_t capacity,int64_t width,int64_t pairs,int64_t reverse){
 tide_cuda::reset_arena();tide_weighted_rotary_device::apply(x,c,s,roots,out,error,rows,capacity,width,pairs,reverse);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),"-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(cpp),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.reduce.argtypes=lib.unfold.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64];lib.numeric_time.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int64];lib.phase.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64]*2+[ctypes.c_float]
    lib.apply.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int64]*5;return lib


def phases(lib,pairs):
    ages=torch.tensor([[0,1,3],[2**24+1,2**54+3,2**63-1]],dtype=torch.int64)
    f=torch.tensor([.13,-.07][:pairs]);out=torch.zeros(2,3,max(1,pairs));error=torch.zeros(1,dtype=torch.int32)
    if pairs:
        lib.phase(*(t.data_ptr() for t in (ages,f,out,error)),6,pairs,7.)
        torch.testing.assert_close(out,-ages.float().unsqueeze(-1)/7*f,rtol=0,atol=0)
    assert not error.item();return out.cos(),out.sin(),error


@pytest.mark.parametrize("pairs",[0,1,2])
def test_dense_rotation_and_saved_inverse(library,pairs):
    cosine,sine,error=phases(library,pairs)
    x=torch.sin(torch.arange(30).float()*.3).reshape(2,3,5);bar=torch.cos(torch.arange(30).float()*.7).reshape_as(x)
    output=torch.zeros_like(x);roots=torch.tensor([True,False]);dx=torch.zeros_like(x)
    library.apply(*(t.data_ptr() for t in (x,cosine,sine,roots,output,error)),2,3,5,pairs,False)
    # Independent dense matrices; all leading identity coordinates retained.
    matrix=torch.eye(5,dtype=torch.double).expand(2,3,5,5).clone()
    for j in range(pairs):
        a=5-2*pairs+2*j;c=cosine[:,:,j].double();s=sine[:,:,j].double()
        matrix[:,:,a,a]=c;matrix[:,:,a,a+1]=-s;matrix[:,:,a+1,a]=s;matrix[:,:,a+1,a+1]=c
    leaf=x.double().requires_grad_();reference=(matrix@leaf.unsqueeze(-1)).squeeze(-1)
    expected,=torch.autograd.grad(reference,leaf,bar.double())
    torch.testing.assert_close(output.double(),reference,rtol=1e-6,atol=1e-7)
    bar[1]=float("nan");library.apply(*(t.data_ptr() for t in (bar,cosine,sine,roots,dx,error)),2,3,5,pairs,True)
    assert not error.item() and dx[1].eq(0).all();torch.testing.assert_close(dx[0].double(),expected[0],rtol=1e-6,atol=1e-7)
    if pairs==0:assert torch.equal(output,x)


def test_invalid_ages_and_phase_overflow(library):
    age=torch.tensor([-1],dtype=torch.int64);f=torch.ones(1);out=torch.zeros(1);error=torch.zeros(1,dtype=torch.int32)
    library.phase(*(t.data_ptr() for t in (age,f,out,error)),1,1,1.);assert error.item()==33
    error.zero_();age[0]=2**63-1;f[0]=torch.finfo(f.dtype).max
    library.phase(*(t.data_ptr() for t in (age,f,out,error)),1,1,1.);assert error.item()==33


def test_int64_to_fp32_rounding_boundaries(library):
    samples={0,1,2**63-1}
    for exponent in range(24,63):
        step=2**(exponent-23)
        for low in (0,1,2,3,2**23-2):
            base=2**exponent+low*step
            for extra in (-1,0,1,step//2-1,step//2,step//2+1,step-1):
                value=base+extra
                if 0<=value<2**63:samples.add(value)
    values=torch.tensor(sorted(samples),dtype=torch.int64);actual=torch.zeros(len(samples))
    library.numeric_time(values.data_ptr(),actual.data_ptr(),len(samples))
    assert torch.equal(actual,values.float())


def test_full_finite_fp32_argument_reduction(library):
    import math
    values=[0.,-0.,math.pi/4,math.pi/2,math.pi,65536.,1e14,1e17]
    for exponent in range(1,255):
        for mantissa in (0,1,0x234567,0x7fffff):
            bits=torch.tensor([(exponent<<23)|mantissa],dtype=torch.int32)
            values.extend((float(bits.view(torch.float32)[0]),-float(bits.view(torch.float32)[0])))
    x=torch.tensor(values);reduced=torch.zeros_like(x);quadrant=torch.zeros(x.numel(),dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    library.reduce(*(t.data_ptr() for t in (x,reduced,quadrant,error)),x.numel())
    assert not error.item() and reduced.abs().le(math.pi/4+1e-7).all()
    cosine,sine=reduced.cos(),reduced.sin()
    library.unfold(*(t.data_ptr() for t in (cosine,sine,quadrant,error)),x.numel())
    expected_c=torch.tensor([math.cos(float(v)) for v in x]);expected_s=torch.tensor([math.sin(float(v)) for v in x])
    torch.testing.assert_close(cosine,expected_c,rtol=0,atol=2e-7)
    torch.testing.assert_close(sine,expected_s,rtol=0,atol=2e-7)


def test_nonfinite_reduction_refused(library):
    x=torch.tensor([float("inf")]);out=torch.zeros_like(x);q=torch.zeros(1,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    library.reduce(*(t.data_ptr() for t in (x,out,q,error)),1);assert error.item()==33
