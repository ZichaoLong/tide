"""Saved-coordinate RMS contraction against independent FP64 formulas/differences."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("projection-norm")
    cpp=directory/"adapter.cpp";output=directory/"adapter.so"
    cpp.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_projection_norm.h"
extern "C" void run(float* n,float* r,float* g,uint8_t* roots,int32_t* error,int64_t rows,int64_t width){
 tide_weighted_projection_device::reverse_norm(n,r,g,roots,error,rows,width);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(cpp),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int64]*2;return lib


@pytest.mark.parametrize("width",[1,3,127])
def test_saved_rms_vjp_and_joint_direction(library,width):
    x=torch.sin(torch.arange(4*width,dtype=torch.float64)).reshape(4,width);x[0].zero_();x.requires_grad_()
    epsilon=.13;y=x/(x.square().mean(-1,keepdim=True)+epsilon).sqrt();g=torch.cos(torch.arange(x.numel(),dtype=x.dtype)).reshape_as(x)
    expected,=torch.autograd.grad(y,x,g);normal=y.detach().float();rstd=(x.detach().square().mean(-1,keepdim=True)+epsilon).rsqrt().float()
    actual=g.float();roots=torch.tensor([1,1,0,1],dtype=torch.bool);error=torch.zeros(1,dtype=torch.int32)
    actual[2]=float("nan");normal[2]=float("nan");rstd[2]=float("nan")
    library.run(*(t.data_ptr() for t in (normal,rstd,actual,roots,error)),4,width)
    assert error.item()==0 and actual[2].eq(0).all();torch.testing.assert_close(actual[roots],expected[roots].float(),rtol=2e-5,atol=2e-6)
    direction=torch.sin(torch.arange(x.numel(),dtype=x.dtype)*.23).reshape_as(x);direction[2]=0;h=1e-5
    def loss(z):return ((z/(z.square().mean(-1,keepdim=True)+epsilon).sqrt())*g)[roots].sum()
    finite_difference=(loss(x.detach()+h*direction)-loss(x.detach()-h*direction))/(2*h)
    torch.testing.assert_close((actual.double()*direction).sum(),finite_difference,rtol=2e-5,atol=2e-6)


def test_nonfinite_and_sticky_error(library):
    normal=torch.ones(1,3);rstd=torch.ones(1,1);g=torch.full((1,3),float("inf"));root=torch.ones(1,dtype=torch.bool);error=torch.zeros(1,dtype=torch.int32)
    library.run(*(t.data_ptr() for t in (normal,rstd,g,root,error)),1,3);assert error.item()==23
    error.fill_(41);g.fill_(8);library.run(*(t.data_ptr() for t in (normal,rstd,g,root,error)),1,3);assert error.item()==41 and g.eq(8).all()
