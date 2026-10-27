"""Original-time sparse taps, vendor-expression reference, independent full VJP."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("weighted-convolution")
    cpp=directory/"adapter.cpp";output=directory/"adapter.so"
    cpp.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_convolution_math.h"
using I=int64_t;
extern "C" void plan(void** t,I rows,I width,I cap,I off,I features,I size,I stride,I banks){
 tide_weighted_conv_device::prepare((float*)t[0],(I*)t[1],(I*)t[2],(float*)t[3],(I*)t[4],(I*)t[5],
 (I*)t[6],(float*)t[7],(int32_t*)t[8],rows,width,cap,off,features,size,stride,banks);
}
extern "C" void reverse(void** t,I rows,I width,I cap,I off,I features,I size,I banks){
 tide_weighted_conv_device::reverse((I*)t[0],(I*)t[1],(I*)t[2],(float*)t[3],(float*)t[4],(float*)t[5],
 (uint8_t*)t[6],(float*)t[7],(float*)t[8],(uint8_t*)t[9],(float*)t[10],(float*)t[11],(uint8_t*)t[12],
 (int32_t*)t[13],rows,width,cap,off,features,size,banks);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(cpp),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.plan.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int64]*8
    lib.reverse.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int64]*7
    return lib


def pointers(ts):return (ctypes.c_void_p*len(ts))(*(t.data_ptr() for t in ts))


def plan(lib,stamps,cache,current,times,owners,size,stride,banks):
    rows,cap,features=cache.shape;meta=torch.zeros(rows,cap+4,dtype=torch.int64)
    for row,ts in enumerate(stamps):meta[row,2]=len(ts);meta[row,4:4+len(ts)]=torch.tensor(ts,dtype=torch.int64)
    old=cache.reshape(rows,-1);indices=torch.zeros(rows,dtype=torch.int64);mapping=torch.zeros(rows,size,dtype=torch.int64)
    taps=torch.zeros(rows,size,features);error=torch.zeros(1,dtype=torch.int32)
    ts=[old,meta,times,current,owners,indices,mapping,taps,error]
    lib.plan(pointers(ts),rows,cap*features,cap,0,features,size,stride,banks)
    return ts


def test_textbook_weak_slot_and_current_not_cache(library):
    for epsilon in (0.,.001,.5,1.):
        # Stored current time may already contain w*z. Tap zero must still read z.
        cache=torch.tensor([[[4.],[5*epsilon],[.3*6],[float("nan")]]])
        ts=plan(library,[[0,2,4]],cache,torch.tensor([[6.]]),torch.tensor([4]),torch.tensor([0]),3,2,1)
        assert ts[-1].item()==0
        actual=(ts[7]*torch.tensor([1.,2.,3.]).reshape(1,3,1)).sum()
        torch.testing.assert_close(actual,torch.tensor(18.+10*epsilon))
        assert ts[6].tolist()==[[-1,1,0]]


@pytest.mark.parametrize("size",[1,2,4])
@pytest.mark.parametrize("stride",[1,3])
def test_grouped_dense_convolution_and_independent_shared_vjp(library,size,stride):
    rows=6;features=3;cap=5;banks=3;base=2**54+11
    stamps=[[],[base,base+1,base+3],[0,1,3,4],[2**63-5,2**63-2],[0,1,3],[0,2]]
    stamps=[ts[-min(cap,(size-1)*stride+1):] for ts in stamps]
    times=torch.tensor([0,base+4,9,2**63-1,-1,4]);owners=torch.tensor([0,1,0,1,-1,1])
    generator=torch.Generator().manual_seed(72);cache=torch.randn(rows,cap,features,generator=generator)*.2
    for row,ts in enumerate(stamps):cache[row,len(ts):]=float("nan")
    current=torch.randn(rows,features,generator=generator)*.2;current[4]=float("nan")
    weight=torch.randn(banks,size,features,generator=generator)*.2;bias=torch.randn(banks,features,generator=generator)*.2
    t=plan(library,stamps,cache,current,times,owners,size,stride,banks);assert not t[-1].item()
    rw=weight[t[5]];rb=bias[t[5]];rw[owners<0]=0;rb[owners<0]=0
    actual=(t[7]*rw).sum(1)+rb
    leaves_cache=[cache[row,:len(ts)].clone().requires_grad_() for row,ts in enumerate(stamps)]
    leaves_current=[current[row].clone().requires_grad_() for row in range(rows)]
    weights=[w.clone().requires_grad_() for w in weight];biases=[b.clone().requires_grad_() for b in bias]
    outputs=[];roots=torch.tensor([1,1,0,1,1,1],dtype=torch.bool)
    cotangents=torch.randn(rows,features,generator=generator);cotangents[3]=0;cotangents[4]=float("nan")
    selected=[];bars=[]
    for row,owner in enumerate(owners.tolist()):
        if owner<0:assert actual[row].eq(0).all();continue
        theta=times[row].item();history=leaves_cache[row]
        if size==1 or not len(stamps[row]):
            y=biases[owner]+weights[owner][0]*leaves_current[row]
        else:
            # Build a dense logical-time signal, then use independent grouped conv1d.
            horizon=(size-1)*stride;coordinates=range(theta-horizon,theta)
            matches=torch.tensor([[float(t==s) for s in stamps[row]] for t in coordinates])
            past=matches@history;sequence=torch.cat([past,leaves_current[row].unsqueeze(0)],dim=0)
            y=F.conv1d(sequence.T.unsqueeze(0),weights[owner].flip(0).T.unsqueeze(1),
                       biases[owner],dilation=stride,groups=features).flatten()
        torch.testing.assert_close(actual[row],y,rtol=1e-6,atol=1e-7)
        if roots[row]:selected.append(y);bars.append(cotangents[row])
    expected=torch.autograd.grad(selected,leaves_cache+leaves_current+weights+biases,bars,allow_unused=True)
    flags=roots&(owners>=0);g=cotangents.clone();g[~flags]=0
    dt=g[:,None,:]*rw;dw=g[:,None,:]*t[7]
    out=[torch.zeros_like(t[0]),torch.zeros_like(current),torch.zeros_like(flags),torch.zeros_like(weight),torch.zeros_like(bias),torch.zeros(banks,dtype=torch.bool)]
    tensors=[owners,t[1],t[6],dt,dw,g,flags,*out,t[-1]]
    library.reverse(pointers(tensors),rows,cap*features,cap,0,features,size,banks);assert not t[-1].item()
    for row in range(rows):
        gc,gz=expected[row],expected[rows+row]
        assert out[2][row].item()==(gc is not None);assert flags[row].item()==(gz is not None)
        if gc is not None:torch.testing.assert_close(out[0][row,:gc.numel()].reshape_as(gc),gc,rtol=1e-6,atol=1e-7)
        if gz is not None:torch.testing.assert_close(out[1][row],gz,rtol=1e-6,atol=1e-7)
    for bank in range(banks):
        gw,gb=expected[2*rows+bank],expected[2*rows+banks+bank]
        assert out[5][bank].item()==(gw is not None)==(gb is not None)
        if gw is not None:torch.testing.assert_close(out[3][bank],gw,rtol=1e-6,atol=1e-7);torch.testing.assert_close(out[4][bank],gb,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize("case",["owner","time","order","nonfinite"])
def test_invalid_cache_query_is_explicit(library,case):
    cache=torch.ones(1,3,2);stamps=[[0,2]];time=torch.tensor([4]);owner=torch.tensor([0]);current=torch.ones(1,2)
    if case=="owner":owner[0]=3
    elif case=="time":time[0]=-1
    elif case=="order":stamps=[[2,0]]
    else:cache[0,1,0]=float("nan")
    t=plan(library,stamps,cache,current,time,owner,3,2,1);assert t[-1].item()==27


def test_missing_tap_temporary_overflow_does_not_invent_cache_gradient(library):
    # y=0*z+huge*absent+b has finite derivatives, although bar*huge overflows.
    cache=torch.full((1,2,1),float("nan"));t=plan(library,[[]],cache,torch.zeros(1,1),torch.tensor([0]),torch.tensor([0]),2,2,1)
    dt=torch.tensor([[[0.],[float("inf")]]]);dw=torch.zeros(1,2,1);g=torch.full((1,1),1e20);roots=torch.ones(1,dtype=torch.bool)
    out=[torch.zeros_like(t[0]),torch.zeros(1,1),torch.zeros_like(roots),torch.zeros(1,2,1),torch.zeros(1,1),torch.zeros_like(roots)]
    ts=[t[4],t[1],t[6],dt,dw,g,roots,*out,t[-1]]
    library.reverse(pointers(ts),1,2,2,0,1,2,1)
    assert t[-1].item()==0 and not out[2].any() and out[-1].all()
    assert out[0].eq(0).all() and out[1].eq(0).all() and out[3].eq(0).all()
    torch.testing.assert_close(out[4],g)
