"""KV original-time windows: exact int64 masks and independent raw field roots."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online";directory=tmp_path_factory.mktemp("kv-window")
    cpp,output=directory/"adapter.cpp",directory/"adapter.so"
    cpp.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_kv_window_math.h"
extern "C" void prepare(float* raw,int64_t* meta,int64_t* time,uint8_t* present,float* k,float* v,float* p,float* w,float* decay,int64_t* ages,int64_t* mapping,int64_t* lengths,uint8_t* selected,int32_t* error,int64_t rows,int64_t capacity,int64_t d,int64_t ds,int64_t horizon,float beta){
 tide_cuda::reset_arena();tide_weighted_window_device::prepare(raw,meta,time,present,k,v,p,w,decay,ages,mapping,lengths,selected,error,rows,capacity,d,ds,horizon,beta);
}
extern "C" void reverse(int64_t* mapping,int64_t* lengths,uint8_t* present,float* decay,float* k,float* v,float* p,float* w,uint8_t* roots,float* raw,uint8_t* connected,int32_t* error,int64_t rows,int64_t capacity,int64_t d,int64_t ds){
 tide_cuda::reset_arena();tide_weighted_window_device::reverse(mapping,lengths,present,decay,k,v,p,w,roots,raw,connected,error,rows,capacity,d,ds);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),"-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(cpp),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.prepare.argtypes=[ctypes.c_void_p]*14+[ctypes.c_int64]*5+[ctypes.c_float]
    lib.reverse.argtypes=[ctypes.c_void_p]*12+[ctypes.c_int64]*4;return lib


def fixture():
    c,d,ds=4,3,2;offsets=[d+1,d+1+c*d,d+1+2*c*d,d+1+2*c*d+c*ds];width=offsets[-1]+c
    big=2**54+3;last=2**63-1
    stamps=[[],[big,big+2],[0,2],[last-4,last-2],[0],[]]
    times=torch.tensor([0,big+5,30,last,0,-99]);present=torch.tensor([1,1,1,1,1,0],dtype=torch.bool)
    meta=torch.zeros(6,c+4,dtype=torch.int64);raw=torch.full((6,width),float("nan"))
    fields=[]
    for r,ts in enumerate(stamps):
        meta[r,:4]=torch.tensor([ts[-1] if ts else -1,len(ts),len(ts),0]);meta[r,4:4+len(ts)]=torch.tensor(ts,dtype=torch.int64)
        raw[r,:d+1]=0;row=[]
        for j,w in enumerate((d,d,ds,1)):
            x=torch.sin(torch.arange(len(ts)*w).float()*.2+r*.3).reshape(len(ts),w)*.3
            if j==3:x=x.abs()
            row.append(x.reshape(-1) if j==3 else x)
            raw[r,offsets[j]:offsets[j]+x.numel()]=x.flatten()
        fields.append(row)
    raw[5]=float("nan");meta[5]=-99
    return raw,meta,times,present,fields,stamps,offsets


def prepare(lib,horizon):
    raw,meta,times,present,fields,stamps,offsets=fixture();r,c,d,ds=6,4,3,2
    k,v,p,w,decay=[torch.zeros(shape) for shape in ((r,c,d),(r,c,d),(r,c,ds),(r,2,c),(r,2,c))]
    ages,mapping,lengths=[torch.zeros(shape,dtype=torch.int64) for shape in ((r,c),(r,c),(r,))]
    selected=torch.zeros(r,dtype=torch.bool);error=torch.zeros(1,dtype=torch.int32)
    args=(raw,meta,times,present,k,v,p,w,decay,ages,mapping,lengths,selected,error)
    lib.prepare(*(t.data_ptr() for t in args),r,c,d,ds,horizon,.2)
    return args,fields,stamps,offsets


@pytest.mark.parametrize("horizon",[0,3,5])
def test_exact_masks_and_full_raw_vjp(library,horizon):
    args,fields,stamps,offsets=prepare(library,horizon)
    raw,meta,times,present,k,v,p,w,decay,ages,mapping,lengths,selected,error=args
    assert error.item()==0 and torch.equal(selected,present)
    roots=torch.tensor([[1,0,1,0,1],[1,1,1,1,1],[0,1,1,1,0],[1,1,0,0,1],[1,1,1,1,1],[1,1,1,1,1]],dtype=torch.bool)
    bars=[torch.full_like(t,float("nan")) for t in (k,v,p,w)]
    expected=[]
    for r in range(6):
        leaves=[x.double().detach().requires_grad_() for x in fields[r]]
        if not present[r]:expected.append([None]*4);continue
        selected_ids=[i for i,t in enumerate(stamps[r]) if t<=int(times[r]) and (not horizon or int(times[r])-t<horizon)]
        assert lengths[r]==len(selected_ids) and mapping[r,:len(selected_ids)].tolist()==selected_ids
        selected_ages=[int(times[r])-stamps[r][i] for i in selected_ids]
        assert ages[r,:len(selected_ids)].tolist()==selected_ages
        index=torch.tensor(selected_ids,dtype=torch.int64);age=torch.tensor(selected_ages,dtype=torch.float32)
        dec=(-.2*age).exp().double()
        ys=[leaves[j][index] for j in range(3)]+[leaves[3][index]*dec,leaves[3][index]]
        outputs=[];cotangents=[]
        for j,y in enumerate(ys):
            actual=(k,v,p)[j][r,:len(index)] if j<3 else w[r,j-3,:len(index)]
            torch.testing.assert_close(actual.double(),y,rtol=2e-6,atol=2e-7)
            if roots[r,j]:
                g=torch.cos(torch.arange(y.numel()).float()*.3).reshape(y.shape)
                if r==2:g.zero_()
                (bars[j][r,:len(index)] if j<3 else bars[3][r,j-3,:len(index)]).copy_(g)
                outputs.append(y);cotangents.append(g.double())
        expected.append(torch.autograd.grad(outputs,leaves,cotangents,allow_unused=True))
    actual=torch.zeros_like(raw);connected=torch.zeros(6,5,dtype=torch.bool)
    library.reverse(*(t.data_ptr() for t in (mapping,lengths,selected,decay,*bars,roots,actual,connected,error)),6,4,3,2)
    assert not error.item() and not connected[:,0].any()
    for r in range(6):
        for j,g in enumerate(expected[r]):
            assert connected[r,j+1].item()==(g is not None)
            if g is not None:torch.testing.assert_close(actual[r,offsets[j]:offsets[j]+g.numel()].reshape_as(g).double(),g,rtol=2e-6,atol=2e-7)
    # An expired/all-zero numerical short channel still retains field identity.
    assert connected[2,4] and actual[2,offsets[3]:].eq(0).all()


def test_future_entries_and_invalid_metadata(library):
    args,_,_,_=prepare(library,0);raw,meta,times,present,k,v,p,w,decay,ages,mapping,lengths,selected,error=args
    times[1]=2**54+4
    library.prepare(*(t.data_ptr() for t in args),6,4,3,2,0,.2)
    assert not error.item() and lengths[1]==1 and ages[1,0]==1
    meta[1,5]=meta[1,4]
    library.prepare(*(t.data_ptr() for t in args),6,4,3,2,0,.2)
    assert error.item()==31


def test_raw_cache_exceeds_declared_window(library):
    args,_,_,_=prepare(library,1)
    assert args[-1].item()==31


def test_decay_underflow_retains_raw_visibility_and_connectivity(library):
    args,_,_,offsets=prepare(library,0);raw,meta,times,present,k,v,p,w,decay,ages,mapping,lengths,selected,error=args
    times[2]=2**54+3
    library.prepare(*(t.data_ptr() for t in args),6,4,3,2,0,.2)
    assert not error.item() and lengths[2]==2 and ages[2,:2].tolist()==[2**54+3,2**54+1]
    assert w[2,0].eq(0).all() and (w[2,1,:2]>0).all()
    bars=[torch.full_like(t,float("nan")) for t in (k,v,p,w)];bars[3][2,0,:2]=1
    roots=torch.zeros(6,5,dtype=torch.bool);roots[2,3]=True
    actual=torch.zeros_like(raw);connected=torch.zeros(6,5,dtype=torch.bool)
    library.reverse(*(t.data_ptr() for t in (mapping,lengths,selected,decay,*bars,roots,actual,connected,error)),6,4,3,2)
    assert not error.item() and connected[2,4] and actual[2,offsets[3]:].eq(0).all()
