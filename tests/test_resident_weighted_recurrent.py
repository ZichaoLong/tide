"""Pinned recurrent §1: independent age/attenuation and raw field cotangents.

Complete Upd/Mem/Full + shared parameters: weighted_recurrent_check.cpp.
"""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online"
    directory=tmp_path_factory.mktemp("recurrent")
    source,output=directory/"adapter.cpp",directory/"adapter.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_recurrent_snapshot_math.h"
extern "C" void prepare(float* raw,int64_t* meta,int64_t* time,float* mass,int64_t* owners,uint8_t* present,float* rates,float* clean,float* decay,int64_t* selected,int32_t* error,int64_t rows,int64_t width,int64_t capacity,int64_t d,int64_t matrices,int64_t mw,int64_t ds,int64_t cf,int64_t banks,int64_t positive,int64_t horizon){
 tide_cuda::reset_arena();tide_weighted_recurrent_device::prepare(raw,meta,time,mass,owners,present,rates,clean,decay,selected,error,rows,width,capacity,d,matrices,mw,ds,cf,banks,positive,horizon);}
extern "C" void reverse(int64_t* selected,int64_t* meta,float* decay,int64_t* fields,float* bars,uint8_t* roots,float* bypass,uint8_t* bypass_roots,float* output,uint8_t* connected,int32_t* error,int64_t rows,int64_t width,int64_t capacity,int64_t count){
 tide_cuda::reset_arena();tide_weighted_recurrent_device::reverse(selected,meta,decay,fields,bars,roots,bypass,bypass_roots,output,connected,error,rows,width,capacity,count);}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-I"+str(root/"cuda"),"-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),str(source),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.prepare.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int64]*11
    lib.reverse.argtypes=[ctypes.c_void_p]*11+[ctypes.c_int64]*4
    return lib


@pytest.mark.parametrize("positive",[False,True])
def test_exact_age_underflow_and_independent_identity_roots(library,positive):
    # value=2,H=2*2,Z=2*1,W=2,cache=3*2; no event-count compaction.
    rows,width,capacity=5,16,3
    raw=torch.arange(rows*width,dtype=torch.float32).reshape(rows,width)/50
    raw[:,:2]=0
    clock=2**54+3
    meta=torch.tensor([[-1,0,0,0,0,0,0],[clock,2,2,clock,clock-2,clock,0],
                       [2,2,2,2,0,2,0],[2,2,2,2,0,2,0],[-1,0,0,0,0,0,0]])
    time=torch.tensor([0,clock-1,2**63-1,9,-1]);mass=torch.tensor([0.,.2,1.,0.,float("nan")]);owners=torch.tensor([0,1,0,1,-1])
    present=torch.tensor([True,True,True,True,False]);rates=torch.tensor([.1,0.,.1,0.])
    clean=torch.full_like(raw,99);decay=torch.full_like(raw,99);selected=torch.zeros(rows,dtype=torch.int64);error=torch.zeros(1,dtype=torch.int32)
    library.prepare(*(x.data_ptr() for x in (raw,meta,time,mass,owners,present,rates,clean,decay,selected,error)),rows,width,capacity,2,2,2,1,2,2,positive,4)
    assert not error.item()
    expected=[]
    for row in range(rows):
        if not present[row] or (positive and mass[row]<=0):
            expected.append(torch.zeros(width));assert selected[row]==-1;continue
        age=max(0,int(time[row])-int(meta[row,3]));delta=torch.tensor(age,dtype=torch.int64).float()
        scale=(-rates*delta).exp();n=int(meta[row,2]);factors=torch.cat((torch.ones(2),scale[:2].repeat_interleave(2),scale[2:],scale[2:],torch.ones(n*2),torch.zeros((capacity-n)*2)))
        expected.append(raw[row]*factors)
    torch.testing.assert_close(clean*decay,torch.stack(expected))
    # An underflowed short matrix is still connected to the original H root.
    schema=torch.tensor([[0,2,0],[2,4,0],[6,2,0],[8,2,0],[10,2,1]])
    roots=torch.zeros(rows,5,dtype=torch.bool);roots[2,1]=True;roots[1,4]=True
    bypass_roots=torch.zeros_like(roots);bypass_roots[3,2]=True
    bar=torch.ones_like(raw);bar[~roots.any(1)]=float("nan")
    bypass=torch.full_like(raw,float("nan"));bypass[3,6:8]=torch.tensor([.3,.4])
    output=torch.full_like(raw,99);flags=torch.ones_like(roots)
    library.reverse(*(x.data_ptr() for x in (selected,meta,decay,schema,bar,roots,bypass,bypass_roots,output,flags,error)),rows,width,capacity,5)
    assert not error.item() and flags[2,1] and output[2,2:4].eq(0).all() and output[2,4:6].eq(1).all()
    torch.testing.assert_close(output[3,6:8],torch.tensor([.3,.4]))
    assert flags[3,2] and output[1,10:14].eq(1).all() and output[1,14:].eq(0).all()


def test_clock_metadata_rejected_before_numeric_projection(library):
    raw=torch.zeros(1,10);meta=torch.tensor([[2,1,0,1]]);time=torch.tensor([2]);mass=torch.ones(1);owners=torch.zeros(1,dtype=torch.int64)
    present=torch.ones(1,dtype=torch.bool);rates=torch.tensor([.1,0.,.1,0.]);clean=torch.zeros_like(raw);decay=torch.zeros_like(raw);selected=torch.zeros_like(owners);error=torch.zeros(1,dtype=torch.int32)
    library.prepare(*(x.data_ptr() for x in (raw,meta,time,mass,owners,present,rates,clean,decay,selected,error)),1,10,0,2,2,2,1,0,1,False,0)
    assert error.item()==38
