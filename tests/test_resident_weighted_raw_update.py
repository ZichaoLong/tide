"""Pinned raw KV/conv retention, distinct from effective memory arithmetic.

Independent per-field autograd and timestamp-list formulas; no graph scheduler.
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online"
    directory=tmp_path_factory.mktemp("raw-update");source=directory/"adapter.cpp";output=directory/"adapter.so"
    source.write_text('''#define TIDE_KERNEL_CPU
#include "weighted_raw_update_math.h"
using I=int64_t;
extern "C" void forward(void** p,I rows,I width,I cap,I count,I kv,I conv,I horizon){
 tide_weighted_raw_device::plan((I*)p[0],(I*)p[1],(float*)p[2],(I*)p[3],(float*)p[4],(uint8_t*)p[5],
   (float*)p[6],(I*)p[7],(I*)p[8],(float*)p[9],(uint8_t*)p[10],(int32_t*)p[11],rows,width,cap,count,kv,conv,horizon);
 tide_weighted_raw_device::apply((I*)p[0],(float*)p[2],(float*)p[6],(I*)p[8],(float*)p[9],(uint8_t*)p[10],
   (float*)p[12],(int32_t*)p[11],rows,width,cap,count,conv);
}
extern "C" void reverse(void** p,I rows,I width,I cap,I count,I conv){
 tide_weighted_raw_device::reverse((I*)p[0],(I*)p[1],(I*)p[2],(float*)p[3],(I*)p[4],(float*)p[5],
   (uint8_t*)p[6],(float*)p[7],(uint8_t*)p[8],(float*)p[9],(float*)p[10],(float*)p[11],
   (uint8_t*)p[12],(uint8_t*)p[13],(uint8_t*)p[14],(int32_t*)p[15],rows,width,cap,count,conv);
}
''')
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"),"-I"+str(Path(torch.__file__).parent/"include"),str(source),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output));lib.forward.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int64]*7
    lib.reverse.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int64]*5
    return lib


def pointers(ts):return (ctypes.c_void_p*len(ts))(*(t.data_ptr() for t in ts))


@dataclass
class Schema:
    family:str
    capacity:int=4
    horizon:int=4

    def __post_init__(self):
        self.kv=self.family=="kv";self.conv=self.family in ("gated_delta","ssd")
        if not self.kv and not self.conv:self.capacity=0
        specs=[(4,False,0)]
        specs+=([(3,True,1),(3,True,1),(2,True,1),(1,True,2)] if self.kv
                else [(12,False,1),(4,False,1),(2,False,3)]+([(7,True,1)] if self.conv else []))
        self.fields=[];self.width=0
        for size,ragged,role in specs:
            self.fields.append([self.width,size,int(ragged),role]);self.width+=size*(self.capacity if ragged else 1)
        self.table=torch.tensor(self.fields,dtype=torch.int64);self.count=len(specs)


def fixture(schema):
    s=schema;rows=7;clock=2**54+3;last=[-1,clock+2,3,2**63-2,0,4,2]
    stamps=[[],[clock,clock+2],[0,1,3],[2**63-4,2**63-2],[0],[1,2,3,4],[0,2]]
    meta=torch.zeros((rows,4+s.capacity),dtype=torch.int64)
    old=torch.full((rows,s.width),float("nan"));proposal=torch.full_like(old,float("nan"))
    for row in range(rows):
        n=len(stamps[row]) if s.capacity else 0
        meta[row,:4]=torch.tensor([last[row],0 if row==0 else 9,n,0 if s.kv else max(0,last[row])])
        if n:meta[row,4:4+n]=torch.tensor(stamps[row])
        for off,size,ragged,role in s.fields:
            count=n if ragged else 1
            x=torch.sin(torch.arange(count*size)*.13+row)*.2
            if role==0:x.zero_()
            if role in (2,3):x=x.abs()+.1
            old[row,off:off+count*size]=x
            if role not in (0,2):proposal[row,off:off+size]=torch.arange(size)*.03+.12+row*.01
    times=torch.tensor([0,clock+4,27,2**63-1,9,-1,9],dtype=torch.int64)
    mass=torch.tensor([.25,1.,0.,.3,-1.,float("nan"),1.5]);present=torch.tensor([1,1,1,1,1,0,1],dtype=torch.bool)
    return meta,old,times,mass,present,proposal


def forward(lib,s,inputs):
    meta,old,times,mass,present,proposal=inputs;rows=len(mass)
    t=[s.table,meta.clone(),old,times,mass,present,proposal.clone(),torch.full_like(meta,-42),
       torch.full((rows,max(1,s.capacity)),-42,dtype=torch.int64),torch.zeros(rows,2),torch.zeros(rows,dtype=torch.bool),
       torch.zeros(1,dtype=torch.int32),torch.full_like(old,42.)]
    def replay():lib.forward(pointers(t),rows,s.width,s.capacity,s.count,s.kv,s.conv,s.horizon)
    replay();return t,replay


def reference(s,inputs):
    meta,old,times,mass,present,proposal=inputs;outputs=[];leaves=[];new_meta=meta.clone()
    for row in range(len(mass)):
        n=meta[row,2].item();stamp=meta[row,4:4+n].tolist()
        write=bool(present[row]) and mass[row].item()>0;theta=times[row].item()
        keep=[i for i,t in enumerate(stamp) if 0<=theta-t and (s.horizon==0 or theta-t<s.horizon)] if s.kv else [i for i,t in enumerate(stamp) if theta-s.horizon<=t<theta]
        if write:
            new_stamps=[stamp[i] for i in keep]+[theta] if s.capacity else []
            new_meta[row,:4]=torch.tensor([theta,meta[row,1].item()+1,len(new_stamps),0 if s.kv else theta])
            if s.capacity:new_meta[row,4:]=0;new_meta[row,4:4+len(new_stamps)]=torch.tensor(new_stamps,dtype=torch.int64)
        m=mass[row].clone().requires_grad_();row_outputs=[];old_fields=[];new_fields=[]
        for off,size,ragged,role in s.fields:
            a=old[row,off:off+(n if ragged else 1)*size].clone().reshape(-1,size).requires_grad_()
            b=proposal[row,off:off+size].clone().requires_grad_();old_fields.append(a);new_fields.append(b)
            if not write or role==0:y=a
            elif not ragged:y=b.unsqueeze(0)
            else:
                item=m.clamp_max(1).reshape(1) if role==2 else b*(m.clamp_max(1) if s.conv else 1)
                y=torch.cat([a[keep],item.unsqueeze(0)])
            row_outputs.append(y)
        outputs.append(row_outputs);leaves.append((old_fields,new_fields,m))
    return outputs,leaves,new_meta


@pytest.mark.parametrize("family",["kv","affine","delta","gated_delta","ssd"])
def test_raw_retention_clocks_and_all_independent_field_vjps(library,family):
    s=Schema(family);inputs=fixture(s);t,_=forward(library,s,inputs);assert t[11].item()==0
    expected,leaves,metadata=reference(s,inputs);assert torch.equal(t[7],metadata)
    bar=torch.full_like(t[12],float("nan"));roots=torch.zeros((7,s.count),dtype=torch.bool)
    expected_grads=[]
    for row,fields in enumerate(expected):
        ys=[];bars=[]
        for j,(y,(off,_,_,_)) in enumerate(zip(fields,s.fields)):
            torch.testing.assert_close(t[12][row,off:off+y.numel()].reshape_as(y),y,rtol=0,atol=0)
            roots[row,j]=(row+j)%3!=0
            if roots[row,j]:
                b=torch.cos(torch.arange(y.numel())*.17).reshape_as(y)*(.0 if row==2 else .4)
                bar[row,off:off+y.numel()]=b.flatten();ys.append(y);bars.append(b)
        old_fields,new_fields,m=leaves[row]
        expected_grads.append(torch.autograd.grad(ys,old_fields+new_fields+[m],bars,allow_unused=True))
    # Saved forward values must suffice after the original metadata/input changes.
    inputs[0].fill_(-999);inputs[3].fill_(float("nan"));inputs[5].fill_(float("nan"))
    out=[torch.zeros_like(t[12]),torch.zeros_like(t[12]),torch.zeros(7),torch.zeros_like(roots),torch.zeros_like(roots),torch.zeros(7,dtype=torch.bool)]
    reverse=[t[0],t[1],t[7],t[6],t[8],t[9],t[10],bar,roots,*out,t[11]]
    library.reverse(pointers(reverse),7,s.width,s.capacity,s.count,s.conv);assert t[11].item()==0
    for row,grads in enumerate(expected_grads):
        for part in (0,1):
            for j,g in enumerate(grads[part*s.count:(part+1)*s.count]):
                assert out[3+part][row,j].item()==(g is not None)
                if g is not None:
                    off=s.fields[j][0];torch.testing.assert_close(out[part][row,off:off+g.numel()],g.flatten(),rtol=1e-6,atol=1e-7)
        assert out[5][row].item()==(grads[-1] is not None)
        if grads[-1] is not None:torch.testing.assert_close(out[2][row],grads[-1],rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize("family",["kv","gated_delta","ssd"])
def test_capacity_and_zero_receive_expired_tuple(library,family):
    s=Schema(family,horizon=0 if family=="kv" else 100);inputs=list(fixture(s))
    inputs[2][5]=5;inputs[3][5]=.5;inputs[4][5]=True
    t,replay=forward(library,s,inputs);assert t[11].item()==25 and t[12].eq(42).all()
    inputs[3][5]=0;t[11].zero_();replay();assert t[11].item()==0
    # Zero reception keeps all actual stale entries, not an effective projection.
    assert torch.equal(t[7][5],inputs[0][5]);torch.testing.assert_close(t[12][5],inputs[1][5],equal_nan=True,rtol=0,atol=0)


@pytest.mark.parametrize("bad",["duplicate","count","float","negative_weight","time","overflow_count"])
def test_malformed_raw_state_and_write_rejected(library,bad):
    s=Schema("kv");inputs=list(fixture(s))
    if bad=="duplicate":inputs[0][1,5]=inputs[0][1,4]
    elif bad=="count":inputs[0][1,1]=0
    elif bad=="float":inputs[1][1,s.fields[1][0]]=float("inf")
    elif bad=="negative_weight":inputs[1][1,s.fields[-1][0]]=-.1
    elif bad=="time":inputs[2][1]=inputs[0][1,0]
    else:inputs[0][1,1]=2**63-1
    t,replay=forward(library,s,inputs);assert t[11].item()==24 and t[12].eq(42).all()
    t[11].fill_(99);replay();assert t[11].item()==99
