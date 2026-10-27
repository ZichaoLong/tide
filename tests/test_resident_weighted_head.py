"""0770593 ScalarHead/Query: owner selection and complete shared adjoints.

These exercise registered metadata/reduction kernels on CPU; vendor bmm/SiLU and
their backward are independently checked by weighted_head_check on real devices.
"""
import ctypes
from pathlib import Path
import subprocess
import sys
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    device=Path(__file__).resolve().parents[1]/"tools/device_online"
    directory=tmp_path_factory.mktemp("resident-head");generated=directory/"generated"
    subprocess.run([sys.executable,str(device/"cuda/generate_kernels.py"),"--source",str(device/"ascendc"),
                    "--output",str(generated),"--backend","cpu"],check=True)
    names=("tide_weighted_head_plan","tide_weighted_head_guard","tide_weighted_head_reduce")
    path=directory/"head.so"
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),
                    *(str(generated/(n+".cpp")) for n in names),"-o",str(path)],check=True)
    lib=ctypes.CDLL(str(path));prefix=[ctypes.c_uint32,ctypes.c_void_p]
    for name,pointers,integers in zip(names,(6,11,14),(3,6,7)):
        getattr(lib,"tide_cuda_launch_"+name).argtypes=prefix+[ctypes.c_void_p]*pointers+[ctypes.c_int64]*integers
    return lib


def call(lib,name,tensors,*integers):
    assert getattr(lib,"tide_cuda_launch_tide_weighted_head_"+name)(1,None,*(t.data_ptr() for t in tensors),*integers)==0


def test_owner_plan_preserves_private_rows_and_ignores_absence(library):
    x=torch.tensor([[.3,.2],[float("nan"),float("nan")],[.7,-.2],[.5,.1]])
    ids=torch.tensor([1,-1,1,0]);clean=torch.full_like(x,42.);safe=torch.full_like(ids,99)
    errors=torch.full((4,),42,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    call(library,"plan",[x,ids,clean,safe,errors,error],4,2,2)
    assert errors.eq(0).all() and safe.tolist()==[1,0,1,0]
    assert clean[1].eq(0).all() and torch.equal(clean[0],x[0]) and torch.equal(clean[2],x[2])
    assert not torch.equal(clean[0],clean[2]) # Parameter owner sharing never merges input rows.


@pytest.mark.parametrize("owner",[-2,2,2**60])
def test_owner_plan_rejects_before_vendor_gather(library,owner):
    x=torch.ones(1,2);ids=torch.tensor([owner]);clean=torch.empty_like(x);safe=torch.empty_like(ids)
    errors=torch.zeros(1,dtype=torch.int32);error=torch.zeros_like(errors)
    call(library,"plan",[x,ids,clean,safe,errors,error],1,2,2)
    assert errors.item()==2 and safe.item()==0 and clean.eq(0).all()


def test_forward_guard_masks_inactive_parameter_banks(library):
    ids=torch.tensor([-1,0]);values=[torch.tensor([[float("nan")]*n,[.2]*n]) for n in (6,3,6,2)]
    errors=torch.zeros(2,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    call(library,"guard",[ids,*values,error,error,error,error,errors,error],2,6,3,6,2,0)
    assert errors.eq(0).all() and all(v[0].eq(0).all() for v in values)
    values[2][1,0]=float("inf")
    call(library,"guard",[ids,*values,error,error,error,error,errors,error],2,6,3,6,2,0)
    assert errors.tolist()==[0,20]


def test_reverse_guard_keeps_connected_zero_separate(library):
    ids=torch.tensor([0,-1,0,1]);roots=torch.tensor([True,True,False,True])
    g=torch.tensor([[0.,0.],[float("nan")]*2,[float("nan")]*2,[.3,.4]])
    clean=torch.full_like(g,42.);connected=torch.zeros(4,dtype=torch.bool)
    errors=torch.zeros(4,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    call(library,"guard",[ids,g,g,g,g,roots,g,clean,connected,errors,error],4,2,2,2,2,2)
    assert errors.eq(0).all() and connected.tolist()==[True,False,False,True]
    assert clean[:3].eq(0).all() and torch.equal(clean[3],g[3])
    g[0,0]=float("nan")
    call(library,"guard",[ids,g,g,g,g,roots,g,clean,connected,errors,error],4,2,2,2,2,2)
    assert errors[0]==23


def test_shared_owners_reduce_every_root_with_absent_banks(library):
    ids=torch.tensor([1,0,1,-1,0,2]);roots=torch.tensor([True,True,True,False,False,True])
    dx=torch.tensor([[.1,.2],[.3,.4],[.5,.6],[float("nan")]*2,[float("nan")]*2,[0.,0.]])
    row=[torch.arange(6*n,dtype=torch.float32).reshape(6,n)*.1 for n in (6,3,6,2)]
    for value in row:value[3:5]=float("nan");value[5]=0
    bank=[torch.full((4,n),42.) for n in (6,3,6,2)]
    connected=torch.zeros(4,dtype=torch.bool);errors=torch.zeros(4,dtype=torch.int32);error=torch.zeros(1,dtype=torch.int32)
    call(library,"reduce",[ids,roots,dx,*row,*bank,connected,errors,error],6,4,2,6,3,6,2)
    assert errors.eq(0).all() and connected.tolist()==[True,True,True,False]
    assert dx[3:].eq(0).all()
    for src,dst in zip(row,bank):
        torch.testing.assert_close(dst[0],src[1]);torch.testing.assert_close(dst[1],src[0]+src[2])
        assert dst[2:].eq(0).all()


def test_shared_gradient_overflow_refused_and_sticky_error(library):
    ids=torch.tensor([0,0]);roots=torch.ones(2,dtype=torch.bool);dx=torch.ones(2,1)
    row=[torch.full((2,1),3e38) for _ in range(4)];bank=[torch.zeros(1,1) for _ in range(4)]
    connected=torch.zeros(1,dtype=torch.bool);errors=torch.zeros(1,dtype=torch.int32);error=torch.zeros_like(errors)
    call(library,"reduce",[ids,roots,dx,*row,*bank,connected,errors,error],2,1,1,1,1,1,1)
    assert errors.item()==23
    error.fill_(23)
    for b in bank:b.fill_(7)
    call(library,"reduce",[ids,roots,dx,*row,*bank,connected,errors,error],2,1,1,1,1,1,1)
    assert all(b.item()==7 for b in bank)
