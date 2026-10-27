"""0770593 SelStep metadata: registered entry points, absence and exact clocks.

CPU compilation validates kernel indexing/control; real vendor operators and
complete composed reverse are covered by weighted_region_check on the target.
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
    directory=tmp_path_factory.mktemp("region-kernels");generated=directory/"generated"
    subprocess.run([sys.executable,str(device/"cuda/generate_kernels.py"),"--source",str(device/"ascendc"),
                    "--output",str(generated),"--backend","cpu"],check=True)
    shapes={"prepare":(17,7,0),"finish":(11,7,5),"result":(7,2,0),
            "reverse_finish":(19,8,4),"reverse_prepare":(18,5,0)}
    path=directory/"region.so"
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off",
                    "-I"+str(device/"cuda"),"-I"+str(device/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),
                    *(str(generated/("tide_weighted_region_"+n+".cpp")) for n in shapes),"-o",str(path)],check=True)
    lib=ctypes.CDLL(str(path))
    for name,(p,i,f) in shapes.items():
        getattr(lib,"tide_cuda_launch_tide_weighted_region_"+name).argtypes=(
            [ctypes.c_uint32,ctypes.c_void_p]+[ctypes.c_void_p]*p+[ctypes.c_int64]*i+[ctypes.c_float]*f)
    return lib


def call(lib,name,tensors,*scalars):
    assert getattr(lib,"tide_cuda_launch_tide_weighted_region_"+name)(
        1,None,*(t.data_ptr() for t in tensors),*scalars)==0


def inputs():
    w=torch.tensor([[.2,.3,0.],[float("nan")]*3,[.1,0.,float("nan")]])
    mu=torch.tensor([[.4,.6,0.],[float("nan")]*3,[.3,.2,float("nan")]])
    score=torch.zeros(3,3);e=torch.arange(18).float().reshape(3,3,2)*.1;x=e.flip(-1).contiguous()
    ids=torch.tensor([[2**60+7,2**60+2,2**60+4],[-42]*3,[4,2,-42]])
    times=torch.tensor([2**60+1,-42,2**60+11]);lengths=torch.tensor([3,0,2]);owners=torch.tensor([0,42,1])
    ctx=torch.full((3,4),42.);qry=torch.full((3,2),42.);stats=torch.full((3,3),42.)
    cr=torch.full((3,),42,dtype=torch.int64);qr=cr.clone();errors=torch.zeros(3,dtype=torch.int32)
    error=torch.zeros(1,dtype=torch.int32)
    return [w,mu,score,e,x,ids,times,lengths,owners,owners.clone(),ctx,qry,stats,cr,qr,errors,error]


def test_complete_reductions_and_absence(library):
    t=inputs();call(library,"prepare",t,3,3,2,2,1,1,1)
    assert t[15].eq(0).all() and t[13].tolist()==[0,-1,1] and torch.equal(t[13],t[14])
    for row in (0,2):
        n=t[7][row];w=t[0][row,:n];mu=t[1][row,:n]
        expected=torch.cat(((w[:,None]*t[3][row,:n]).sum(0)/w.sum().clamp_min(1),
                            w.sum().log1p()[None],mu.sum().log1p()[None]))
        torch.testing.assert_close(t[10][row],expected)
        torch.testing.assert_close(t[11][row],(w[:,None]*t[4][row,:n]).sum(0)/w.sum().clamp_min(1))
    assert all(t[i][1].eq(0).all() for i in (10,11,12))
    # Replaying with a changed complete length resets every prior accumulation.
    t[7][0]=1;call(library,"prepare",t,3,3,2,2,1,1,1)
    torch.testing.assert_close(t[10][0,:2],t[0][0,0]*t[3][0,0])


@pytest.mark.parametrize("fault",["id","time","length","owner","mass","evidence"])
def test_invalid_rows_and_sticky_error(library,fault):
    t=inputs()
    if fault=="id":t[5][0,1]=t[5][0,0]
    elif fault=="time":t[6][0]=-1
    elif fault=="length":t[7][0]=4
    elif fault=="owner":t[8][0]=-1
    elif fault=="mass":t[0][0,0]=.9
    else:t[3][0,0,0]=float("nan")
    call(library,"prepare",t,3,3,2,2,1,1,1)
    assert t[15][0].item()==(20 if fault in ("mass","evidence") else 2)
    t[16].fill_(23);t[10].fill_(17)
    call(library,"prepare",t,3,3,2,2,1,1,1)
    assert t[10].eq(17).all()


def test_actual_positive_support_and_raw_time(library):
    lengths=torch.tensor([3,0,2]);times=torch.tensor([2**60+1,-42,2**60+11])
    c=torch.tensor([[.5,0.,1e-35],[float("nan")]*3,[0.,.25,float("nan")]])
    active=torch.ones(3,3,dtype=torch.bool);history=torch.full_like(times,42)
    present=torch.ones(3,dtype=torch.bool);error=torch.zeros(1,dtype=torch.int32)
    call(library,"result",[lengths,times,c,active,history,present,error],3,3)
    assert active.tolist()==[[True,False,True],[False]*3,[False,True,False]]
    assert history.tolist()==[2**60+1,-1,2**60+11] and present.tolist()==[True,False,True]
