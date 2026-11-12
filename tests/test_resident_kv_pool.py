"""Shared-value bank pooling against independent FP64 formulas and derivatives."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def pool_kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]/"tools/device_online"
    work = tmp_path_factory.mktemp("kv_shared_pool")
    source, library = work/"adapter.cpp", work/"adapter.so"
    names = ("prepare", "center", "scale", "roots", "finish")
    source.write_text('#define TIDE_KERNEL_CPU\n'+''.join(
        f'#include "tide_block_kv_pool_{name}.cpp"\n' for name in names)+
        'extern "C" void reset(){tide_cuda::worker=0;tide_cuda::workers=1;tide_cuda::reset_arena();}\n')
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I"+str(root/"cuda"),
        "-I"+str(root/"ascendc"), "-I"+str(Path(torch.__file__).parent/"include"),
        str(source), "-o", str(library)], check=True)
    dll = ctypes.CDLL(str(library))
    for name, pointers, sizes in (("prepare",8,3),("center",8,5),("scale",7,3),("roots",10,3),("finish",13,3)):
        getattr(dll,"tide_block_kv_pool_"+name).argtypes = [ctypes.c_void_p]*pointers+[ctypes.c_int64]*sizes
    return dll


def candidate(dll, w, v, logits, visible, present, bar, stat, roots, attention):
    o,q,k=w.shape; d=v.shape[-1]; rows=o*q
    error=torch.zeros(1,dtype=torch.int32)
    def call(name,arrays,*sizes):
        dll.reset()
        getattr(dll,"tide_block_kv_pool_"+name)(*(x.data_ptr() for x in arrays),*sizes)
        assert error.item()==0
    clean=torch.empty_like(w);scores=torch.empty_like(w);meta=torch.empty(o,q,2,dtype=torch.int64)
    call("prepare",(w,logits,visible,present,clean,scores,meta,error),rows,k,attention)
    values=torch.empty_like(v);centered=torch.empty_like(v);anchor=torch.empty(o,1,d)
    call("center",(v,clean,visible,present,values,centered,anchor,error),o,q,k,d,attention)
    total=clean.sum(-1);prob=scores.softmax(-1);coeff=torch.empty_like(w);amp=torch.empty(o,q,1)
    call("scale",(clean,total,prob,meta,coeff,amp,error),rows,k,attention)
    mean=coeff@values; output=amp*mean
    sanitized=torch.empty_like(bar);scaled=torch.empty_like(bar);scalar=torch.empty_like(stat)
    flags=torch.empty(o,q,3,dtype=torch.bool)
    call("roots",(meta,total,bar,stat,roots,sanitized,scaled,scalar,flags,error),rows,d,attention)
    dv=coeff.transpose(1,2)@scaled
    dots=sanitized@centered.transpose(1,2)
    mean_dot=(sanitized*mean).sum(-1);center_dot=(sanitized*(coeff@centered)).sum(-1)
    anchor_dot=(sanitized@anchor.transpose(1,2)).squeeze(-1).contiguous()
    dw=torch.empty_like(w);dl=torch.empty_like(w)
    call("finish",(clean,prob,total,dots,mean_dot,center_dot,anchor_dot,scalar,visible,flags,dw,dl,error),rows,k,attention)
    return output,total,dw,dv,dl,flags


@pytest.mark.parametrize("attention",[False,True])
@pytest.mark.parametrize("offset",[0.,2**18])
def test_shared_pool_formula_and_independent_roots(pool_kernel,attention,offset):
    torch.manual_seed(671)
    w=torch.rand(2,4,5)*.6; w[0,0]=0; w[0,1]=torch.tensor([.25,.25,.5,0,0]);w[1,1,2]=0
    v=torch.randn(2,5,3)+offset; logits=torch.randn_like(w)*.4
    visible=torch.ones_like(w,dtype=torch.bool);visible[1,2,3:]=False
    present=torch.ones(2,4,dtype=torch.bool);present[1,3]=False;visible[1,3]=False
    roots=torch.ones(2,4,2,dtype=torch.bool);roots[0,2,0]=False;roots[1,0,1]=False
    bar=torch.randn(2,4,3);stat=torch.randn(2,4)
    bar[0,2]=float("nan");stat[1,0]=float("nan")
    actual=candidate(pool_kernel,w,v,logits,visible,present,bar,stat,roots,attention)
    wd=w.double().requires_grad_();vd=v.double().requires_grad_();ld=logits.double().requires_grad_()
    output=torch.zeros(2,4,3,dtype=torch.float64);totals=torch.zeros(2,4,dtype=torch.float64)
    loss=wd.sum()*0+vd.sum()*0+ld.sum()*0
    for o in range(2):
        for q in range(4):
            if not present[o,q]:continue
            ids=visible[o,q].nonzero().flatten();weights=wd[o,q,ids];total=weights.sum();totals[o,q]=total
            if attention:
                ids=ids[w[o,q,ids]>0]
                if len(ids):
                    probabilities=(ld[o,q,ids]+wd[o,q,ids].log()).softmax(0)
                    # The declared derivative at W=1 is left amplitude.
                    amplitude=total if float(total.detach())<=1 else total*0+1
                    result=amplitude*(probabilities[:,None]*vd[o,ids]).sum(0)
                else:result=total*0+torch.zeros(3,dtype=torch.float64)
            else:
                denominator=total if float(total.detach())>=1 else total*0+1
                result=(weights[:,None]*vd[o,ids]).sum(0)/denominator
            output[o,q]=result
            if roots[o,q,0]:loss=loss+(result*bar[o,q].double()).sum()
            if roots[o,q,1]:loss=loss+total*stat[o,q].double()
    expected=torch.autograd.grad(loss,(wd,vd,ld))
    torch.testing.assert_close(actual[0].double(),output,rtol=3e-6,atol=.07 if offset else 1e-6)
    torch.testing.assert_close(actual[1].double(),totals,rtol=1e-6,atol=2e-7)
    for index,(got,want) in enumerate(zip(actual[2:5],expected)):
        # The score derivative remains translation invariant at a large offset.
        tolerance=.08 if offset and index==0 else 8e-6
        torch.testing.assert_close(got.double(),want,rtol=3e-6,atol=tolerance)
    assert not actual[5][1,3].any()
    assert bool(actual[5][0,0,1]) is (not attention)
