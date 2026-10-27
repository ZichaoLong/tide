"""KV §3/4 pool against per-row FP64 formula/autograd and joint directions."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]/"tools/device_online"
    directory=tmp_path_factory.mktemp("weighted-pool");cpp=directory/"adapter.cpp";output=directory/"adapter.so"
    signatures={
        "prepare": ([("w","float"),("v","float"),("l","float"),("n","int64_t"),("p","uint8_t"),("cw","float"),("cv","float"),("s","float"),("m","int64_t"),("e","int32_t")],4),
        "scale": ([("w","float"),("t","float"),("p","float"),("m","int64_t"),("c","float"),("a","float"),("e","int32_t")],3),
        "reverse_prepare": ([("m","int64_t"),("w","float"),("v","float"),("p","float"),("c","float"),("t","float"),("y","float"),("bar","float"),("stat","float"),("roots","uint8_t"),("cb","float"),("ct","float"),("center","float"),("vc","float"),("connected","uint8_t"),("e","int32_t")],4),
        "reverse_finish": ([("m","int64_t"),("w","float"),("p","float"),("t","float"),("dot","float"),("md","float"),("stat","float"),("connected","uint8_t"),("dw","float"),("dv","float"),("dl","float"),("e","int32_t")],4),
    }
    source='#define TIDE_KERNEL_CPU\n#include "weighted_pool_math.h"\n'
    for name,(pointers,nints) in signatures.items():
        ints=["rows","capacity","width","attention"] if nints==4 else ["rows","capacity","attention"]
        args=[t+"* "+n for n,t in pointers]+["int64_t "+n for n in ints]
        source+='extern "C" void '+name+'('+','.join(args)+'){tide_cuda::reset_arena();tide_weighted_pool_device::'+name+'('+','.join([n for n,t in pointers]+ints)+');}\n'
    cpp.write_text(source)
    subprocess.run(["g++","-std=c++17","-shared","-fPIC","-O2","-ffp-contract=off","-I"+str(root/"cuda"),"-I"+str(root/"ascendc"),
                    "-I"+str(Path(torch.__file__).parent/"include"),str(cpp),"-o",str(output)],check=True)
    lib=ctypes.CDLL(str(output))
    for name,(pointers,nints) in signatures.items():getattr(lib,name).argtypes=[ctypes.c_void_p]*len(pointers)+[ctypes.c_int64]*nints
    return lib


def run(lib,w,v,logits,lengths,present,bars,stats,roots,attention):
    rows,capacity,width=v.shape;error=torch.zeros(1,dtype=torch.int32)
    cw,cv,scores=torch.zeros_like(w),torch.zeros_like(v),torch.zeros_like(w);meta=torch.zeros(rows,3,dtype=torch.int64)
    lib.prepare(*(t.data_ptr() for t in (w,v,logits,lengths,present,cw,cv,scores,meta,error)),rows,capacity,width,attention)
    if error.item():return (error,)
    total=cw.sum(1);prob=scores.softmax(1) if attention else torch.zeros_like(w);coeff=torch.zeros_like(w);amp=torch.zeros_like(total)
    lib.scale(*(t.data_ptr() for t in (cw,total,prob,meta,coeff,amp,error)),rows,capacity,attention)
    mean=(coeff.unsqueeze(1)@cv).squeeze(1);output=mean*amp[:,None]
    cb,ct,center,vc=torch.zeros_like(bars),torch.zeros_like(stats),torch.zeros_like(v),torch.zeros_like(w)
    connected=torch.zeros(rows,3,dtype=torch.bool)
    lib.reverse_prepare(*(t.data_ptr() for t in (meta,cw,cv,prob,coeff,total,output,bars,stats,roots,cb,ct,center,vc,connected,error)),rows,capacity,width,attention)
    dv=vc.unsqueeze(2)@cb.unsqueeze(1);dots=(center@cb.unsqueeze(2)).squeeze(2);md=(mean[:,None,:]@cb[:,:,None]).reshape(rows)
    dw,dl=torch.zeros_like(w),torch.zeros_like(w)
    lib.reverse_finish(*(t.data_ptr() for t in (meta,cw,prob,total,dots,md,ct,connected,dw,dv,dl,error)),rows,capacity,width,attention)
    return error,output,total,dw,dv,dl,connected


def fixture(width=3):
    w=torch.tensor([[.2,0.,.3,0.],[.4,0.,.6,0.],[1.,2.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
    v=torch.sin(torch.arange(6*4*width).float()*.13).reshape(6,4,width)
    l=torch.cos(torch.arange(24).float()*.17).reshape(6,4)
    lengths=torch.tensor([3,3,3,3,0,-99]);present=torch.tensor([1,1,1,1,1,0],dtype=torch.bool)
    v[5]=float("nan");w[5]=float("nan");l[5]=float("nan")
    return w,v,l,lengths,present


def formula(w,v,l,lengths,present,attention,*,sum_dtype=None):
    leaves=[];outputs=[];totals=[]
    for row in range(w.shape[0]):
        wr,vr,lr=[x[row].double().detach().requires_grad_() for x in (w,v,l)];leaves.extend((wr,vr,lr))
        if not present[row]:outputs.append(torch.zeros(v.shape[-1],dtype=torch.double));totals.append(torch.tensor(0.,dtype=torch.double));continue
        n=int(lengths[row]);weights=wr[:n];values=vr[:n];scores=lr[:n];total=weights.sum(dtype=sum_dtype).to(weights.dtype);totals.append(total)
        if attention:
            positive=weights>0
            y=total.clamp_max(1)*(scores[positive]+weights[positive].log()).softmax(0)@values[positive] if positive.any() else torch.zeros(v.shape[-1],dtype=torch.double)
        else:y=(weights[:,None]*values).sum(0)/total.clamp_min(1)
        outputs.append(y)
    return leaves,outputs,totals


@pytest.mark.parametrize("attention",[False,True])
@pytest.mark.parametrize("root",[0,1,2])
def test_complete_return_and_independent_vjp(library,attention,root):
    w,v,l,n,p=fixture()
    # The declared FP32 total selects the clamp branch. Lifting .4f+.6f to
    # FP64 before summing changes that branch; all other oracle math is FP64.
    leaves,ys,ws=formula(w,v,l,n,p,attention,sum_dtype=torch.float32)
    roots=torch.ones(6,2,dtype=torch.bool)
    if root<2:roots[:,1-root]=False
    bars=torch.cos(torch.arange(18).float()).reshape(6,3);stats=torch.arange(6).float()*.1
    bars[2].zero_();stats[2]=0;bars[~roots[:,0]]=float("nan");stats[~roots[:,1]]=float("nan")
    bars[5]=float("nan");stats[5]=float("nan")
    result=run(library,w,v,l,n,p,bars,stats,roots,attention);error,output,total,dw,dv,dl,connected=result
    assert error.item()==0
    torch.testing.assert_close(output.double(),torch.stack(ys),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(total.double(),torch.stack(ws),rtol=1e-6,atol=1e-7)
    outputs=[];cotangents=[]
    for r in range(6):
        for j,y in enumerate((ys[r],ws[r])):
            if roots[r,j] and y.requires_grad:outputs.append(y);cotangents.append((bars[r] if j==0 else stats[r]).double())
    expected=torch.autograd.grad(outputs,leaves,cotangents,allow_unused=True)
    for r in range(6):
        for j,x in enumerate((dw,dv,dl)):
            ref=expected[3*r+j];assert connected[r,j].item()==(ref is not None)
            if ref is None:assert x[r].eq(0).all()
            else:torch.testing.assert_close(x[r].double(),ref,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize("attention",[False,True])
def test_joint_direction(library,attention):
    w=torch.tensor([[.2,.1,.3,.1]]);v=torch.sin(torch.arange(12).float()).reshape(1,4,3);l=torch.cos(torch.arange(4).float()).reshape(1,4)
    n=torch.tensor([4]);p=torch.ones(1,dtype=torch.bool);bars=torch.tensor([[.2,-.3,.4]]);stats=torch.tensor([.3]);roots=torch.ones(1,2,dtype=torch.bool)
    error,_,_,dw,dv,dl,_=run(library,w,v,l,n,p,bars,stats,roots,attention);assert not error.item()
    direction=[torch.cos(torch.arange(x.numel()).double()*.43).reshape_as(x) for x in (w,v,l)]
    def objective(sign):
        xs=[x.double()+sign*1e-5*d for x,d in zip((w,v,l),direction)]
        _,ys,ws=formula(*xs,n,p,attention);return (ys[0]*bars[0]).sum()+ws[0]*stats[0]
    fd=(objective(1)-objective(-1))/(2e-5);actual=sum((g.double()*d).sum() for g,d in zip((dw,dv,dl),direction))
    torch.testing.assert_close(actual,fd,rtol=2e-5,atol=2e-6)


def test_attention_translation_invariance_and_masked_poison(library):
    w,v,l,n,p=fixture(2);v[:3]=16384.;v[0,1]=float("nan");l[0,1]=float("nan")
    bars=torch.ones(6,2);stats=torch.zeros(6);roots=torch.ones(6,2,dtype=torch.bool)
    error,_,_,_,_,dl,_=run(library,w,v,l,n,p,bars,stats,roots,True)
    assert not error.item() and dl.eq(0).all()


@pytest.mark.parametrize("bad",["negative","capacity","value","overflow"])
def test_explicit_invalid_pool(library,bad):
    w,v,l,n,p=fixture()
    if bad=="negative":w[0,0]=-1
    if bad=="capacity":n[0]=5
    if bad=="value":v[0,0,0]=float("inf")
    if bad=="overflow":w[0,:]=torch.finfo(w.dtype).max
    result=run(library,w,v,l,n,p,torch.zeros(6,3),torch.zeros(6),torch.zeros(6,2,dtype=torch.bool),True)
    assert result[0].item()==29


def test_cpu_adapter_arena_lifetime(library):
    # More invocations than fit in the semantic adapter's arena without reset.
    w=torch.ones(1,1);v=torch.ones(1,1,1);l=torch.zeros(1,1);n=torch.ones(1,dtype=torch.int64);p=torch.ones(1,dtype=torch.bool)
    cw=torch.zeros_like(w);cv=torch.zeros_like(v);scores=torch.zeros_like(w);meta=torch.zeros(1,3,dtype=torch.int64);error=torch.zeros(1,dtype=torch.int32)
    args=tuple(t.data_ptr() for t in (w,v,l,n,p,cw,cv,scores,meta,error))
    for _ in range(2048):library.prepare(*args,1,1,1,True)
    assert not error.item() and scores.eq(0).all() and meta.tolist()==[[1,1,1]]
