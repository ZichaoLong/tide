"""0770593 architecture §1/2: independent physical-atom formula and full VJP.

This checks arithmetic helpers only, not vendor execution or resident graph
integration. Complete target checks must cover the eventual composed component.
"""
import pytest
import torch
from resident_aggregate_adapter import build,call


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("aggregate-math"))


def transform(x,owners):
    a=torch.tensor([[[.3,-.2],[.1,.4],[-.2,.5]],[[.1,.7],[-.4,.2],[.3,.6]],[[.2,.3],[.4,-.1],[.2,.5]]])
    b=torch.tensor([[[.1,-.2,.3],[.4,.2,-.1]],[[.3,.1,.2],[.1,-.4,.2]],[[.4,.2,-.1],[-.2,.3,.1]]])
    e=torch.tensor([[.1,.2,-.1],[-.2,.3,.1],[.1,-.1,.2]])
    return torch.stack([b[o]@torch.nn.functional.silu(a[o]@row+e[o]) for row,o in zip(x,owners)])


def buffers(masses,alpha,root):
    n=len(masses);capacity=n+1;d=2
    payload=torch.tensor([[m,.2+j*.1,(-1)**j*.3] for j,m in enumerate(masses)]+[[float("nan")]*3])
    ids=torch.tensor(([0,0,1,2][:n])+[-42])
    if alpha==0 or max(masses)<=0:ids.fill_(-42)
    t=[payload,ids,torch.zeros(capacity,d),torch.full((capacity,),42,dtype=torch.int64),torch.zeros(1),
       torch.zeros(capacity,d),torch.zeros(capacity,d),torch.zeros(d+1),torch.tensor([.4,.3,-.7]),
       torch.full_like(payload,42),torch.zeros(capacity,d),torch.zeros(capacity,dtype=torch.bool),
       torch.zeros(capacity,dtype=torch.bool),torch.zeros(capacity,d),torch.zeros(capacity),torch.zeros(capacity,d)]
    if not root:t[8].fill_(float("nan"))
    return t,torch.tensor([n,capacity,d,root],dtype=torch.int64)


@pytest.mark.parametrize("masses",[[.3,.7,0.,-.2],[0.,-.3],[.2]])
@pytest.mark.parametrize("alpha",[0.,.25])
@pytest.mark.parametrize("root",[False,True])
def test_mean_source_chain_and_independent_payload_roots(library,masses,alpha,root):
    t,meta=buffers(masses,alpha,root);n=len(masses)
    assert call(library,0,t,meta,alpha)==0
    norm=(t[2]/torch.sqrt(1e-5+t[2].square().mean(-1,keepdim=True))).requires_grad_()
    t[13]=norm.detach().clone();t[14]=torch.rsqrt(1e-5+t[2].square().mean(-1))
    if alpha and max(masses)>0:
        f=transform(norm[:n],t[1][:n]);t[5][:n]=f.detach()
    assert call(library,1,t,meta,alpha)==0
    raw=t[0][:n].clone().requires_grad_();m=raw[:,0].clamp_min(0);total=m.sum()
    if total>0:
        z=raw[:,1:]
        if alpha:z=z+alpha*transform(z/torch.sqrt(1e-5+z.square().mean(-1,keepdim=True)),t[1][:n])
        mean=(m[:,None]*z).sum(0)/total
    else:mean=torch.zeros(2)
    wanted=torch.cat((total[None],mean));torch.testing.assert_close(t[7],wanted,atol=1e-6,rtol=1e-5)
    assert call(library,2,t,meta,alpha)==0
    if t[12].any():t[15]=torch.autograd.grad(f,norm,t[10][:n])[0].contiguous()
    assert call(library,3,t,meta,alpha)==0
    if root:
        expected=torch.autograd.grad(wanted,raw,t[8])[0]
        torch.testing.assert_close(t[9][:n],expected,atol=1e-6,rtol=1e-5)
    else:assert t[9].eq(0).all()
    assert t[11].tolist()==[root]*n+[False]
    assert t[12].tolist()==[bool(root and alpha and max(masses)>0)]*n+[False]
    assert t[9][-1].eq(0).all()
    # Duplicate class 0 contributes once for each physical row, with distinct z.
    if n==4 and alpha:assert not torch.equal(t[6][0],t[6][1])


@pytest.mark.parametrize("fault",["owner","nan","overflow"])
def test_invalid_actual_atoms(library,fault):
    t,meta=buffers([.3,.7],.25,True)
    if fault=="owner":t[1][0]=-1
    elif fault=="nan":t[0][0,1]=float("nan")
    else:t[0][:2,0]=3e38
    assert call(library,0,t,meta,.25)==(2 if fault=="owner" else 20)
