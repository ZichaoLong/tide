"""0770593 allocation §2–4: reductions, head/bounds/query chain and all sending maps.

Head math here is ordinary Torch; its device implementation has a separate
actual-device acceptance. This does not yet certify a complete resident SelStep.
"""
import pytest
import torch
from resident_region_adapter import Frame,build
from resident_weighted_adapter import Run as Alloc,build as build_alloc
from resident_sending_adapter import Run as Send,build as build_send
from tidegraph.weighted import attention_send,proportional,total_zero,hard_selection
from tidegraph.weighted.entropy_allocation import allocate


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    return (build(tmp_path_factory.mktemp("region-math")),
            build_alloc(tmp_path_factory.mktemp("region-alloc")),
            build_send(tmp_path_factory.mktemp("region-send")))


def head(x,out):
    hidden=x.numel();matrix=torch.sin(torch.arange(hidden*x.numel()).float().reshape(hidden,-1)*.1)*.2
    projection=torch.cos(torch.arange(out*hidden).float().reshape(out,hidden)*.3)*.2
    return projection@torch.nn.functional.silu(matrix@x+.1)+.07


def features(fields,received):
    w,mu,s,e,x,key=fields;weight=w if received else mu
    ctx=torch.cat(((w[:,None]*e).sum(0)/w.sum().clamp_min(1),w.sum().log1p()[None],mu.sum().log1p()[None]))
    qry=(weight[:,None]*x).sum(0)/weight.sum().clamp_min(1)
    return ctx,qry


def reference(fields,mode,context,att,received,regularizer):
    w,mu,s,e,x,key=fields;ctx,qry=features(fields,received)
    if att:s=.7*s+.5*(key*head(qry,x.shape[1])).sum(1)/(x.shape[1]**.5)
    control=head(ctx,1)[0] if context else w.new_tensor(.35 if mode<=1 else .12)
    low=w.sum().clamp_max(1) if mode<=3 else mu.sum().clamp_max(1)
    high=w.new_tensor(1.);shift=None
    if mode<=1:
        rho=control.sigmoid() if context else control
        low=low+rho*(mu.sum().clamp_max(1)-low);high=low
    else:shift=control
    if mode in (0,2,4):
        return allocate(s,mu,low,high,s.new_tensor(.8),regularizer=regularizer,
                        ent_alpha=1.5 if regularizer=="ent" else None,score_shift=shift).controls
    if mode==1:return total_zero(s,mu,low,(7,2,4))
    if mode in (3,5):return hard_selection(s-shift,mu,1,(7,2,4))
    if mode==9:return proportional(mu,1)
    return attention_send(s,mu,w,mode={6:"direct_attention",7:"unit_attention",8:"pooled_attention"}[mode],
                          base="uniform" if mode==7 else "mu",temperature=.8)


CASES=[(mode,mode<4 and mode%2==0,att,"quad") for mode in range(10) for att in (False,True)]
CASES += [(0,True,True,"bern"),(0,True,True,"ent"),(1,True,True,"quad"),(3,True,True,"quad"),(2,False,True,"quad")]


@pytest.mark.parametrize("mode,context,att,regularizer",CASES)
def test_complete_composition_values_fields_and_declared_vjp(libraries,mode,context,att,regularizer):
    lib,allocation,sending=libraries
    fields=[torch.tensor(x,dtype=torch.float32,requires_grad=True) for x in (
        [.2,.1,.3],[.3,.5,.6],[.1,-.2,.8],[[.2,-.3],[.1,.4],[-.5,.2]],
        [[.2,.4],[-.3,.1],[.6,-.2]],[[.3,-.2],[.1,.5],[-.4,.2]])]
    received=mode<4;frame=Frame(lib,fields,mode,context,att,received);frame.stage(0)
    cf,qf=features(fields,received)
    if context:torch.testing.assert_close(frame.t[7],cf,atol=1e-6,rtol=1e-5)
    if att:torch.testing.assert_close(frame.t[8],qf,atol=1e-6,rtol=1e-5)
    ctx=frame.t[7].clone().requires_grad_();qry=frame.t[8].clone().requires_grad_()
    hc=head(ctx,1);hq=head(qry,2)
    if context:frame.t[9]=hc.detach().clone()
    if att:frame.t[10]=hq.detach().clone()
    frame.stage(1);s=frame.t[11][None];mu=frame.t[1][None];w=frame.t[0][None];g=torch.tensor([[.3,-.7,.9]])
    if mode in (0,2,4):
        rule={"quad":0,"bern":1,"ent":2}[regularizer]
        obj=Alloc(allocation,s,mu,frame.t[12][None],[3],rule,1.5 if rule==2 else 1.)
        dv,ds,flags=obj.backward(g,has_shift=context and mode==2)
    else:
        kind={1:4,3:5,5:5,6:0,7:1,8:2,9:3}[mode]
        obj=Send(sending,s,mu,w,frame.t[13][None],[[7,2,4]],[3],kind,2 if mode==7 else 0,1 if mode in (3,5,9) else 0)
        dv,ds,flags=obj.backward(g)
    assert obj.error.item()==0
    expected=reference(fields,mode,context,att,received,regularizer)
    torch.testing.assert_close(obj.controls[0],expected,atol=1e-6,rtol=1e-5)
    frame.t[14],frame.t[15],frame.t[16]=dv[0].contiguous(),ds[0].contiguous(),flags[0].contiguous()
    frame.stage(2)
    if frame.t[24].item():frame.t[26]=torch.autograd.grad(hc,ctx,frame.t[21])[0].contiguous()
    if frame.t[25].item():frame.t[27]=torch.autograd.grad(hq,qry,frame.t[22])[0].contiguous()
    frame.stage(3)
    wanted=torch.autograd.grad(expected,fields,g[0],allow_unused=True)
    actual=[frame.t[i] for i in (17,18,19,28,29,20)]
    for i,(a,e) in enumerate(zip(actual,wanted)):
        assert bool(frame.t[23][i])==(e is not None),(mode,context,att,i)
        if e is not None:torch.testing.assert_close(a,e,atol=1e-6,rtol=1e-5)
        else:assert a.eq(0).all()
