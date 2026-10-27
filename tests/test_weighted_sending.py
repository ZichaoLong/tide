"""Allocation §4/§6.3: independent equations, LP vertices and branch derivatives."""
import itertools
import math
import pytest
import torch
from tidegraph.weighted import (RegionalConfig, attention_shares, attention_send,
                                proportional, total_zero, hard_selection)
from tidegraph.weighted.config import REGIONAL, GAIN
from test_weighted_kv_formulas import close


MODES = sorted(REGIONAL | GAIN)


def test_attention_zero_base_right_derivative(dtype):
    s=torch.tensor([0.,math.log(2),math.log(3)],dtype=dtype,requires_grad=True)
    a=torch.tensor([1.,0.,1.],dtype=dtype,requires_grad=True)
    tau=s.new_tensor(1.,requires_grad=True)
    p=attention_shares(s,a,tau)
    close(p,s.new_tensor([.25,0.,.75]))
    ds,da,dt=torch.autograd.grad(p[1],(s,a,tau),retain_graph=True)
    close(ds,torch.zeros_like(s));close(da,s.new_tensor([0.,.5,0.]));close(dt,s.new_zeros(()))
    ds,da,dt=torch.autograd.grad(p[0],(s,a,tau))
    close(ds,s.new_tensor([3/16,0.,-3/16]));close(da,s.new_tensor([3/16,-1/8,-3/16]))
    close(dt,s.new_tensor(3*math.log(3)/16))
    eps=1e-3 if dtype==torch.float32 else 1e-6
    perturbed=a.detach().clone();perturbed[1]=eps
    close((attention_shares(s.detach(),perturbed)[1]/eps)*(1+eps/2),s.new_tensor(.5))


@pytest.mark.parametrize("mode,base",[("direct_attention","mu"),("direct_attention","received"),
    ("unit_attention","mu"),("unit_attention","uniform"),("pooled_attention","mu")])
def test_attention_formula_and_all_independent_roots(dtype,mode,base):
    s=torch.tensor([.2,-.3,.1],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.15,.35,.2],dtype=dtype,requires_grad=True)
    w=torch.tensor([.1,.2,.15],dtype=dtype,requires_grad=True)
    tau=s.new_tensor(.7,requires_grad=True)
    a=mu if base=="mu" else w if base=="received" else torch.ones_like(mu)
    # Plain exp quotient is an independent smooth-domain oracle, without custom VJP.
    exp=a*torch.exp(s/tau);p=exp/exp.sum()
    expected=mu*p if mode=="direct_attention" else p if mode=="unit_attention" else mu.sum()*p
    actual=attention_send(s,mu,w,mode=mode,base=base,temperature=tau)
    close(actual,expected)
    for i in range(3):
        ga=torch.autograd.grad(actual[i],(s,mu,w,tau),allow_unused=True,retain_graph=True)
        ge=torch.autograd.grad(expected[i],(s,mu,w,tau),allow_unused=True,retain_graph=True)
        for x,y in zip(ga,ge):
            if x is None or y is None:assert x is y
            else:close(x,y)


def test_attention_candidate_presence_and_exceeding_individual_cap(dtype):
    s=torch.tensor([math.log(9),0.],dtype=dtype);mu=s.new_tensor([.1,.1])
    pooled=attention_send(s,mu,mu,mode="pooled_attention")
    close(pooled,s.new_tensor([.18,.02]));assert pooled[0]>mu[0]
    zero=torch.zeros_like(s)
    close(attention_send(s,zero,zero,mode="unit_attention"),s.new_tensor([.9,.1]))
    close(attention_send(s,zero,zero,mode="unit_attention",base="mu"),zero)
    assert attention_send(s[:0],mu[:0],mu[:0],mode="unit_attention").shape==(0,)
    with pytest.raises(ValueError):attention_send(s,mu,mu,mode="pooled_attention",base="uniform")


def test_attention_all_zero_vjp_differs_from_directional_derivative(dtype):
    s=torch.zeros(2,dtype=dtype,requires_grad=True);mu=torch.zeros(2,dtype=dtype,requires_grad=True)
    c=attention_send(s,mu,mu,mode="direct_attention")
    ds,dm=torch.autograd.grad(c.sum(),(s,mu));close(ds,s.detach());close(dm,mu.detach())
    # At the corner the directional rule is delta^2/sum(delta), not a linear map.
    eps=1e-4;direction=s.new_tensor([1.,2.])
    close(attention_send(s.detach(),eps*direction,eps*direction,mode="direct_attention")/eps,
          s.new_tensor([1/3,4/3]))


def test_sending_positive_support_and_versions(dtype):
    s=torch.tensor([-10000.,0.],dtype=dtype)
    with pytest.raises(ArithmeticError,match="support underflowed"):attention_shares(s,torch.ones_like(s))
    for mutate in ("score","base","temperature"):
        s=torch.zeros(2,dtype=dtype,requires_grad=True);a=torch.ones_like(s,requires_grad=True);tau=s.new_tensor(1.,requires_grad=True)
        p=attention_shares(s,a,tau)
        with torch.no_grad():{"score":s,"base":a,"temperature":tau}[mutate].add_(.1)
        with pytest.raises(RuntimeError,match="modified by an inplace operation"):p[0].backward()


def test_fractional_zero_temperature_is_not_hard_topk(dtype):
    s=torch.tensor([3.,2.,1.,0.],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.3,.3,.3,.3],dtype=dtype,requires_grad=True);total=s.new_tensor(1.,requires_grad=True)
    c=total_zero(s,mu,total,(3,2,1,0));close(c,s.new_tensor([.3,.3,.3,.1]))
    assert (c>0).sum()==4 # A total-strength bound k=1 does not restrict K to 1.
    ds,dm,db=torch.autograd.grad(c[3],(s,mu,total),allow_unused=True)
    assert ds is None;close(dm,s.new_tensor([-1.,-1.,-1.,0.]));close(db,s.new_tensor(1.))
    close(hard_selection(s,mu,1,(3,2,1,0)),s.new_tensor([.3,0.,0.,0.]))


def test_sorted_fill_optimal_against_independent_vertices(dtype):
    s=torch.tensor([-.2,1.3,.4],dtype=dtype);mu=s.new_tensor([.7,.6,.8]);b=.95
    actual=total_zero(s,mu,b,(8,3,5));value=float(actual@s)
    # Enumerate all vertices: at most one free coordinate, remaining at either cap or zero.
    for free in range(3):
        others=[i for i in range(3) if i!=free]
        for mask in itertools.product((0,1),repeat=2):
            candidate=[0.]*3
            for i,bit in zip(others,mask):candidate[i]=float(mu[i])*bit
            candidate[free]=b-sum(candidate)
            if 0<=candidate[free]<=float(mu[free]):
                assert value+1e-6>=float(s@s.new_tensor(candidate))
    close(actual.sum(),s.new_tensor(b))


def test_stable_ids_hard_fallback_and_caps_branch(dtype):
    s=torch.tensor([1.,1.,1.],dtype=dtype,requires_grad=True);mu=torch.tensor([.2,.4,.5],dtype=dtype,requires_grad=True)
    close(total_zero(s,mu,.6,(9,2,4)),s.new_tensor([0.,.4,.2]))
    hard=hard_selection(s,mu,1,(9,2,4));close(hard,s.new_tensor([0.,.4,0.]))
    ds,dm=torch.autograd.grad(hard.sum(),(s,mu),allow_unused=True);assert ds is None;close(dm,s.new_tensor([0.,1.,0.]))
    close(hard_selection(-s,mu,2,(9,2,4)),hard)
    close(hard_selection(s,torch.zeros_like(mu),2,(9,2,4)),torch.zeros_like(mu))
    with pytest.raises(ValueError):total_zero(s,mu,.5,(1,1,3))


def test_sorted_fill_full_cap_endpoint_and_versions(dtype):
    s=torch.tensor([1.,3.,2.],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.1,.3,.6],dtype=dtype,requires_grad=True);total=mu.detach().sum().requires_grad_()
    c=total_zero(s,mu,total,(0,1,2));assert torch.equal(c,mu)
    gm,gb=torch.autograd.grad(c.sum(),(mu,total));close(gm,torch.ones_like(mu));close(gb,total.new_zeros(()))
    c=total_zero(s,mu,total,(0,1,2))
    with torch.no_grad():total.add_(.01)
    with pytest.raises(RuntimeError,match="modified by an inplace operation"):c.sum().backward()


def test_proportional_and_zero_corner_derivative(dtype):
    mu=torch.tensor([.3,.4,.6],dtype=dtype,requires_grad=True)
    close(proportional(mu,1),mu/1.3);close(proportional(mu,2),mu)
    zero=torch.zeros(3,dtype=dtype,requires_grad=True)
    close(torch.autograd.grad(proportional(zero,1).sum(),zero)[0],torch.ones_like(zero))


def test_sending_smooth_branch_gradcheck():
    s=torch.tensor([.7,.3,-.1],dtype=torch.float64,requires_grad=True)
    mu=torch.tensor([.4,.5,.3],dtype=torch.float64,requires_grad=True);b=s.new_tensor(.6,requires_grad=True);t=s.new_tensor(.8,requires_grad=True)
    assert torch.autograd.gradcheck(attention_shares,(s,mu,t))
    assert torch.autograd.gradcheck(lambda s,mu,b:total_zero(s,mu,b,(2,1,0)),(s,mu,b))


@pytest.mark.parametrize("mode",MODES)
def test_sending_configuration_defaults_and_capacity(mode):
    c=RegionalConfig(selection_mode=mode,k=2 if "attention" not in mode else None)
    assert c.capacity(4)==(2 if mode in {"hard_selection","regional_threshold_hard"} else 4)
    assert (c.region_control_mode is not None)==(mode in REGIONAL)
    assert c.backward_identity and "hst" not in c.backward_identity.lower()
    if "attention" in mode:
        with pytest.raises(ValueError):RegionalConfig(selection_mode=mode,k=1)
    if mode in {"regional_total_zero","regional_threshold_hard","hard_selection","proportional"}:
        for field,value in (("allocation_temperature",1.),("regularizer","quad"),("ent_alpha",1.)):
            with pytest.raises(ValueError):RegionalConfig(selection_mode=mode,**{field:value})


@pytest.mark.parametrize("kwargs",[dict(selection_mode="regional_total",eta=0.),
    dict(selection_mode="regional_threshold",total_fraction=.5),dict(selection_mode="constrained",region_evidence="cat"),
    dict(selection_mode="unit_attention",send_gain="context",fixed_gain=1.),
    dict(selection_mode="proportional",gamma_max=2.,fixed_gain=3.)])
def test_reject_inapplicable_sending_fields(kwargs):
    with pytest.raises(ValueError):RegionalConfig(**kwargs)
