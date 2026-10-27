"""Allocation §3 equations (2–5): hand values, independent roots and dependencies."""
import math
import pytest
import torch
from tidegraph.weighted import KVConfig,RegionalConfig
from tidegraph.weighted.scoring import Encoder
from test_weighted_kv_formulas import fixture,close


def groups(dtype):
    return {name:torch.tensor(value,dtype=dtype,requires_grad=True)
            for name,value in (("cur",[.3]),("mem",[.2,-.1]),("stat",[.4,.5,.6]))}


def test_match_and_grouped_literal_values_and_bias_roots(dtype):
    r=RegionalConfig(local_score="match",region_evidence="grouped",selection_width=1,
                     hidden_width=1,region_control_width=1,alpha_match=1.5,alpha_res=.25)
    b=fixture(dtype,memories=KVConfig(width=1,summary_width=1,epsilon=1.),regions=r)
    read=b.model.nodes[0].read_program
    with torch.no_grad():
        for p in read.parameters():p.zero_()
        for module in (read.score,read.evidence):
            for name,value in (("cur",1.),("mem",2.),("stat",-.5)):module.encoders[name].out_bias.fill_(value)
        read.score.gate_bias.fill_(math.log(2));read.score.up_bias.fill_(3);read.score.vector.fill_(2);read.score.bias.fill_(.2)
        read.evidence.cur.fill_(2);read.evidence.mem.fill_(-1);read.evidence.stat.fill_(.5);read.evidence.interaction.fill_(3);read.evidence.bias.fill_(.25)
    x,e,z=1/math.sqrt(2),2/math.sqrt(5),-1/math.sqrt(5)
    s=read.score(groups(dtype));expected=.2+1.5*x*e+math.log(2)
    close(s,s.new_tensor(expected))
    close(torch.autograd.grad(s,read.score.encoders["cur"].out_bias)[0],s.new_tensor([1.5*e/(2**1.5)]))
    value=2*x-e+.5*z+3*x*e+.25
    evidence=read.evidence(groups(dtype));close(evidence,evidence.new_tensor([value/math.sqrt(1+value*value)]))
    derivative=(2+3*e)/(2**1.5)/(1+value*value)**1.5
    close(torch.autograd.grad(evidence.sum(),read.evidence.encoders["cur"].out_bias)[0],evidence.new_tensor([derivative]))
    assert read.score.encoders["mem"].in_weight is not read.evidence.encoders["mem"].in_weight


def scalar_encoder(module,x):
    hidden=[]
    for i in range(module.in_weight.shape[0]):
        value=sum(module.in_weight[i,j]*x[j] for j in range(x.numel()))+module.hidden_bias[i]
        hidden.append(value*torch.sigmoid(value))
    out=torch.stack([sum(module.out_weight[i,j]*hidden[j] for j in range(len(hidden)))+module.out_bias[i]
                     for i in range(module.out_weight.shape[0])])
    return out/torch.sqrt(module.epsilon+sum(v*v for v in out)/out.numel())


def test_encoder_independent_scalar_equations_all_roots(dtype):
    module=Encoder(3,2,4,.2,torch.Generator().manual_seed(19),dtype)
    x=torch.tensor([.3,-.6,.4],dtype=dtype,requires_grad=True)
    actual=module(x);expected=scalar_encoder(module,x);close(actual,expected)
    roots=(x,*module.parameters())
    for i in range(4):
        for a,b in zip(torch.autograd.grad(actual[i],roots,retain_graph=True),torch.autograd.grad(expected[i],roots,retain_graph=True)):close(a,b)
    assert actual.norm()<=2


@pytest.mark.parametrize("key",["content_stats","content"])
def test_key_direct_dependency_and_nonzero_empty_memory_bias(dtype,key):
    rc=RegionalConfig(region_score="att",region_key=key,selection_width=2)
    b=fixture(dtype,memories=KVConfig(width=1,summary_width=1,epsilon=1.),regions=rc)
    read=b.model.nodes[0].read_program;g=groups(dtype)
    value=read.memory_key(g)
    ds=torch.autograd.grad(value.sum(),g["stat"],allow_unused=True,retain_graph=True)[0]
    if key=="content":
        assert ds is None;assert value.norm()<=math.sqrt(2)
        assert read.memory_key({**g,"mem":torch.zeros_like(g["mem"])}).count_nonzero()>0
    else:
        assert ds[0]==0;close(ds[1:],read.key_weight[:,-2:].sum(0))
    assert torch.autograd.grad(value.sum(),g["cur"],allow_unused=True)[0] is None
    changed={**g,"stat":g["stat"]+2}
    if key=="content":assert torch.equal(value,read.memory_key(changed))
    else:assert not torch.equal(value,read.memory_key(changed))


@pytest.mark.parametrize("weight",["received","mu"])
def test_region_query_hand_formula_and_independent_field_roots(dtype,weight):
    r=RegionalConfig(selection_mode="direct_attention",region_score="att",query_weight=weight,
                     selection_width=2,alpha_loc=.5,alpha_sel=1.3)
    program=fixture(dtype,regions=r).model.regions[0]
    with torch.no_grad():
        program.query.in_weight.copy_(torch.eye(2,dtype=dtype));program.query.hidden_bias.zero_()
        program.query.out_weight.copy_(torch.eye(2,dtype=dtype));program.query.out_bias.zero_()
    f={name:torch.tensor(value,dtype=dtype,requires_grad=True) for name,value in (
        ("w",[.2,.3,0.]),("mu",[.4,.6,0.]),("x",[[1.,0.],[0.,2.],[100.,100.]]),
        ("key",[[1.,2.],[-1.,.3],[.1,.4]]),("score",[.3,.2,.1]))}
    z=[.2,.6] if weight=="received" else [.4,1.2]
    q=[v/(1+math.exp(-v)) for v in z]
    expected=[.5*sl+1.3*(k[0]*q[0]+k[1]*q[1])/math.sqrt(2)
              for sl,k in zip((.3,.2,.1),((1.,2.),(-1.,.3),(.1,.4)))]
    score=program.scores(f);close(score,score.new_tensor(expected))
    absent="mu" if weight=="received" else "w"
    assert torch.autograd.grad(score.sum(),f[absent],allow_unused=True,retain_graph=True)[0] is None
    close(torch.autograd.grad(score.sum(),f["x"],retain_graph=True)[0][2],score.new_zeros(2))
    close(torch.autograd.grad(score.sum(),f["score"])[0],score.new_full((3,),.5))


def test_disabled_region_match_is_only_a_score_projection_equivalence(dtype):
    r=RegionalConfig(region_score="att",alpha_loc=1.,alpha_sel=0.)
    program=fixture(dtype,regions=r).model.regions[0]
    f={"w":torch.tensor([.2,.3],dtype=dtype),"mu":torch.tensor([.2,.3],dtype=dtype),
       "x":torch.ones((2,4),dtype=dtype),"key":torch.ones((2,4),dtype=dtype,requires_grad=True),
       "score":torch.tensor([1.,2.],dtype=dtype,requires_grad=True)}
    score=program.scores(f);assert torch.equal(score,f["score"])
    grad=torch.autograd.grad(score.sum(),f["key"])[0];assert torch.equal(grad,torch.zeros_like(grad))


@pytest.mark.parametrize("kwargs",[dict(region_score="ind",region_key="content"),dict(query_weight="mu"),
    dict(region_score="att",query_weight="mu"),dict(local_score="cat",alpha_match=1.),
    dict(selection_width=3),dict(local_score="match",alpha_res=-1.),
    dict(region_score="att",region_key="invalid")])
def test_scoring_configuration_applicability(kwargs):
    with pytest.raises(ValueError):RegionalConfig(**kwargs)
