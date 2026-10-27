"""All formal sending rules: complete records, independent roots, training/cuts.

Local truth anchors are in test_weighted_sending; these establish integration.
"""
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.checkpoint import save,load
from tidegraph.compare import equivalent,objective
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import RegionalConfig,bind
from tidegraph.weighted.config import ATTENTION,REGIONAL,TOTAL
from isolated_cases import vjp
from test_weighted_execution import setup,variables
from test_weighted_kv_formulas import fixture,close
from test_weighted_sending import MODES


def configured(dtype,mode,family="fc",control="context",gain="context"):
    old,_,xs,stop=setup(dtype,family)
    rs=[]
    for layout in old.graph.region_layouts:
        fields=dict(selection_mode=mode,gamma_max=1.7)
        if mode not in ATTENTION:fields["k"]=len(layout.members)
        if mode in REGIONAL:
            fields["region_control_mode"]=control
            if control=="fixed":fields["total_fraction" if mode in TOTAL else "eta"]=.3
        else:
            fields["send_gain"]=gain
            if gain=="fixed":fields["gain_logit"]=-.3
        rs.append(RegionalConfig(**fields))
    return bind(old.graph,memories=old.memories,regions=tuple(rs),alpha_agg=.2,dtype=dtype),xs,stop


@pytest.mark.parametrize("mode",MODES)
def test_all_sending_native_records_roots_and_every_cut(dtype,mode):
    family="hb" if MODES.index(mode)%2 else "dag"
    b,xs,stop=configured(dtype,mode,family)
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    engine=Native(b.graph,b.model,algorithm="greedy",packed=True)
    actual=engine.run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual)
    roots=variables(b,xs)
    equivalent(vjp(objective(expected),roots),vjp(objective(actual),roots))
    equivalent(vjp(expected.outputs[0][-1],roots),vjp(actual.outputs[0][-1],roots))
    for key in expected.trace[0]["descriptor"]:
        equivalent(vjp(expected.trace[0]["descriptor"][key],roots),vjp(actual.trace[0]["descriptor"][key],roots))
    for cut in range(1,stop):
        head=engine.run(q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
        tail=[x for x in xs if x.time>=cut]
        continued=engine.run(head.continuation,tail,stop,sealed_until=stop)
        equivalent(expected.continuation,continued.continuation)
        check=reference(b.graph,b.model,head.continuation,tail,stop,sealed_until=stop)
        equivalent(check,continued);equivalent(vjp(objective(check),roots),vjp(objective(continued),roots))


@pytest.mark.parametrize("mode",MODES)
def test_all_sending_optimizer_checkpoint_inference(dtype,mode,tmp_path):
    a,xa,stop=configured(dtype,mode,control="fixed",gain="fixed")
    b,xb,_=configured(dtype,mode,control="fixed",gain="fixed")
    native=Native(b.graph,b.model,algorithm="frontier",packed=True)
    oa=torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad],lr=.001)
    ob=torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        ra=reference(a.graph,a.model,Continuation(a.graph.identity,1),xa,stop,sealed_until=stop)
        rb=native.run(Continuation(b.graph.identity,1),xb,stop,sealed_until=stop)
        equivalent(ra,rb);objective(ra).backward();objective(rb).backward()
        for x,y in zip(a.model.parameters(),b.model.parameters()):equivalent(x.grad,y.grad)
        oa.step();ob.step();equivalent(a.model.state_dict(),b.model.state_dict())
    head=native.run(Continuation(b.graph.identity,1),[x for x in xb if x.time<3],3,sealed_until=3)
    save(tmp_path/"sending.pt",b.graph,b.model,head.continuation,ob)
    c,_,_=configured(dtype,mode,control="fixed",gain="fixed")
    oc=torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad],lr=.001)
    restored=load(tmp_path/"sending.pt",c.graph,c.model,oc)
    equivalent(head.continuation.detach(),restored);equivalent(ob.state_dict(),oc.state_dict())
    tail=[x for x in xb if x.time>=3]
    with torch.no_grad():
        rb=native.run(head.continuation,tail,stop,sealed_until=stop)
        rc=Native(c.graph,c.model,algorithm="streaming").run(restored,tail,stop,sealed_until=stop)
        equivalent(rb,rc)


@pytest.mark.parametrize("native",[False,True])
def test_threshold_head_is_unbounded_and_gain_descriptor_has_three_fields(dtype,native):
    rc=RegionalConfig(selection_mode="regional_threshold",gamma_max=2.,k=2)
    b=fixture(dtype,nodes=2,regions=rc)
    with torch.no_grad():
        for node,score in zip(b.model.nodes,(1.4,.6)):
            node.read_program.score.vector.zero_();node.read_program.score.bias.fill_(score)
        b.model.regions[0].head.vector.zero_();b.model.regions[0].head.bias.fill_(.4)
    xs=[External(0,i,0,0,torch.tensor([.25,0.,0.,0.,0.],dtype=dtype)) for i in range(2)]
    q=Continuation(b.graph.identity,1)
    run=Native(b.graph,b.model).run if native else lambda q,x,s,**kw:reference(b.graph,b.model,q,x,s,**kw)
    result=run(q,xs,1,sealed_until=1)
    close(torch.stack([result.trace[i]["control"] for i in range(2)]),torch.tensor([.5,.2],dtype=dtype))
    gain=fixture(dtype,regions=RegionalConfig(selection_mode="proportional",gamma_max=3.,gain_logit=0.))
    node=gain.model.nodes[0]
    assert set(node.read_program.field_layout)=={"w","mu","score"}
    # gamma=2 from the independent sigmoid-logit formula; raw r=.3 -> mu=.6.
    result=reference(gain.graph,gain.model,Continuation(gain.graph.identity,1),
        [External(0,0,0,0,torch.tensor([.3,1.,0.,0.,0.],dtype=dtype))],1,sealed_until=1)
    close(result.trace[0]["descriptor"]["mu"],torch.tensor(.6,dtype=dtype))


def test_uniform_unit_attention_full_runs_on_real_zero_candidate(dtype):
    b=fixture(dtype,nodes=2,regions=RegionalConfig(selection_mode="unit_attention"))
    xs=[External(0,i,0,0,torch.zeros(5,dtype=dtype,requires_grad=True)) for i in range(2)]
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,1,sealed_until=1)
    actual=Native(b.graph,b.model).run(q,xs,1,sealed_until=1)
    equivalent(expected,actual);assert len(actual.outputs)==2
    assert all(e["active"] for e in actual.trace)
    assert all(s.last_time==-1 for s in actual.continuation.states.values())


@pytest.mark.parametrize("native",[False,True])
@pytest.mark.parametrize("gain",["context","fixed"])
def test_gain_scalar_formula_and_independent_bias_derivative(dtype,native,gain):
    fields=dict(selection_mode="proportional",gamma_max=3.,send_gain=gain)
    if gain=="fixed":fields["gain_logit"]=0.
    b=fixture(dtype,regions=RegionalConfig(**fields));node=b.model.nodes[0]
    if gain=="context":
        with torch.no_grad():node.read_program.gain.vector.zero_();node.read_program.gain.bias.zero_()
        root=node.read_program.gain.bias
    else:root=node.read_program.gain_logit
    q=Continuation(b.graph.identity,1)
    xs=[External(0,0,0,0,torch.tensor([.3,1.,0.,0.,0.],dtype=dtype))]
    result=(Native(b.graph,b.model).run(q,xs,1,sealed_until=1) if native else
            reference(b.graph,b.model,q,xs,1,sealed_until=1))
    mu=result.trace[0]["descriptor"]["mu"]
    close(mu,mu.new_tensor(.6));close(torch.autograd.grad(mu,root)[0],mu.new_tensor(.15))


@pytest.mark.parametrize("native",[False,True])
def test_node_gain_lower_bound_and_paired_hard_counterexample(dtype,native):
    controls=[]
    for mode in ("constrained","hard_selection"):
        b=fixture(dtype,nodes=2,regions=RegionalConfig(selection_mode=mode,gamma_max=2.,k=2))
        with torch.no_grad():
            for node in b.model.nodes:node.read_program.score.vector.zero_();node.read_program.score.bias.fill_(-1.)
        q=Continuation(b.graph.identity,1)
        xs=[External(0,i,0,0,torch.tensor([.2,1.,0.,0.,0.],dtype=dtype)) for i in range(2)]
        r=(Native(b.graph,b.model).run(q,xs,1,sealed_until=1) if native else
           reference(b.graph,b.model,q,xs,1,sealed_until=1))
        controls.append(torch.stack([e["control"] for e in r.trace]))
    close(controls[0],controls[0].new_tensor([.4,.4]))
    close(controls[1],controls[1].new_tensor([.4,0.]))


def test_pdg_feedback_sending_and_gain_state_retention(dtype):
    b,xs,stop=configured(dtype,"direct_attention","pdg")
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    actual=Native(b.graph,b.model,algorithm="greedy",packed=True).run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual);equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(actual),variables(b,xs)))
