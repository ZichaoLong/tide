"""Independent native Alloc, plus regional control/optimizer/checkpoint use."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save,load
from tidegraph.compare import equivalent,objective
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import allocate,bind
from test_weighted_execution import setup,variables
from test_weighted_entropy_allocation import RULES,decimal_reference
from isolated_cases import vjp


@pytest.mark.parametrize("kind,alpha",RULES)
def test_native_entropy_decimal_solution_and_exact_map_gradients(dtype,kind,alpha):
    import _tide_native as core
    t=torch.tensor([.2,-.1,.5,-.3],dtype=dtype,requires_grad=True)
    mu=torch.tensor([.7,.6,.8,0.],dtype=dtype,requires_grad=True)
    total=torch.tensor(.9,dtype=dtype,requires_grad=True);tau=torch.tensor(.7,dtype=dtype,requires_grad=True)
    actual=core.weighted_allocate(t,mu,total,total,tau,kind,alpha)
    expected=t.new_tensor(decimal_reference(t.detach(),mu.detach(),total.detach(),total.detach(),tau.detach(),kind,alpha))
    equivalent(actual,expected)
    candidate=allocate(t,mu,total,total,tau,regularizer=kind,ent_alpha=alpha).controls
    roots=dict(t=t,mu=mu,total=total,tau=tau)
    equivalent(vjp(candidate,roots),vjp(actual,roots))
    for i in range(4):equivalent(vjp(candidate[i],roots),vjp(actual[i],roots))


@pytest.mark.parametrize("kind,alpha",[("bern",None),("ent",1.),("ent",1.5)])
def test_entropy_regional_optimizer_and_checkpoint(dtype,kind,alpha,tmp_path):
    def build():
        old,_,xs,stop=setup(dtype,"fc")
        r=tuple(replace(c,regularizer=kind,ent_alpha=alpha) for c in old.regions)
        return bind(old.graph,memories=old.memories,regions=r,alpha_agg=.2,dtype=dtype),xs,stop
    a,xa,stop=build();b,xb,_=build()
    native=Native(b.graph,b.model,algorithm="frontier",packed=True)
    oa=torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad],lr=.001)
    ob=torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        expected=reference(a.graph,a.model,Continuation(a.graph.identity,1),xa,stop,sealed_until=stop)
        actual=native.run(Continuation(b.graph.identity,1),xb,stop,sealed_until=stop)
        equivalent(expected,actual)
        equivalent(vjp(expected.outputs[0][-1],variables(a,xa)),vjp(actual.outputs[0][-1],variables(b,xb)))
        objective(expected).backward();objective(actual).backward()
        for x,y in zip(a.model.parameters(),b.model.parameters()):equivalent(x.grad,y.grad)
        oa.step();ob.step();equivalent(a.model.state_dict(),b.model.state_dict())
    q=Continuation(b.graph.identity,1);head=native.run(q,[x for x in xb if x.time<3],3,sealed_until=3)
    save(tmp_path/"entropy.pt",b.graph,b.model,head.continuation,ob)
    c,_,_=build();oc=torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad],lr=.001)
    restored=load(tmp_path/"entropy.pt",c.graph,c.model,oc)
    tail=[x for x in xb if x.time>=3]
    expected=native.run(head.continuation,tail,stop,sealed_until=stop)
    actual=Native(c.graph,c.model,algorithm="streaming").run(restored,tail,stop,sealed_until=stop)
    equivalent(expected,actual);equivalent(ob.state_dict(),oc.state_dict())


@pytest.mark.parametrize("native",[False,True])
@pytest.mark.parametrize("mutate",["scores","total"])
def test_entropy_backward_checks_input_versions(dtype,native,mutate):
    t=torch.tensor([.2,-.1],dtype=dtype,requires_grad=True);mu=torch.ones_like(t)
    total=t.new_tensor(.7,requires_grad=True);tau=t.new_tensor(.8)
    if native:
        import _tide_native as core
        c=core.weighted_allocate(t,mu,total,total,tau,"bern",None)
    else:c=allocate(t,mu,total,total,tau,regularizer="bern").controls
    with torch.no_grad(): (t if mutate=="scores" else total).add_(.1)
    with pytest.raises(RuntimeError,match="modified by an inplace operation"):c.sum().backward()


def test_native_entropy_underflow_is_not_a_route_change(dtype):
    import _tide_native as core
    t=torch.tensor([-10000.,0.],dtype=dtype)
    for kind,alpha in (("bern",None),("ent",1.)):
        with pytest.raises(RuntimeError,match="support underflowed"):
            core.weighted_allocate(t,torch.ones_like(t),t.new_tensor(0.),t.new_tensor(2.),t.new_tensor(1.),kind,alpha)
