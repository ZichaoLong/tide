"""C++ weighted formulas/independent schedules, separate from Python evidence."""
import pytest
import torch
from tidegraph import Continuation,External
from tidegraph.compare import equivalent,objective
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import allocate_quad
from isolated_cases import vjp
from test_weighted_execution import setup,variables
from test_weighted_allocation import enumerate_quad
from dataclasses import replace
from tidegraph.weighted import bind
from tidegraph.checkpoint import save,load


@pytest.mark.parametrize("family",["fc","hb","dag","pdg"])
@pytest.mark.parametrize("packed",[False,True])
def test_native_weighted_records_roots_and_cuts(dtype,family,packed):
    b,_,xs,stop=setup(dtype,family)
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    for algorithm in (("streaming","greedy") if family=="pdg" else ("streaming","frontier","greedy")):
        native=Native(b.graph,b.model,algorithm=algorithm,packed=packed)
        actual=native.run(q,xs,stop,sealed_until=stop)
        equivalent(expected,actual)
        roots=variables(b,xs)
        equivalent(vjp(objective(expected),roots),vjp(objective(actual),roots))
        for name in ("w","mu","score","evidence"):
            equivalent(vjp(expected.trace[0]["descriptor"][name],roots),
                       vjp(actual.trace[0]["descriptor"][name],roots))
        equivalent(vjp(expected.outputs[0][-1],roots),vjp(actual.outputs[0][-1],roots))
        for cut in (1,3,stop-1):
            first=native.run(q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
            tail=native.run(first.continuation,[x for x in xs if x.time>=cut],stop,sealed_until=stop)
            equivalent(actual.continuation,tail.continuation)
            equivalent([e for e in actual.trace if e["time"]>=cut],tail.trace)


def test_native_independent_solver_and_vjp(dtype):
    import _tide_native as core
    gen=torch.Generator().manual_seed(42)
    for _ in range(12):
        s=torch.randn(4,generator=gen,dtype=dtype,requires_grad=True)
        mu=(torch.rand(4,generator=gen,dtype=dtype)*.7+.2).requires_grad_()
        total=mu.sum()*.61
        temp=torch.tensor(.9,dtype=dtype,requires_grad=True)
        expected=enumerate_quad(s,mu,total,total,temp)
        actual=core.weighted_allocate_quad(s,mu,total,total,temp)
        equivalent(expected,actual)
        # Original independent face solver verifies both scalar candidates.
        weights=s.new_tensor([.2,-.4,.7,.3])
        ga=torch.autograd.grad((actual*weights).sum(),(s,mu,temp),retain_graph=True)
        gp=torch.autograd.grad((allocate_quad(s,mu,total,total,temp).controls*weights).sum(),(s,mu,temp),retain_graph=True)
        equivalent(ga,gp)


@pytest.mark.parametrize("mode",["0","-","+"])
def test_native_qry_and_read_snapshots(dtype,mode):
    old,_,xs,stop=setup(dtype,"fc",read_mode=mode)
    memory=replace(old.memories[0],memory_read="qry",read_frequencies=(.3,))
    b=bind(old.graph,memories=memory,regions=old.regions,dtype=dtype)
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    actual=Native(b.graph,b.model,algorithm="frontier",packed=True).run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual)
    equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(actual),variables(b,xs)))


@pytest.mark.parametrize("family",["fc","hb"])
def test_native_optimizer_multiple_updates(dtype,family):
    a,_,xa,stop=setup(dtype,family);b,_,xb,_=setup(dtype,family)
    oa=torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad],lr=.002)
    ob=torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad],lr=.002)
    engine=Native(b.graph,b.model,algorithm="frontier",packed=True)
    for _ in range(3):
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        expected=reference(a.graph,a.model,Continuation(a.graph.identity,1),xa,stop,sealed_until=stop)
        actual=engine.run(Continuation(b.graph.identity,1),xb,stop,sealed_until=stop)
        equivalent(expected,actual)
        objective(expected).backward();objective(actual).backward()
        for pa,pb in zip(a.model.parameters(),b.model.parameters()):equivalent(pa.grad,pb.grad)
        oa.step();ob.step();equivalent(a.model.state_dict(),b.model.state_dict())


def test_native_pending_checkpoint_new_owner_and_unit_transport(dtype,tmp_path):
    b,_,xs,stop=setup(dtype,"pdg")
    engine=Native(b.graph,b.model,algorithm="greedy",packed=True)
    q=Continuation(b.graph.identity,1)
    first=engine.run(q,[x for x in xs if x.time<2],2,sealed_until=2)
    assert first.continuation.pending
    save(tmp_path/"native.pt",b.graph,b.model,first.continuation)
    other,_,_,_=setup(dtype,"pdg")
    restored=load(tmp_path/"native.pt",other.graph,other.model)
    tail=[x for x in xs if x.time>=2]
    expected=engine.run(first.continuation,tail,stop,sealed_until=stop)
    actual=Native(other.graph,other.model,algorithm="streaming").run(restored,tail,stop,sealed_until=stop)
    equivalent(expected,actual)
    # C++ public model validation also owns the transport invariant, independently
    # of the Python binding preflight.
    core=engine.core
    scales=list(engine.weights.edge_scale)
    scales[0]=scales[0]*2
    engine.weights.edge_scale=scales
    with pytest.raises(ValueError,match="unit scales"):
        core.Streaming(engine.compiled,engine.weights,engine.options)


def test_warm_kv_independent_initial_state_roots(dtype):
    b,_,xs,stop=setup(dtype,"fc",read_mode="-")
    warm=reference(b.graph,b.model,Continuation(b.graph.identity,1),[x for x in xs if x.time<3],3,sealed_until=3)
    q=warm.continuation.detach()
    roots={}
    for owner,state in q.states.items():
        for name,value in state.slots.items():
            if value.is_floating_point():
                state.slots[name]=value.detach().requires_grad_()
                roots[f"{owner}.{name}"]=state.slots[name]
    tail=[x for x in xs if x.time>=3]
    expected=reference(b.graph,b.model,q,tail,stop,sealed_until=stop)
    actual=Native(b.graph,b.model,algorithm="frontier",packed=True).run(q,tail,stop,sealed_until=stop)
    for a,c in ((expected.trace[0]["descriptor"]["score"],actual.trace[0]["descriptor"]["score"]),
                (expected.outputs[0][-1],actual.outputs[0][-1]),(objective(expected),objective(actual))):
        equivalent(vjp(a,roots),vjp(c,roots))
    local=vjp(actual.trace[0]["descriptor"]["score"],roots)
    assert local["(0, 0).keys"] is None and local["(0, 0).values"] is None
    assert local["(0, 0).summaries"] is not None and local["(0, 0).weights"] is not None


def test_native_multihead_partial_rope_distinct_descriptor_width(dtype):
    old,_,xs,stop=setup(dtype,"fc")
    config=replace(old.memories[0],width=4,heads=2,summary_width=3,memory_read="qry",
                   full_frequencies=((),(.31,)),read_frequencies=(.23,),distance_slopes=(.1,.2))
    b=bind(old.graph,memories=config,regions=old.regions,dtype=dtype)
    xs=[External(x.batch,x.port,x.position,x.time,torch.cat((x.value,x.value[1:]*.7))) for x in xs]
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    actual=Native(b.graph,b.model,algorithm="frontier",packed=True).run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual)
    equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(actual),variables(b,xs)))
