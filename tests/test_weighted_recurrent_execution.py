"""Recurrent-memory §4 complete records, independent schedules and exact cuts.

Formula independence is established by test_weighted_recurrent_formulas. Native
runs C++ local functions and schedules; no Python formula callback is installed.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.greedy import run as greedy
from tidegraph.reference import run as reference
from tidegraph.settle_ports import run_ports
from tidegraph.native import Native
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind, bind_layered, hb_lattice, fully_connected
from isolated_cases import vjp
from test_weighted_execution import setup, variables

FAMILIES = ("affine","delta","gated_delta","ssd")


def model(dtype,family,mode="+",topology="pdg"):
    old,spec,xs,stop = setup(dtype,topology,read_mode=mode)
    c = RecurrentConfig(memory_family=family,width=2,summary_width=3,key_width=2,value_width=3,ffn_width=3,
        memory_read="state",**({} if family in {"affine","delta"} else
                              dict(heads=2,conv_size=3,conv_stride=2,matrix_rates=(.07,0.))))
    b = bind(old.graph,memories=c,regions=old.regions,alpha_agg=.2,dtype=dtype)
    # Deliberate parameter sharing is independent of state ownership.
    b.model.nodes[1].kernel.summary = b.model.nodes[0].kernel.summary
    return b,None if spec is None else replace(spec,graph=b.graph),xs,stop


def check_records(b,xs,stop,mode,native):
    q = Continuation(b.graph.identity,1)
    expected = reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    runner = Native(b.graph,b.model,algorithm="greedy",packed=True).run if native else lambda *args,**kw: greedy(b.graph,b.model,*args,**kw)
    actual = runner(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual)
    roots = variables(b,xs)
    equivalent(vjp(objective(expected),roots),vjp(objective(actual),roots))
    for name in ("w","mu","score","evidence"):
        equivalent(vjp(expected.trace[0]["descriptor"][name],roots),vjp(actual.trace[0]["descriptor"][name],roots))
    equivalent(vjp(expected.outputs[0][-1],roots),vjp(actual.outputs[0][-1],roots))
    # Warm exact raw state; int64 slots are not differentiable state roots.
    first = runner(q,[x for x in xs if x.time<3],3,sealed_until=3)
    warm = first.continuation.detach(); state_roots = {}
    for owner,state in warm.states.items():
        for name,value in tuple(state.slots.items()):
            if value.is_floating_point():
                state.slots[name] = value.detach().requires_grad_()
                state_roots[f"{owner}.{name}"] = state.slots[name]
    tail = [x for x in xs if x.time>=3]
    a = reference(b.graph,b.model,warm,tail,stop,sealed_until=stop)
    c = runner(warm,tail,stop,sealed_until=stop)
    equivalent(a,c)
    for va,vc in ((objective(a),objective(c)),(a.trace[0]["descriptor"]["score"],c.trace[0]["descriptor"]["score"]),
                  (a.outputs[0][-1],c.outputs[0][-1])):
        equivalent(vjp(va,state_roots),vjp(vc,state_roots))
    if mode=="0":
        assert all(x is None for x in vjp(c.trace[0]["descriptor"]["score"],state_roots).values())


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("mode",["0","-","+"])
def test_python_recurrent_complete_records_and_state_roots(dtype,family,mode):
    b,_,xs,stop=model(dtype,family,mode)
    check_records(b,xs,stop,mode,False)


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("mode",["0","-","+"])
def test_native_recurrent_complete_records_and_state_roots(dtype,family,mode):
    b,_,xs,stop=model(dtype,family,mode)
    check_records(b,xs,stop,mode,True)


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("native",[False,True],ids=["python","native"])
def test_recurrent_every_cut_and_checkpoint(dtype,family,native,tmp_path):
    b,_,xs,stop=model(dtype,family)
    q=Continuation(b.graph.identity,1)
    runner=Native(b.graph,b.model,algorithm="greedy",packed=True).run if native else lambda *args,**kw:greedy(b.graph,b.model,*args,**kw)
    expected=runner(q,xs,stop,sealed_until=stop)
    for cut in range(1,stop):
        head=runner(q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
        tail=[x for x in xs if x.time>=cut]
        continued=runner(head.continuation,tail,stop,sealed_until=stop)
        equivalent(expected.continuation,continued.continuation)
        equivalent([e for e in expected.trace if e["time"]>=cut],continued.trace)
        joined = sum(x[-1].square().sum() for x in head.outputs+continued.outputs)
        whole = sum(x[-1].square().sum() for x in expected.outputs)
        equivalent(vjp(joined,variables(b,xs)),vjp(whole,variables(b,xs)))
        path=tmp_path/f"{cut}.pt";save(path,b.graph,b.model,head.continuation)
        other,_,_,_=model(dtype,family)
        restored=load(path,other.graph,other.model)
        equivalent(head.continuation.detach(),restored)
        other_run=Native(other.graph,other.model,algorithm="streaming").run if native else lambda *args,**kw:greedy(other.graph,other.model,*args,**kw)
        equivalent(continued,other_run(restored,tail,stop,sealed_until=stop))


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("native",[False,True],ids=["python","native"])
def test_recurrent_multistep_optimizer_and_restore(dtype,family,native,tmp_path):
    a,_,xa,stop=model(dtype,family,topology="hb")
    b,_,xb,_=model(dtype,family,topology="hb")
    runner=Native(b.graph,b.model,algorithm="frontier",packed=True).run if native else lambda *args,**kw:frontier(b.graph,b.model,*args,**kw)
    opts=[torch.optim.AdamW([p for p in item.model.parameters() if p.requires_grad],lr=.001) for item in (a,b)]
    for _ in range(3):
        opts[0].zero_grad(set_to_none=True);opts[1].zero_grad(set_to_none=True)
        expected=reference(a.graph,a.model,Continuation(a.graph.identity,1),xa,stop,sealed_until=stop)
        actual=runner(Continuation(b.graph.identity,1),xb,stop,sealed_until=stop)
        equivalent(expected,actual)
        objective(expected).backward();objective(actual).backward()
        for pa,pb in zip(a.model.parameters(),b.model.parameters()):equivalent(pa.grad,pb.grad)
        opts[0].step();opts[1].step();equivalent(a.model.state_dict(),b.model.state_dict())
    save(tmp_path/"trained.pt",b.graph,b.model,actual.continuation.detach(),opts[1])
    c,_,xc,_=model(dtype,family,topology="hb")
    opt=torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad],lr=.001)
    restored=load(tmp_path/"trained.pt",c.graph,c.model,opt)
    equivalent(actual.continuation.detach(),restored)
    for item,inputs,optimizer in ((b,xb,opts[1]),(c,xc,opt)):
        optimizer.zero_grad(set_to_none=True)
        result=reference(item.graph,item.model,Continuation(item.graph.identity,1),inputs,stop,sealed_until=stop)
        objective(result).backward();optimizer.step()
    equivalent(b.model.state_dict(),c.model.state_dict())


@pytest.mark.parametrize("topology",["fc","hb"])
def test_heterogeneous_memories_ports_native_and_layered_binding(dtype,topology):
    spec=fully_connected(3,2) if topology=="fc" else hb_lattice(2,1,2,offsets=((0,1),(0,)))
    options=[KVConfig(width=2,summary_width=3,ffn_width=3)]+[
        RecurrentConfig(memory_family=f,width=2,summary_width=3,ffn_width=3,
                        **({} if f in {"affine","delta"} else dict(conv_size=3))) for f in FAMILIES]
    memories=tuple(options[v%5] for v in range(len(spec.graph.nodes)))
    spec,b=bind_layered(spec,memory=memories,dtype=dtype)
    xs=[External(0,p,i,1+i*spec.stride,torch.tensor([.7,.3+.1*i,-.2],dtype=dtype,requires_grad=True))
        for p in range(len(spec.graph.inputs)) for i in range(2)]
    q=Continuation(b.graph.identity,1);stop=2*spec.stride
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    for actual in (frontier(b.graph,b.model,q,xs,stop,sealed_until=stop),run_ports(spec,b.model,q,xs,stop,sealed_until=stop),
                   Native(b.graph,b.model,algorithm="frontier",packed=True).run(q,xs,stop,sealed_until=stop)):
        equivalent(expected,actual)
        equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(actual),variables(b,xs)))
    for c in b.memories:
        if isinstance(c,RecurrentConfig) and not c.basic:assert c.conv_stride==spec.stride


@pytest.mark.parametrize("family",FAMILIES)
def test_native_recurrent_non_aligned_dag_and_no_grad(dtype,family):
    b,_,xs,stop=model(dtype,family,topology="dag")
    q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    for prefill in (False,True):
        candidate=Native(b.graph,b.model,algorithm="frontier",prefill=prefill,packed=True)
        actual=candidate.run(q,xs,stop,sealed_until=stop)
        equivalent(expected,actual)
        equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(actual),variables(b,xs)))
        with torch.no_grad():
            inference=candidate.run(q,xs,stop,sealed_until=stop)
        equivalent(expected,inference)
        assert all(not value.requires_grad for *_,value in inference.outputs)
