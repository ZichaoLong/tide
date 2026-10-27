"""Weighted KV integration: independent schedules, roots, updates and exact cuts.

Local mathematics is anchored separately in test_weighted_kv_formulas and
test_weighted_allocation. These tests establish execution/ownership equivalence.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.greedy import run as greedy
from tidegraph.reference import run as reference
from tidegraph.settle_ports import run_ports
from tidegraph.streaming import run as streaming
from tidegraph.weighted import KVConfig, RegionalConfig, bind, fully_connected, hb_lattice
from isolated_cases import vjp


def setup(dtype,family,read_mode="+"):
    spec = None
    if family == "fc":
        spec = fully_connected(2,2)
    elif family == "hb":
        spec = hb_lattice(2,1,2,offsets=((0,1),(0,)))
    if spec:
        topology,stop = spec.graph,2*spec.stride
        times = (1,spec.stride+1)
    else:
        edges = (Edge(0,1,1),Edge(0,1,3),Edge(1,2,2))
        if family == "pdg": edges += (Edge(2,0,1),)
        topology = Graph((Node(0),Node(0),Node(1)),edges,(Region(2),Region(1)),(0,1),(1,2))
        stop,times = 8,(0,3)
    rc = tuple(RegionalConfig(k=len(r.members),gamma_max=1.3,read_mode=read_mode) for r in topology.region_layouts)
    b = bind(topology,memories=KVConfig(width=2,summary_width=3,adapter_rank=1,heads=1,ffn_width=3,
                                       window=7,full_frequencies=((.2,),)),regions=rc,
             alpha_agg=.2,dtype=dtype)
    xs = [External(0,p,k,t,torch.tensor([.35+.1*p,.3+.2*k,-.4+.1*p],dtype=dtype,requires_grad=True))
          for p in range(len(topology.inputs)) for k,t in enumerate(times)]
    return b,None if spec is None else replace(spec,graph=b.graph),xs,stop


def variables(b,xs):
    return {**{n:p for n,p in b.model.named_parameters() if p.requires_grad},
            **{f"input.{i}":x.value for i,x in enumerate(xs)}}


@pytest.mark.parametrize("family",["fc","hb","dag","pdg"])
def test_weighted_complete_records_and_independent_roots(dtype,family):
    b,spec,xs,stop = setup(dtype,family)
    q = Continuation(b.graph.identity,1)
    expected = reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    schedules = [streaming,greedy] if family == "pdg" else [streaming,frontier,greedy]
    for schedule in schedules:
        actual = schedule(b.graph,b.model,q,xs,stop,sealed_until=stop)
        equivalent(expected,actual)
        roots = variables(b,xs)
        equivalent(vjp(objective(expected),roots),vjp(objective(actual),roots))
        for key in ("w","mu","score","evidence"):
            equivalent(vjp(expected.trace[0]["descriptor"][key],roots),
                       vjp(actual.trace[0]["descriptor"][key],roots))
        # Root in one output must preserve disconnected later/private owners.
        equivalent(vjp(expected.outputs[0][-1],roots),vjp(actual.outputs[0][-1],roots))
    if spec:
        direct = run_ports(spec,b.model,q,xs,stop,sealed_until=stop)
        equivalent(expected,direct)
        equivalent(vjp(objective(expected),variables(b,xs)),vjp(objective(direct),variables(b,xs)))
    # Region bases share Parameters; private adapters and states do not.
    assert b.model.nodes[0].kernel.base is b.model.nodes[1].kernel.base
    assert b.model.nodes[0].kernel.adapter_u is not b.model.nodes[1].kernel.adapter_u


@pytest.mark.parametrize("family",["hb","pdg"])
def test_every_cut_pending_and_checkpoint(dtype,family,tmp_path):
    b,spec,xs,stop = setup(dtype,family)
    q = Continuation(b.graph.identity,1)
    expected = reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    for cut in range(1,stop):
        first = greedy(b.graph,b.model,q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
        rest = [x for x in xs if x.time>=cut]
        continued = greedy(b.graph,b.model,first.continuation,rest,stop,sealed_until=stop)
        equivalent(expected.continuation,continued.continuation)
        equivalent([e for e in expected.trace if e["time"]>=cut],continued.trace)
        equivalent(vjp(objective(continued),variables(b,xs)),vjp(objective(
            reference(b.graph,b.model,first.continuation,rest,stop,sealed_until=stop)),variables(b,xs)))
        path = tmp_path/f"cut-{cut}.pt"
        save(path,b.graph,b.model,first.continuation)
        loaded = load(path,b.graph,b.model)
        equivalent(first.continuation.detach(),loaded)
        restored = greedy(b.graph,b.model,loaded,rest,stop,sealed_until=stop)
        equivalent(continued,restored)


def test_full_first_order_directional_derivative(dtype):
    if dtype != torch.float64:
        return
    b,_,xs,stop = setup(dtype,"fc")
    parameter = b.model.nodes[0].read_program.evidence_weight
    def evaluate():
        r = greedy(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop)
        return sum(value.square().sum() for _,_,_,value in r.outputs)
    direction = torch.linspace(-.2,.3,parameter.numel(),dtype=dtype).reshape_as(parameter)
    expected = (torch.autograd.grad(evaluate(),parameter)[0]*direction).sum()
    original = parameter.detach().clone()
    with torch.no_grad(): parameter.copy_(original+1e-5*direction)
    plus = evaluate().detach()
    with torch.no_grad(): parameter.copy_(original-1e-5*direction)
    minus = evaluate().detach()
    with torch.no_grad(): parameter.copy_(original)
    torch.testing.assert_close((plus-minus)/2e-5,expected,rtol=2e-5,atol=2e-7)


@pytest.mark.parametrize("family",["fc","hb"])
def test_multistep_adamw_shared_parameters_and_resume(dtype,family,tmp_path):
    a,_,xa,stop = setup(dtype,family)
    b,_,xb,_ = setup(dtype,family)
    opts = [torch.optim.AdamW([p for p in item.model.parameters() if p.requires_grad],lr=.002)
            for item in (a,b)]
    for _ in range(3):
        results = []
        for item,xs,opt,schedule in zip((a,b),(xa,xb),opts,(reference,greedy)):
            opt.zero_grad(set_to_none=True)
            result = schedule(item.graph,item.model,Continuation(item.graph.identity,1),xs,stop,sealed_until=stop)
            objective(result).backward(); opt.step(); results.append(result)
        equivalent(results[0],results[1])
        equivalent(a.model.state_dict(),b.model.state_dict())
        for pa,pb in zip(a.model.parameters(),b.model.parameters()):
            equivalent(pa.grad,pb.grad)
    path = tmp_path/"trained.pt"
    save(path,b.graph,b.model,results[1].continuation.detach(),opts[1])
    c,_,xc,_ = setup(dtype,family)
    oc = torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad],lr=.002)
    restored = load(path,c.graph,c.model,oc)
    equivalent(results[1].continuation.detach(),restored)
    equivalent(b.model.state_dict(),c.model.state_dict())
    # Another update verifies moments/step count and shared owner restoration.
    for item,xs,opt in ((b,xb,opts[1]),(c,xc,oc)):
        opt.zero_grad(set_to_none=True)
        result = greedy(item.graph,item.model,Continuation(item.graph.identity,1),xs,stop,sealed_until=stop)
        objective(result).backward(); opt.step()
    equivalent(b.model.state_dict(),c.model.state_dict())


def test_topology_counts_and_inapplicable_configuration(dtype):
    hb = hb_lattice(2,3,64,offsets=((0,1),)*6)
    assert (len(hb.graph.regions),len(hb.graph.nodes),len(hb.graph.edges)) == (22,1408,3584)
    assert hb.ranks.count(4) == 8
    with pytest.raises(ValueError,match="offsets"):
        hb_lattice(2,1,2,offsets=((0,0),(0,)))
    with pytest.raises(ValueError,match="context"):
        RegionalConfig(total_fraction=.5)
    b,_,_,_ = setup(dtype,"fc")
    with torch.no_grad(): b.model.edge_scale[0].fill_(.5)
    with pytest.raises(ValueError,match="unit scales"):
        greedy(b.graph,b.model,Continuation(b.graph.identity,1),[],1,sealed_until=1)
    b,_,_,_ = setup(dtype,"fc")
    b.model.nodes[0].kernel.config = replace(b.memories[0],window=13)
    with pytest.raises(ValueError,match="new graph binding"):
        greedy(b.graph,b.model,Continuation(b.graph.identity,1),[],1,sealed_until=1)
