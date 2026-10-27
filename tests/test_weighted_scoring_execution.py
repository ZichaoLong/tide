"""Representative formal combinations, independent schedules and shared roots."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save,load
from tidegraph.compare import equivalent,objective
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.weighted import KVConfig,RecurrentConfig,RegionalConfig,bind
from tidegraph.weighted.config import ATTENTION,REGIONAL,ALLOCATED
from test_weighted_execution import setup,variables
from isolated_cases import vjp


# Representative dependency combinations, not all switches' Cartesian product.
CASES=[
    ("regional_total","cat","att","content_stats","cat","received"),
    ("regional_threshold","match","ind",None,"cat",None),
    ("regional_threshold","cat","ind",None,"grouped",None),
    ("regional_total","match","att","content","grouped","received"),
    ("direct_attention","match","att","content_stats",None,"mu"),
    ("pooled_attention","cat","att","content",None,"received"),
    ("hard_selection","match","att","content",None,"mu"),
    ("proportional","match","ind",None,None,None),
    ("regional_total_zero","match","att","content_stats","grouped","received"),
    ("regional_threshold_hard","cat","att","content_stats","grouped","received"),
    ("unit_attention","match","att","content_stats",None,"received"),
    ("constrained","match","att","content",None,"mu")]


def configured(dtype,index,family=None,read_mode="+",tie=False):
    old,_,xs,stop=setup(dtype,family or ("hb" if index%2 else "fc"))
    mode,local,region,key,evidence,query=CASES[index]
    regions=[]
    for layout in old.graph.region_layouts:
        fields=dict(selection_mode=mode,local_score=local,region_score=region,region_key=key,
                    region_evidence=evidence,query_weight=query,read_mode=read_mode,gamma_max=1.6,
                    hidden_width=3,selection_width=2)
        if mode not in ATTENTION:fields["k"]=len(layout.members)
        if mode not in REGIONAL:fields["send_gain"]="context"
        if mode in ALLOCATED:fields["regularizer"]="bern"
        regions.append(RegionalConfig(**fields))
    choices=[KVConfig(width=2,summary_width=2,adapter_rank=1,ffn_width=3)]
    choices += [RecurrentConfig(memory_family=f,width=2,summary_width=2,key_width=2,value_width=2,
                               ffn_width=3,conv_size=2 if f in {"gated_delta","ssd"} else 1)
                for f in ("affine","delta","gated_delta","ssd")]
    memories=tuple(choices[(v+index)%len(choices)] for v in range(len(old.graph.nodes)))
    b=bind(old.graph,memories=memories,regions=tuple(regions),dtype=dtype,alpha_agg=.2)
    if tie:
        for node in b.model.nodes:
            read=node.read_program
            read.key=read.score.encoders["mem"]
            read.evidence.encoders["mem"]=read.score.encoders["mem"]
    return b,xs,stop


@pytest.mark.parametrize("case",range(len(CASES)))
def test_scoring_combinations_records_and_independent_roots(dtype,case):
    b,xs,stop=configured(dtype,case);q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    actual=Native(b.graph,b.model,algorithm="frontier",packed=True).run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual);roots=variables(b,xs)
    equivalent(vjp(objective(expected),roots),vjp(objective(actual),roots))
    equivalent(vjp(expected.outputs[0][-1],roots),vjp(actual.outputs[0][-1],roots))
    for name in expected.trace[0]["descriptor"]:
        equivalent(vjp(expected.trace[0]["descriptor"][name],roots),vjp(actual.trace[0]["descriptor"][name],roots))
    expected_fields=3+(CASES[case][0] in REGIONAL)+2*(CASES[case][2]=="att")
    assert len(actual.trace[0]["descriptor"])==expected_fields


@pytest.mark.parametrize("mode",["0","-","+"])
def test_attention_read_snapshots_pdg_and_every_cut(dtype,mode):
    b,xs,stop=configured(dtype,3,"pdg",mode);q=Continuation(b.graph.identity,1)
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    native=Native(b.graph,b.model,algorithm="greedy",packed=True);actual=native.run(q,xs,stop,sealed_until=stop)
    equivalent(expected,actual)
    for cut in range(1,stop):
        head=native.run(q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
        tail=[x for x in xs if x.time>=cut]
        continued=native.run(head.continuation,tail,stop,sealed_until=stop)
        equivalent(expected.continuation,continued.continuation)
        oracle=reference(b.graph,b.model,head.continuation,tail,stop,sealed_until=stop)
        equivalent(oracle,continued);equivalent(vjp(objective(oracle),variables(b,xs)),vjp(objective(continued),variables(b,xs)))


@pytest.mark.parametrize("case",[0,1,3,4,5,8,11])
def test_scoring_optimizer_shared_parameters_checkpoint(dtype,case,tmp_path):
    a,xa,stop=configured(dtype,case,tie=case==3);b,xb,_=configured(dtype,case,tie=case==3)
    native=Native(b.graph,b.model,algorithm="greedy",packed=True)
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
    save(tmp_path/"scoring.pt",b.graph,b.model,head.continuation,ob)
    c,_,_=configured(dtype,case,tie=case==3)
    oc=torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad],lr=.001)
    restored=load(tmp_path/"scoring.pt",c.graph,c.model,oc)
    equivalent(head.continuation.detach(),restored);equivalent(ob.state_dict(),oc.state_dict())
    tail=[x for x in xb if x.time>=3]
    expected=native.run(head.continuation.detach(),tail,stop,sealed_until=stop)
    actual=Native(c.graph,c.model,algorithm="streaming").run(restored,tail,stop,sealed_until=stop)
    equivalent(expected,actual)
    ob.zero_grad(set_to_none=True);oc.zero_grad(set_to_none=True)
    objective(expected).backward();objective(actual).backward();ob.step();oc.step()
    equivalent(b.model.state_dict(),c.model.state_dict());equivalent(ob.state_dict(),oc.state_dict())
    if case==3:
        for node in c.model.nodes:
            read=node.read_program
            assert read.key is read.score.encoders["mem"] is read.evidence.encoders["mem"]


def test_scoring_full_parameter_and_input_directional_derivative():
    b,xs,stop=configured(torch.float64,3,tie=True)
    roots=tuple(variables(b,xs).values());original=[p.detach().clone() for p in roots]
    directions=[torch.linspace(-.1,.15,p.numel(),dtype=p.dtype).reshape_as(p) for p in roots]
    def evaluate():return objective(reference(b.graph,b.model,Continuation(b.graph.identity,1),xs,stop,sealed_until=stop))
    gradients=torch.autograd.grad(evaluate(),roots,allow_unused=True)
    expected=sum((g*d).sum() for g,d in zip(gradients,directions) if g is not None)
    def displace(amount):
        with torch.no_grad():
            for p,value,d in zip(roots,original,directions):p.copy_(value+amount*d)
    displace(1e-5);plus=evaluate().detach();displace(-1e-5);minus=evaluate().detach();displace(0.)
    torch.testing.assert_close((plus-minus)/2e-5,expected,atol=2e-7,rtol=2e-5)
