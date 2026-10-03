"""Independent exhaustive choices validate static traffic admission inputs."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from flow_topology import ranked_graph,graph_counts
from flow_protocol import make_continuous_packet
from tools.online_bench.traffic_bounds import traffic_bounds
from tools.online_bench.eager_placement import owner_indices


def packet(delayed=False, parallel=False, budget=1):
    g=ranked_graph(layers=3,region_width=2,fanout=2,local_span=2,cross_every=0)
    if delayed:
        g["edges"][0][2]+=2;g["kind"]="timed-local"
    if parallel:
        g["edges"].append(g["edges"][0].copy())
    g["counts"]=graph_counts(g)
    return make_continuous_packet(graph=g,width=4,batch=2,tokens=2,vocab=7,memory="attention",budget=budget,clear=False)


def enumerate_choices(p):
    """Independent count-only execution of every legal top-k choice, one token.

    This oracle is a test, never an input to a runtime or capacity planner. Wire
    multiplicity remains in the inbox; candidates coalesce only by node/time.
    """
    g,c=p["graph"],p["workload"];n=g["nodes"]
    aligned=g["kind"]=="ranked-local";region=g["node_regions"]
    rank=[g["ranks"][r] for r in region]
    inbox={}
    for node in g["inputs"]:
        inbox.setdefault((rank[node] if aligned else 1,region[node]),[]).append(node)
    pending=[(inbox,[0]*n,[0]*n,[0]*n,[])]
    terminals=[]
    while pending:
        box,atoms,events,emissions,outputs=pending.pop()
        if not box:
            terminals.append((atoms,events,emissions,outputs));continue
        when,r=min(box);arrived=box[when,r]
        box={key:list(value) for key,value in box.items() if key!=(when,r)}
        candidates=sorted(set(arrived));atoms=atoms.copy();events=events.copy()
        for node in arrived:atoms[node]+=1
        for node in candidates:events[node]+=1
        for selected in combinations(candidates,min(c["budget"],len(candidates))):
            next_box={key:list(value) for key,value in box.items()};sent=emissions.copy();out=list(outputs)
            for node in selected:
                for a,b,delay in g["edges"]:
                    if a==node:
                        next_box.setdefault((when+delay,region[b]),[]).append(b);sent[node]+=1
                for node_out in g["outputs"]:
                    if node_out==node:
                        out.append(when+(max(g["ranks"])+1-rank[node] if aligned else 1))
            pending.append((next_box,atoms,events,sent,out))
        assert len(pending)+len(terminals)<10000
    return terminals


@pytest.mark.parametrize("delayed",[False,True])
@pytest.mark.parametrize("parallel",[False,True])
@pytest.mark.parametrize("budget",[1,2])
def test_exhaustive_legal_choices(delayed,parallel,budget):
    p=packet(delayed,parallel,budget);owners=[0,1,0,1,0,1,0,0]
    bound=traffic_bounds(p,owners);cases=enumerate_choices(p)
    assert cases
    if budget==1:assert len(cases)>1
    for atoms,events,emissions,outputs in cases:
        for actual,limit in [(atoms,bound["node_atom_factors"]),(events,bound["node_event_factors"])]:
            assert all(a<=b for a,b in zip(actual,limit))
        for d in (0,1):
            for actual,key in [(atoms,"owner_body_atom_factors"),(events,"owner_body_event_factors"),(emissions,"owner_body_emission_factors")]:
                assert sum(a for i,a in enumerate(actual) if owners[i]==d)<=bound[key][d]
        assert len(outputs)<=bound["output_atom_factor"]
        assert len(set(outputs))<=bound["output_frame_factor"]
        assert max(outputs,default=0)<=bound["maximum_position_offset"]<p["workload"]["stride"]


def test_parallel_wire_multiplicity_and_owner_choice():
    p=packet();owners=[0,0,0,0,0,0,0,0]
    plain=traffic_bounds(p,owners);duplicate=traffic_bounds(packet(parallel=True),owners)
    assert duplicate["node_atom_factors"][2]==plain["node_atom_factors"][2]+1
    # Top-1 emits at most one source's two adjacent wires and one skip wire;
    # the first two nodes still each receive a boundary input.
    assert plain["owner_body_atom_factors"]==[7]
    assert plain["output_atom_factor"]==1 and plain["output_frame_factor"]==1
    assert plain["owner_body_atom_factors"][0]<sum(plain["node_atom_factors"])


def test_original_width_shape_only_and_nonuniform_ranks():
    g=ranked_graph(rank_gap=3)
    p=make_continuous_packet(graph=g,width=2048,batch=512,tokens=12,vocab=50304,memory="attention")
    owners,_=owner_indices(p,11)
    bound=traffic_bounds(p,owners)
    assert bound["region_frame_factors"]==[1]*15
    assert bound["node_offset_intervals"]==[[g["ranks"][r]]*2 for r in g["node_regions"]]
    assert bound["output_frame_factor"]==1 and bound["maximum_position_offset"]==44
    for d in range(11):
        assert bound["owner_body_atom_factors"][d]<=sum(v for i,v in enumerate(bound["node_atom_factors"]) if owners[i]==d)


def test_position_seal_feedback_and_owner_rejection():
    p=packet();owners=[0]*8
    with pytest.raises(ValueError,match="owners"):
        traffic_bounds(p,[0]*7)
    p["workload"]["stride"]=3
    with pytest.raises(ValueError,match="sealed"):
        traffic_bounds(p,owners)
    p=packet();p["graph"]["edges"].append([4,0,1]);p["graph"]["kind"]="timed-local"
    with pytest.raises(ValueError,match="feedback"):
        traffic_bounds(p,owners)
