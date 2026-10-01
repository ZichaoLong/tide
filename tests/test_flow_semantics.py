"""Independent Python region order and native encoding on active packet graphs."""
import importlib.util
from pathlib import Path
import pytest
import torch
from tidegraph import Continuation, Edge, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.ops import Model
from tidegraph.native import Native
from tidegraph.reference import run as streaming
from tidegraph.frontier import run as frontier
from tidegraph.settle import SettleGraph, run as settle

spec=importlib.util.spec_from_file_location('flow_topology',Path(__file__).parents[1]/'scripts/flow_topology.py')
flow=importlib.util.module_from_spec(spec);spec.loader.exec_module(flow)


def compare(a,b,dtype):
    # FP64 Read over FP32 state inherits the FP32 payload error, as in the
    # standalone consumer gate. Discrete identities still compare exactly.
    equivalent(a,b,atol=1e-10 if dtype==torch.float64 else 1e-6,
               rtol=1e-8 if dtype==torch.float64 else 1e-5)


def vjp(root,leaves,direction):
    if not root.requires_grad:return dict.fromkeys(leaves)
    index=torch.arange(root.numel()).reshape(root.shape)
    u=((index*(direction+1)+1)%(7+direction*4)-(3+direction*2)).to(root.dtype)/16
    if direction==2:u=torch.zeros_like(u)
    return dict(zip(leaves,torch.autograd.grad(root,list(leaves.values()),u,allow_unused=True,retain_graph=True)))


def loss(result):
    return torch.stack([x for _,_,_,x in result.outputs]).square().mean()


def fixture(dtype,memory,clear):
    packet=flow.make_packet(graph=flow.ranked_graph(layers=3,region_width=2,fanout=2,skip=1,local_span=2,cross_every=0),
                            memory=memory,width=4,batch=2,tokens=3,vocab=17,clear=clear)
    body=packet['graph'];profile='lh-add-repeat-v1' if memory=='add' else 'lh-fiber-attention-all-softmax-repeat-v1'
    # Parallel physical wires must stay distinct even though they share endpoints.
    edges=[Edge(*e) for e in body['edges']];edges.append(edges[0])
    graph=Graph(tuple(Node(r,clear=clear,memory=profile,full='lh-silu-rms-v1',
                           aggregation='all_softmax' if memory=='add' else 'sum',
                           readout='norm-fp64-v1',emission='slot_affine',query_heads=4,kv_heads=4)
                      for r in body['node_regions']),tuple(edges),tuple(Region(1) for _ in body['ranks']),
                tuple(body['inputs']),tuple(body['outputs']))
    spec=SettleGraph(graph,tuple(body['ranks']));model=Model(graph,width=4,dtype=dtype)
    x=(torch.arange(24,dtype=dtype).reshape(2,3,4)/32-.2).requires_grad_()
    return spec,model,x


def execute(spec,model,x,algorithm):
    q=Continuation(spec.graph.identity,len(x))
    if algorithm=='python-settle':return settle(spec,model,q,x,mode='hard')
    g,m=spec.embed(model);eq=spec.embed_initial(q,g);xs=spec.external(x,encoded=True);stop=x.size(1)*spec.stride
    if algorithm.startswith('python'):
        fn=streaming if algorithm=='python-streaming' else frontier
        result=fn(g,m,eq,xs,stop,sealed_until=stop,mode='hard')
    else:result=Native(g,m,algorithm=algorithm.split('-')[1],packed=True,workers=2,mode='hard').run(eq,xs,stop,sealed_until=stop)
    return spec.project(result)


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('clear',[False,True])
@pytest.mark.parametrize('algorithm',['python-streaming','python-frontier','native-streaming','native-frontier'])
def test_ranked_packet_mapping_observables_and_isolated_roots(dtype,memory,clear,algorithm):
    spec,m,x=fixture(dtype,memory,clear)
    expected=execute(spec,m,x,'python-settle');actual=execute(spec,m,x,algorithm)
    compare(expected,actual,dtype)
    leaves=dict(m.named_parameters())|{'input':x}
    def roots(result):
        e=next(e for e in result.trace if e['active'])
        state=next(result.continuation.states[k] for k in sorted(result.continuation.states)
                   if result.continuation.states[k].value.requires_grad)
        return [result.outputs[0][3],e['content'],e['proposal'],e['descriptor'],e['control'],e['full'],
                state.value,*(state.slots[k] for k in sorted(state.slots))]
    for a,b in zip(roots(expected),roots(actual)):
        for direction in range(3):compare(vjp(a,leaves,direction),vjp(b,leaves,direction),dtype)


@pytest.mark.parametrize('memory',['add','attention'])
@pytest.mark.parametrize('algorithm',['python-frontier','native-streaming','native-frontier'])
@pytest.mark.parametrize('optimizer',['sgd','adamw'])
def test_ranked_packet_three_independent_optimizer_updates(dtype,memory,algorithm,optimizer):
    fixtures=[fixture(dtype,memory,True) for _ in range(2)]
    def make(model):
        if optimizer=='sgd':return torch.optim.SGD(model.parameters(),lr=1e-4,momentum=.9,weight_decay=.01)
        return torch.optim.AdamW(model.parameters(),lr=1e-4,eps=1e-5,weight_decay=.01)
    optimizers=[make(m) for _,m,_ in fixtures]
    for step in range(3):
        outputs=[]
        for (s,m,x),opt,alg in zip(fixtures,optimizers,['python-settle',algorithm]):
            opt.zero_grad(set_to_none=True)
            out=execute(s,m,x+step/64,alg);loss(out).backward();outputs.append(out)
        compare(*outputs,dtype)
        for a,b in zip(fixtures[0][1].parameters(),fixtures[1][1].parameters()):compare(a.grad,b.grad,dtype)
        for opt in optimizers:opt.step()
        for a,b in zip(fixtures[0][1].parameters(),fixtures[1][1].parameters()):compare(a,b,dtype)
