"""0770593 architecture §2–4 / KV §1–4 / allocation §2.5 formula anchors."""
from dataclasses import replace
import math
import pytest
import torch
from tidegraph import Atom, Continuation, External, Graph, Node, Region
from tidegraph.aggregate import request as aggregate_request
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.reference import run
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, Payload, Received, RegionalConfig, bind


def close(a,b):
    torch.testing.assert_close(a,b,rtol=1e-5 if a.dtype == torch.float32 else 1e-9,
                               atol=1e-6 if a.dtype == torch.float32 else 1e-11)


def fixture(dtype, nodes=1, **kw):
    graph = Graph(tuple(Node(0) for _ in range(nodes)),(),(Region(nodes),),tuple(range(nodes)),tuple(range(nodes)))
    return bind(graph,dtype=dtype,**kw)


def test_kv_write_mass_window_zero_and_hand_full(dtype):
    cfg = KVConfig(width=1,summary_width=1,adapter_rank=1,heads=1,ffn_width=1,
                   epsilon=1.,window=5,beta_short=math.log(2),alpha_ff=0.)
    b = fixture(dtype,memories=cfg)
    w = b.model.nodes[0]; memory = w.kernel; full = w.full_program.content
    with torch.no_grad():
        memory.base.copy_(torch.tensor([[2.],[3.]],dtype=dtype))
        memory.adapter_u.zero_(); memory.summary.fill_(4)
        full.query.zero_(); full.out.fill_(1)
    old = memory.initial(w)
    clock = 2**54+7
    first = memory.step(w,old,Content(torch.tensor([.5,1.],dtype=dtype)),clock)
    close(first.slots["keys"], torch.tensor([[math.sqrt(2)]],dtype=dtype))
    close(first.slots["values"], torch.tensor([[3/math.sqrt(2)]],dtype=dtype))
    close(first.slots["weights"],torch.tensor([.5],dtype=dtype))
    zero = memory.step(w,first,Content(torch.tensor([0.,100.],dtype=dtype)),clock+20)
    assert zero is first and zero.slots["times"].item() == clock
    second = memory.step(w,first,Content(torch.tensor([.25,-1.],dtype=dtype)),clock+2)
    short,long,ws,wl = memory.mem(second,clock+2,torch.tensor([10.],dtype=dtype))
    close(ws,ws.new_tensor(.375)); close(wl,wl.new_tensor(.75))
    close(short,short.new_tensor([-.5/math.sqrt(2)]))
    close(long,long.new_tensor([1/math.sqrt(2)]))
    # Zero query, weighted value mean (.5*3/sqrt2-.25*3/sqrt2)/.75,
    # multiplied once by .75, then residual u=-1.
    close(full(memory,second,clock+2,torch.tensor([-1.],dtype=dtype)),long.new_tensor([-1+.75/math.sqrt(2)]))
    expired = memory.mem(zero,clock+20,torch.ones(1,dtype=dtype))
    assert all(value.count_nonzero() == 0 for value in expired)
    third = memory.step(w,second,Content(torch.tensor([2.,2.],dtype=dtype)),clock+5)
    assert third.slots["times"].tolist() == [clock+2,clock+5]
    close(third.slots["weights"],torch.tensor([.25,1.],dtype=dtype))


def test_relative_positions_keep_int64_distance_and_raw_keys(dtype):
    cfg = KVConfig(width=2,summary_width=2,adapter_rank=1,ffn_width=2,epsilon=1.,
                   memory_read="qry",full_frequencies=((.3,),),read_frequencies=(.7,),
                   distance_slopes=(.2,),position_scale=3.,alpha_ff=0.)
    b = fixture(dtype,memories=cfg); w = b.model.nodes[0]; m = w.kernel; f = w.full_program.content
    clock = 2**54+13
    old = m.initial(w)
    for time,h in ((clock,[.4,.3,-.5]),(clock+2,[.7,.6,.2])):
        old = m.step(w,old,Content(torch.tensor(h,dtype=dtype)),time)
    u = torch.tensor([.5,-.8],dtype=dtype)
    q = f.query@(u/torch.sqrt(1+u.square().mean()))
    raw = old.slots["keys"]
    rotated = []
    for key,age in zip(raw,(2,0)):
        p = -.3*age/3
        rotated.append(torch.stack((key[0]*math.cos(p)-key[1]*math.sin(p),key[0]*math.sin(p)+key[1]*math.cos(p))))
    logits = torch.stack(rotated)@q/math.sqrt(2)-u.new_tensor([.2*2/3,0.])
    probability = (logits+old.slots["weights"].log()).softmax(0)
    expected = u+f.out@(probability.unsqueeze(-1)*old.slots["values"]).sum(0)
    close(f(m,old,clock+2,u),expected)
    # Translation of all integer times leaves position differences and memories invariant.
    translated = replace(old,last_time=2,slots={**old.slots,"times":old.slots["times"]-clock})
    close(f(m,translated,2,u),expected)
    for a,c in zip(m.mem(old,clock+2,u),m.mem(translated,2,u)):
        close(a,c)


def test_srcclass_multiple_physical_atoms_and_single_strength_projection(dtype):
    g = Graph((Node(0),),(),(Region(1),),(0,0,0),(0,))
    b = bind(g,memories=KVConfig(width=1,summary_width=1,epsilon=1.),
             source_classes=SourceClasses((),(0,0,0)),alpha_agg=.5,dtype=dtype)
    transform = b.model.nodes[0].aggregate_program.transforms["0"]
    with torch.no_grad():
        transform.a.fill_(1); transform.b.fill_(.25); transform.embedding.zero_()
    x = torch.tensor([.2,2.],dtype=dtype,requires_grad=True)
    atoms = [Atom(0,0,0,0,i,0,x) for i in range(3)]
    request = aggregate_request(b.graph,b.model,atoms)
    h = b.model.nodes[0].aggregate_program.step(b.model.nodes[0],request).value
    normalized = 2/math.sqrt(5)
    expected_content = 2+.5*normalized/(1+math.exp(-normalized))
    close(h,h.new_tensor([.6,expected_content]))
    p = Payload(x[0],x[1:]); mass,numeric = p.numerical()
    close(numeric,x.new_tensor([.4]))
    close(torch.autograd.grad(h[0],x)[0],x.new_tensor([3.,0.]))


def test_textbook_evidence_to_total_complete_run(dtype):
    cfg = KVConfig(width=1,summary_width=1,epsilon=1.,alpha_att=0.,alpha_ff=0.)
    rc = RegionalConfig(k=2,hidden_width=2,region_control_width=1,allocation_temperature=2.)
    b = fixture(dtype,nodes=3,memories=cfg,regions=rc)
    with torch.no_grad():
        for v,w in enumerate(b.model.nodes):
            w.read_program.score.vector.zero_(); w.read_program.score.bias.fill_((2.,1.,100.)[v])
            w.read_program.evidence_weight.zero_(); w.read_program.evidence_bias.zero_()
            w.read_program.evidence_weight[0,0] = (-math.sqrt(2/3),math.sqrt(2/3),0.)[v]
        head = b.model.regions[0].head
        head.matrix.zero_(); head.matrix[:,0].copy_(torch.tensor([2*math.log(1.5),-2*math.log(1.5)],dtype=dtype))
        head.hidden_bias.zero_(); head.vector.copy_(torch.tensor([1.,-1.],dtype=dtype)); head.bias.zero_()
    xs = [External(0,v,0,0,torch.tensor(x,dtype=dtype,requires_grad=True))
          for v,x in enumerate(([1.,1.],[1.,-1.],[0.,20.]))]
    r = run(b.graph,b.model,Continuation(b.graph.identity,1),xs,1,sealed_until=1)
    for i,expected in enumerate((.95,.45,0.)):
        close(r.trace[i]["control"],torch.tensor(expected,dtype=dtype))
    assert len(r.outputs) == 2 and r.continuation.states[0,2].observations == 0
    for _,_,port,value in r.outputs:
        close(Payload.decode(value).content,xs[port].value[1:])
    # Independent autograd expression for the complete evidence -> rho -> B -> c path.
    u = torch.stack([x.value[1] for x in xs[:2]])
    normed = u/torch.sqrt(1+u.square())
    z = normed*u.new_tensor([-math.sqrt(2/3),math.sqrt(2/3)])
    z = z/torch.sqrt(1+z.square())
    total = 1+(2*math.log(1.5)*z.mean()).sigmoid()
    expected = (total+.5)/2
    actual_grad = torch.autograd.grad(r.trace[0]["control"],xs[0].value,retain_graph=True)[0][1]
    expected_grad = torch.autograd.grad(expected,xs[0].value)[0][1]
    close(actual_grad,expected_grad)


@pytest.mark.parametrize("mode",["0","-","+"])
def test_read_snapshot_and_full_are_separate(dtype,mode):
    b = fixture(dtype,memories=KVConfig(width=2,summary_width=1),regions=RegionalConfig(read_mode=mode))
    w = b.model.nodes[0]; m = w.kernel
    old = m.step(w,m.initial(w),Content(torch.tensor([.7,.4,-.3],dtype=dtype)),0)
    h = Content(torch.tensor([.4,-.2,.8],dtype=dtype))
    new = m.step(w,old,h,2)
    given = None if mode == "0" else old if mode == "-" else new
    descriptor = w.read_program.step(w,ReadInput(given,2,h))
    chosen = m.initial(w) if mode == "0" else given
    features = w.read_program.features(m,chosen,2,Received.decode(h.value))
    close(descriptor["score"],w.read_program.score(features))
    assert m.mem(chosen,2,h.value[1:])[3].item() == pytest.approx({"0":0.,"-":.7,"+":1.1}[mode],rel=1e-6)
    assert not torch.equal(w.full_program.content(m,new,2,h.value[1:]),w.full_program.content(m,old,2,h.value[1:]))
