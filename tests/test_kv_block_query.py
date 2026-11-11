"""KV block queries: literal prefix inputs, independent roots and causal transpose."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from tidegraph import Graph, Node, Region
from tidegraph.records import State
from tidegraph.content import Content
from tidegraph.readout import ReadInput
from tidegraph.full import FullInput
from tidegraph.compare import equivalent
from tidegraph.weighted import KVConfig, RegionalConfig, bind
from tidegraph.weighted.kv_block_state import Bank, View
from tidegraph.weighted.kv_block_query import dispatch


def fixture(dtype, read='qry', grouped=False):
    c = KVConfig(width=4, summary_width=2, heads=2, memory_read=read)
    rc = RegionalConfig(local_score='match', region_score='att', region_key='content',
        region_evidence='grouped', selection_width=2, region_control_width=3) if grouped else RegionalConfig()
    b = bind(Graph((Node(0),), (), (Region(1),), (0,), (0,)), memories=c, regions=rc, dtype=dtype)
    w = b.model.nodes[0]
    if not grouped:
        w.read_program.score.vector = w.read_program.score.hidden_bias
        w.read_program.score.bias.requires_grad_(False)
    if read == "qry":
        w.kernel.read_v = w.kernel.read_k
    roots = [p for p in w.parameters() if p.requires_grad]
    def leaf(shape):
        t = (torch.rand(shape, dtype=dtype)*.3+.1).requires_grad_()
        roots.append(t)
        return t
    rows = []
    for owner, length in enumerate((4, 3)):
        old = State(torch.zeros(5, dtype=dtype), 2**54+8, 1, dict(times=torch.tensor([2**54+8]),
            keys=leaf((1,4)), values=leaf((1,4)), summaries=leaf((1,2)), weights=leaf((1,))))
        values = tuple(leaf((8,)) for _ in range(length))
        summaries = tuple(leaf((2,)) for _ in range(length))
        contents = tuple(leaf((5,)) for _ in range(length))
        times = tuple([2**54+8]+[2**54+11+3*j for j in range(length)])
        bank = Bank(old, values, summaries, contents, times)
        for j in range(length-1):
            # Independent ordinary concatenations of the original leaf inputs.
            key, value = torch.stack(values[:j+1]).chunk(2, -1)
            fresh = dict(keys=key, values=value, summaries=torch.stack(summaries[:j+1]),
                         weights=torch.stack(contents[:j+1])[:,0].clamp(max=1))
            slots = {name: torch.cat((old.slots[name], t)) for name,t in fresh.items()}
            slots['times'] = torch.tensor(times[:j+2])
            state = State(old.value, times[j+1], j+2, slots)
            state._kv_block_view = View(bank,j+2,dict(slots),state.value)
            rows.append(ReadInput(state,times[j+1],Content(leaf((5,)))))
    return w, rows, roots


def gradients(value, roots, zero=False):
    return torch.autograd.grad(value.sum()*(0 if zero else 1), roots, allow_unused=True, retain_graph=True)


@pytest.mark.parametrize('kind', ['read','full'])
@pytest.mark.parametrize('read,grouped', [('qry',False),('sum',True)])
def test_block_fields_causal_write_roots(dtype, kind, read, grouped):
    w, rows, roots = fixture(dtype,read,grouped)
    p = w.read_program if kind == 'read' else w.full_program
    p.kv_blocks = True
    if kind == 'full':
        controls = [torch.tensor(.3,dtype=dtype,requires_grad=True) for _ in rows]
        roots += controls
        rows = [FullInput(r.state,r.time,r.content,c) for r,c in zip(rows,controls)]
    expected = [p.step(w,r) if kind == 'read' else p.step(w,r,2,'hard',1.) for r in rows]
    actual = dispatch(p,w,rows,kind=kind,slots=2)
    assert len(actual) == len(rows)
    for i, e in enumerate(expected):
        equivalent(actual[i], e)
        a_fields = actual[i] if kind == 'read' else {'value':actual[i].value}
        e_fields = e if kind == 'read' else {'value':e.value}
        for name in a_fields:
            for zero in (False,True):
                equivalent(gradients(a_fields[name],roots,zero), gradients(e_fields[name],roots,zero))
    with torch.no_grad():
        inference = dispatch(p,w,rows,kind=kind,slots=2)
    equivalent(inference,actual)
    first = next(iter(actual[0].values())) if kind == 'read' else actual[0].value
    with torch.no_grad():
        rows[0].content.value.add_(.1)
    with pytest.raises(RuntimeError,match='modified|version|inplace'):
        first.sum().backward()


def test_block_declines_heterogeneous_roots_relocated_and_positioned(dtype):
    w, rows, _ = fixture(dtype)
    p = w.read_program; p.kv_blocks = True
    rows[0] = replace(rows[0],content=Content(rows[0].content.value.detach()))
    assert dispatch(p,w,rows) == {}
    rows = [replace(r,state=replace(r.state,slots={k:v.clone() for k,v in r.state.slots.items()})) for r in rows]
    assert dispatch(p,w,rows) == {}


@pytest.mark.parametrize('native',[False,True],ids=['python','native'])
@pytest.mark.parametrize('read_mode',['0','-','+'])
def test_default_block_public_outputs_terminal_roots_and_retained_cut(dtype,native,read_mode):
    import os
    from tidegraph import Continuation,External,ExecutionPlacement
    from tidegraph.weighted import fully_connected
    from tidegraph.weighted.execution import eager_runner
    from tidegraph.reference import run as reference
    from tidegraph.weighted.packing_observation import observe
    from tidegraph.runtime import resolve_device
    device,_=resolve_device(os.environ.get("TIDE_WINDOW_DEVICE","cpu"))
    spec=fully_connected(2,2)
    b=bind(spec.graph,memories=KVConfig(width=4,summary_width=2,heads=2),
           regions=RegionalConfig(k=2,read_mode=read_mode),dtype=dtype)
    b.model.to(device)
    execute=eager_runner(b,'native' if native else 'python',ExecutionPlacement(preset='cpu' if device.type=='cpu' else 'mixed-c'),
                         native_library=os.environ.get('TIDE_BUILD_DIR'),trace=False)
    xs=[External(batch,port,j,1+t*spec.stride,
                 torch.tensor([.4 if j!=1 else 0.,.3+.01*j,-.2,.4,-.3],dtype=dtype,device=device,requires_grad=True))
        for batch in range(2) for port in range(len(b.graph.inputs)) for j,t in enumerate((0,1,3,4))]
    q=Continuation(b.graph.identity,2);stop=6*spec.stride
    expected=reference(b.graph,b.model,q,xs,stop,sealed_until=stop)
    with observe() as work:
        actual=execute(q,xs,stop,sealed_until=stop)
    if not native:
        assert work.metrics().get('packing/projection/kv_block_full/rows',0)>0
    equivalent(actual.outputs,expected.outputs);equivalent(actual.continuation,expected.continuation)
    roots=[p for p in b.model.parameters() if p.requires_grad]+[x.value for x in xs]
    for index in (0,-1):
        a,e=actual.outputs[index][-1],expected.outputs[index][-1]
        equivalent(gradients(a,roots),gradients(e,roots))
        # Returned state handles are independent public roots, including future None.
        pairs=[(actual.continuation.states[owner].slots[name],s.slots[name])
               for owner,s in expected.continuation.states.items() for name in s.slots
               if s.slots[name].requires_grad]
        equivalent(gradients(a,[p[0] for p in pairs]),gradients(e,[p[1] for p in pairs]))
    for owner,s in actual.continuation.states.items():
        for name,t in s.slots.items():
            if t.requires_grad:
                equivalent(gradients(t,roots),gradients(expected.continuation.states[owner].slots[name],roots))
        assert not hasattr(s,'_kv_block_view')
    cut=3*spec.stride
    head=execute(q,[x for x in xs if x.time<cut],cut,sealed_until=cut)
    tail=[x for x in xs if x.time>=cut]
    continued=execute(head.continuation,tail,stop,sealed_until=stop)
    anchor=reference(b.graph,b.model,head.continuation,tail,stop,sealed_until=stop)
    equivalent(continued.outputs,anchor.outputs);equivalent(continued.continuation,anchor.continuation)
    equivalent(gradients(continued.outputs[-1][-1],roots),gradients(anchor.outputs[-1][-1],roots))
