"""Independent semantics for CSR pooling and immutable per-owner KV reuse."""
import copy
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run
from tidegraph.records import Result
from fiber_cases import fixture, physical
from fiber_pool_cases import analytic, profile
from isolated_cases import vjp
from test_fiber_single import ragged


@pytest.mark.parametrize('pooling,cache', [('csr','cloned'), ('event','owned'), ('csr','owned')])
@pytest.mark.parametrize('algorithm', ['streaming', 'frontier'])
@pytest.mark.parametrize('packing', ['exact', 'single'])
@pytest.mark.parametrize('layout', ['event', 'head'])
def test_ragged_prefill_values_caches_and_isolated_vjps(dtype, pooling, cache, algorithm, packing, layout):
    g,m,q,xs,leaves,_ = ragged(dtype)
    original = {k: {name:t.detach().clone() for name,t in s.slots.items()} for k,s in q.states.items()}
    expected = run(g,m,q,xs,12,sealed_until=12)
    actual = Native(g,m,algorithm=algorithm,workers=3,packed=True,attention_packing=packing,
                    fiber_pooling=pooling,fiber_cache=cache,attention_layout=layout).run(q,xs,12,sealed_until=12)
    equivalent(expected,actual)
    equivalent(original,{k:{name:t.detach() for name,t in s.slots.items()} for k,s in q.states.items()})
    for state in actual.continuation.states.values():
        for value in (state.value,*state.slots.values()):
            assert value.untyped_storage().nbytes() == value.numel()*value.element_size()
    # This isolated-node fixture has no edges and therefore no pending root.
    assert not expected.continuation.pending and not actual.continuation.pending
    for root in ('output','state'):
        equivalent(vjp(objective(expected,root),leaves),vjp(objective(actual,root),leaves))
    first = vjp(actual.outputs[0][3],leaves)
    assert first['input.0.0.1'] is None and first['input.1.0.0'] is None
    assert first['input_scale.2'] is None and first['nodes.0.extra.fiber_decay'] is None


@pytest.mark.parametrize('kind', ['sum','mean','linear','active-softmax','all-softmax'])
@pytest.mark.parametrize('mode', ['hard','softp','hst'])
@pytest.mark.parametrize('packing,trace,workers', [('exact',True,1), ('single',False,3)])
def test_cycles_clear_zero_sources_and_all_public_roots(dtype, kind, mode, packing, trace, workers):
    g,m,q,xs,leaves = fixture(dtype,'clear',cyclic=True,profile=profile(kind))
    expected = run(g,m,q,xs,9,sealed_until=9,mode=mode)
    actual = Native(g,m,workers=workers,packed=True,attention_packing=packing,mode=mode,trace=trace,
                    parallel_regions=True,compact_events=True,defer_state_release=True,
                    fiber_pooling='csr',fiber_cache='owned',attention_layout='head').run(q,xs,9,sealed_until=9)
    if trace: equivalent(expected,actual)
    else:
        equivalent(expected.outputs,actual.outputs)
        equivalent(expected.continuation,actual.continuation)
        assert not actual.trace and not actual.messages
    equivalent(physical(expected,m),physical(actual,m))
    for root in ('output','pending','state'):
        equivalent(vjp(objective(expected,root),leaves),vjp(objective(actual,root),leaves))


@pytest.mark.parametrize('packing', ['exact','single'])
def test_active_domain_extreme_logits_avoid_absent_slot_underflow(dtype, packing):
    g,m,w = analytic(dtype,'active-softmax')
    with torch.no_grad(): w.extra['fiber_pool'].copy_(torch.tensor([-1000.,-1001.,1000.],dtype=dtype))
    x=torch.tensor([1.,3.],dtype=dtype,requires_grad=True)
    xs=[External(0,p,0,0,x[p:p+1]) for p in (0,1)]
    q=Continuation(g.identity,1);leaves=dict(m.named_parameters())|{'input':x}
    expected=run(g,m,q,xs,1,sealed_until=1)
    actual=Native(g,m,packed=True,attention_packing=packing,fiber_pooling='csr',fiber_cache='owned').run(q,xs,1,sealed_until=1)
    equivalent(expected,actual)
    equivalent(vjp(objective(expected),leaves),vjp(objective(actual),leaves))
    assert torch.isfinite(actual.outputs[0][3]).all()


@pytest.mark.parametrize('packing', ['exact','single'])
def test_retained_snapshots_and_checkpoint_can_change_execution_policy(dtype, packing, tmp_path):
    g,m,q,xs,_=fixture(dtype,'clear',cyclic=True,profile=profile('all-softmax'))
    with torch.no_grad():
        expected=run(g,m,q,xs,12,sealed_until=12)
        cursor=Native(g,m,workers=3,packed=True,attention_packing=packing,fiber_pooling='csr',fiber_cache='owned',attention_layout='head',
                      compact_events=True,defer_state_release=True).cursor(q)
        first=cursor.advance([x for x in xs if x.time<4],4,sealed_until=4)
        snapshot=cursor.snapshot();before=copy.deepcopy(snapshot)
        path=tmp_path/'state.pt';save(path,g,m,snapshot)
        last=cursor.advance([x for x in xs if x.time>=4],12,sealed_until=12)
        equivalent(snapshot,before)
        actual=Result(cursor.snapshot(),first.trace+last.trace,first.outputs+last.outputs,
                      first.messages+last.messages,last.stats)
        equivalent(expected,actual)
        restored=load(path,g,m)
        plain=Native(g,m,workers=1,packed=True).cursor(restored)
        suffix=plain.advance([x for x in xs if x.time>=4],12,sealed_until=12)
        equivalent(cursor.snapshot(),plain.snapshot());equivalent(last.outputs,suffix.outputs)
        equivalent(snapshot,before)


def training(dtype, optimized):
    g,m,q,xs,_=fixture(dtype,profile=profile('all-softmax'));m.nodes[1]=m.nodes[0]
    # Non-contiguous matrices have the same values and one explicit parameter
    # owner. The public native program accepts this layout without cached copies.
    if optimized:
        for w in (m.nodes[0],m.nodes[2]):
            for key in ('fiber_qkv','fiber_out'):
                w.extra[key]=torch.nn.Parameter(w.extra[key].detach().t().contiguous().t())
    opt=torch.optim.AdamW(m.parameters(),lr=.002,eps=1e-5);records=[]
    for stop in (4,12):
        opt.zero_grad(set_to_none=True);inputs=[x for x in xs if q.cut<=x.time<stop]
        if optimized:
            r=Native(g,m,workers=3,packed=True,trace=False,compact_events=True,defer_state_release=True,
                     fiber_pooling='csr',fiber_cache='owned',attention_layout='head').run(q,inputs,stop,sealed_until=stop)
        else:r=run(g,m,q,inputs,stop,sealed_until=stop)
        objective(r).backward();opt.step();q=r.continuation.detach()
        records.append((q,{k:None if p.grad is None else p.grad.clone() for k,p in m.named_parameters()},
                        {k:v.clone() for k,v in m.state_dict().items()}))
    return records


def test_shared_strided_parameters_optimizer_and_continuation(dtype):
    equivalent(training(dtype,False),training(dtype,True))


@pytest.mark.parametrize('kwargs', [{'fiber_pooling':'bad'}, {'fiber_cache':'bad'}, {'defer_state_release':True}, {'attention_layout':'bad'}])
def test_invalid_execution_options_rejected(dtype, kwargs):
    g,m,*_=fixture(dtype)
    with pytest.raises(ValueError):Native(g,m,**kwargs)
