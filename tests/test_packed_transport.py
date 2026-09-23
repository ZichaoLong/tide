"""Optional transport against independent event semantics, including public VJPs."""
import copy
from dataclasses import replace
import pytest
import torch
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.reference import run
from aggregate_cases import fixture as aggregate_fixture
from fiber_cases import fixture
from fiber_pool_cases import profile
from isolated_cases import vjp
from next_cases import blend_fixture


FLAGS = [dict(packed_sources=True), dict(batch_next=True), dict(packed_sources=True, batch_next=True)]


@pytest.mark.parametrize('flags', FLAGS)
@pytest.mark.parametrize('packing,workers', [('exact', 1), ('single', 3)])
@pytest.mark.parametrize('mode', ['hard', 'softp', 'hst'])
@pytest.mark.parametrize('policy', ['all', 'selected', 'clear', 'old'])
def test_cycles_ragged_sources_clear_and_isolated_roots(dtype, flags, packing, workers, mode, policy):
    g,m,q,xs,leaves=fixture(dtype,policy,cyclic=True,profile=profile('all-softmax'))
    # Local slot order differs from canonical source atom order.
    layout=replace(g.ports,
        edge_target=tuple(g.source_counts[e.target]-1-s for e,s in zip(g.edges,g.ports.edge_target)),
        input=tuple(g.source_counts[v]-1-s for v,s in zip(g.inputs,g.ports.input)))
    g=replace(g,layout=layout);q.identity=g.identity
    expected=run(g,m,q,xs,9,sealed_until=9,mode=mode)
    engine=Native(g,m,workers=workers,packed=True,mode=mode,attention_packing=packing,
                  compact_events=True,parallel_regions=True,**flags)
    actual=engine.run(q,xs,9,sealed_until=9)
    equivalent(expected,actual)
    for root in ('output','pending','state'):
        equivalent(vjp(objective(expected,root),leaves),vjp(objective(actual,root),leaves))
    for key in ('content','next'):
        equivalent(vjp(expected.trace[0][key],leaves),vjp(actual.trace[0][key],leaves))
    if flags.get('packed_sources'): assert actual.stats['packed_source_batches'] > 0
    if flags.get('batch_next'):
        assert actual.stats['next_batches'] > 0
        assert actual.stats['semantic_next_replays'] == actual.stats['next_steps']
        if policy=='clear': assert actual.stats['next_reset_batches'] > 0
    with torch.no_grad():
        inference=engine.run(q,xs,9,sealed_until=9)
        equivalent(expected,inference)
        assert inference.stats.get('semantic_next_replays',0)==0


@pytest.mark.parametrize('kind', ['sum','mean','weighted_mean','active_softmax','all_softmax'])
def test_aggregate_profiles_slots_parallel_edges_and_nonfiber_consumers(dtype, kind):
    g,m,q,xs,leaves=aggregate_fixture(dtype,kind)
    # Cut before the last edge deliveries, so pending is a nonempty public root.
    expected=run(g,m,q,xs,4,sealed_until=4,mode='hst')
    actual=Native(g,m,packed=True,workers=3,packed_sources=True,batch_next=True,mode='hst').run(q,xs,4,sealed_until=4)
    equivalent(expected,actual)
    for root in ('output','pending','state'):
        equivalent(vjp(objective(expected,root),leaves),vjp(objective(actual,root),leaves))
    first=expected.trace[0];slot=next(iter(first['contributions']))
    equivalent(vjp(first['contributions'][slot],leaves),vjp(actual.trace[0]['contributions'][slot],leaves))


@pytest.mark.parametrize('clear', [False,True])
def test_control_blend_uses_counted_fallback(dtype, clear):
    g,m,q,xs,leaves=blend_fixture(dtype,clear=clear)
    expected=run(g,m,q,xs,5,sealed_until=5)
    actual=Native(g,m,packed=True,workers=3,packed_sources=True,batch_next=True).run(q,xs,5,sealed_until=5)
    equivalent(expected,actual)
    equivalent(vjp(objective(expected),leaves),vjp(objective(actual),leaves))
    assert actual.stats['next_scalar_fallback_steps']==actual.stats['next_steps']
    assert actual.stats.get('next_batches',0)==0


@pytest.mark.parametrize('packing', ['exact','single'])
def test_snapshots_trace_off_and_checkpoint_policy_switch(dtype, packing, tmp_path):
    g,m,q,xs,_=fixture(dtype,'clear',cyclic=True)
    with torch.no_grad():
        expected=run(g,m,q,xs,12,sealed_until=12)
        cursor=Native(g,m,packed=True,workers=3,trace=False,packed_sources=True,batch_next=True,
                      fiber_cache='owned',fiber_pooling='csr',attention_layout='head',attention_packing=packing,
                      compact_events=True,defer_state_release=True).cursor(q)
        first=cursor.advance([x for x in xs if x.time<4],4,sealed_until=4)
        snapshot=cursor.snapshot();before=copy.deepcopy(snapshot)
        save(tmp_path/'state.pt',g,m,snapshot)
        last=cursor.advance([x for x in xs if x.time>=4],12,sealed_until=12)
        equivalent(snapshot,before)
        equivalent(expected.continuation,cursor.snapshot())
        equivalent(expected.outputs,first.outputs+last.outputs)
        assert not first.trace and not last.trace
        restored=load(tmp_path/'state.pt',g,m)
        plain=Native(g,m,packed=True).run(restored,[x for x in xs if x.time>=4],12,sealed_until=12)
        equivalent(plain.continuation,cursor.snapshot());equivalent(plain.outputs,last.outputs)


def training(dtype, optimized):
    torch.manual_seed(29)
    # These two nodes have matching learned source domains and can share a module.
    g,m,q,xs,_=fixture(dtype,'clear',profile=profile('all-softmax'));m.nodes[1]=m.nodes[0]
    optimizer=torch.optim.AdamW(m.parameters(),lr=.002,eps=1e-5);records=[]
    for stop in (4,9):
        optimizer.zero_grad(set_to_none=True)
        inputs=[x for x in xs if q.cut<=x.time<stop]
        if optimized:
            r=Native(g,m,packed=True,workers=3,packed_sources=True,batch_next=True,
                     trace=False,compact_events=True).run(q,inputs,stop,sealed_until=stop)
        else:r=run(g,m,q,inputs,stop,sealed_until=stop)
        objective(r).backward();optimizer.step();q=r.continuation.detach()
        records.append((q,{k:None if p.grad is None else p.grad.clone() for k,p in m.named_parameters()},
                        {k:v.clone() for k,v in m.state_dict().items()},copy.deepcopy(optimizer.state_dict())))
    return records


def test_shared_parameter_optimizer_and_chunked_continuation(dtype):
    equivalent(training(dtype,False),training(dtype,True))


@pytest.mark.parametrize('context', [torch.no_grad,torch.inference_mode])
def test_local_clock_clear_and_inference_modes(dtype, context):
    from tidegraph.clocks import StateClock
    from clock_cases import fixture as clock_fixture, project
    clock=StateClock(5,2,2)
    with context():
        g,m,q,eg,em,eq,xs,exs,_=clock_fixture(dtype,clock,profile('all-softmax'),'clear')
        expected=run(g,m,q,xs,7,sealed_until=7)
        actual=Native(eg,em,packed=True,workers=3,packed_sources=True,batch_next=True).run(
            eq,exs,clock.to_global(7),sealed_until=clock.to_global(7))
        equivalent(expected,project(actual,g,clock))
        assert actual.stats['next_reset_batches']>0
        assert actual.stats.get('semantic_next_replays',0)==0


@pytest.mark.parametrize('flags', FLAGS)
def test_unsupported_schedules_rejected(dtype, flags):
    g,m,*_=fixture(dtype)
    with pytest.raises(ValueError,match='packed'):Native(g,m,**flags)
    for algorithm in ('frontier','chain','self_loop'):
        with pytest.raises(ValueError,match='streaming'):Native(g,m,algorithm=algorithm,packed=True,**flags)
