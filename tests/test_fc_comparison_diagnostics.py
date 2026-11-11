"""Cost-only boundary omissions must preserve the declared small task projection."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fc_comparison_diagnostics import boundary_scope, boundary_calls
from fc_tensor_baseline import run
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.planning.recipes import prepare
from tidegraph.planning.materialize import materialize
from tidegraph.settle import SettleGraph
from tidegraph.weighted import layered_inputs, readout
from tidegraph.weighted.execution import eager_runner


@pytest.mark.parametrize('boundary',['public','query-direct','region-direct','both-direct'])
def test_task_projection_and_scoped_restoration(dtype,boundary):
    from tidegraph.weighted import query_vjp,region_vjp
    originals=(query_vjp.evaluate,region_vjp.evaluate)
    plan,_=prepare(dict(topology='fc',family='kv',width=4,depth=2,region_width=2,sparse_denominator=2,
        memory_overrides=dict(memory_read='qry'),region_overrides=dict(allocation_temperature=.25)),
        samples=2,positions=3,seed=7)
    binding,_,_=materialize(plan,max_bytes=2**24,dtype=dtype)
    spec=SettleGraph(binding.graph,tuple(plan['graph']['ranks']),binding='ports',time_stride=plan['stride'])
    x=torch.linspace(-.2,.4,24,dtype=dtype).reshape(2,3,4).requires_grad_()
    expected=run(binding,spec,x)['hidden']
    roots=[x,*(p for p in binding.model.parameters() if p.requires_grad)]
    expected_grad=torch.autograd.grad(expected.square().sum(),roots,allow_unused=True)
    execute=eager_runner(binding,'python',ExecutionPlacement(preset='cpu',scoring_dtype=str(dtype).split('.')[-1]),trace=False)
    with boundary_scope(boundary):
        result=execute(Continuation(binding.graph.identity,2),layered_inputs(spec,x),3*spec.stride,sealed_until=3*spec.stride)
        actual=torch.stack([torch.stack([readout(result.outputs,
            [(pos*spec.stride+spec.rank(v),p) for p,v in enumerate(binding.graph.outputs)],x[b,pos],batch=b)
            for pos in range(3)]) for b in range(2)])
        gradients=torch.autograd.grad(actual.square().sum(),roots,allow_unused=True)
        calls=boundary_calls()
        if boundary in ('query-direct','both-direct'):
            assert calls.get('read_rows',0)>0 and calls.get('full_rows',0)>0
            assert calls.get('block_read_rows',0)>0 and calls.get('block_full_rows',0)>0
        if boundary in ('region-direct','both-direct'):
            assert calls.get('region_rows',0)>0
    assert originals==(query_vjp.evaluate,region_vjp.evaluate)
    from tidegraph.compare import equivalent
    equivalent(actual,expected)
    # This is explicitly a numeric derivative projection, not independent-root
    # connectivity qualification. Absence is checked against actual numeric zero.
    for a,e,root in zip(gradients,expected_grad,roots):
        equivalent(torch.zeros_like(root) if a is None else a,torch.zeros_like(root) if e is None else e)


def test_diagnostic_patches_restore_after_failure():
    from tidegraph.weighted import query_vjp,region_vjp,kv_block_query
    block_original=kv_block_query._Query.apply
    originals=(query_vjp.evaluate,region_vjp.evaluate)
    with pytest.raises(RuntimeError):
        with boundary_scope('both-direct'):
            raise RuntimeError('diagnostic failed')
    assert kv_block_query._Query.apply == block_original
    assert originals==(query_vjp.evaluate,region_vjp.evaluate)


@pytest.mark.parametrize('attention',['sdpa','sdpa4d'])
def test_fc_sdpa_projection_and_parameter_derivatives(dtype,attention):
    from test_fc_tensor_baseline import fixture
    from tidegraph.compare import equivalent
    binding,spec=fixture(dtype)
    x=torch.linspace(-.2,.4,24,dtype=dtype).reshape(2,3,4).requires_grad_()
    roots=[x,*(p for p in binding.model.parameters() if p.requires_grad)]
    a=run(binding,spec,x,full_attention='manual')
    e=run(binding,spec,x,full_attention=attention)
    equivalent(a,e)
    equivalent(torch.autograd.grad(a['hidden'].square().sum(),roots,allow_unused=True),
               torch.autograd.grad(e['hidden'].square().sum(),roots,allow_unused=True))
