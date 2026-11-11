"""Half tensor windows: explicit scoring, independent roots, masters and raw resume.

The existing half error policy applies to descriptors/masters derived from half
payloads too. None/discrete results and same-stack recovery remain exact.
TIDE_COLUMN_DEVICE selects a real device; CPU execution never certifies NPU.
"""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.checkpoint import save, load
from tidegraph.comparison import equivalent
from tidegraph.placement import place_model
from tidegraph.precision import FP32MasterOptimizer
from tidegraph.reference import run as reference
from tidegraph.runtime import resolve_device
from tidegraph.weighted import KVConfig, RecurrentConfig, eager_runner
from tidegraph.weighted.block_attention import attention
from tidegraph.weighted.column_attention import logit_attention
from test_settle_columns import general, inputs, external, observations
from test_weighted_precision import half_close


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def objective(fields):
    return sum(v.float().square().sum().cpu()*.0001 for v in fields.values() if v.is_floating_point())


def owner(family, implementation, device, positions=False):
    config = (KVConfig(width=4, summary_width=2, heads=2, memory_read="qry", window=30, epsilon=.3,
        **(dict(read_frequencies=(.17,), full_frequencies=((.11,),(.23,)),
                distance_slopes=(.01,.02), position_scale=3.) if positions else {}))
        if family == "kv" else RecurrentConfig(memory_family=family, width=4, summary_width=2,
            memory_read="state", epsilon=.3, **(dict(heads=2, conv_size=3) if family in {"gated_delta","ssd"} else {})))
    binding, spec = general(torch.float16, material_dtype=torch.float32, memory=config)
    binding.model.to(device=device)
    placement = ExecutionPlacement("native", scoring_dtype="float32")
    execute = eager_runner(binding, implementation, placement, trace=False)
    opt = FP32MasterOptimizer((p for p in binding.model.parameters() if p.requires_grad),
                             optimizer="adamw", lr=.0001, loss_scale=16, foreach=False)
    return binding, spec, opt, execute, placement


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family,positions", [(x,False) for x in ("kv","affine","delta","gated_delta","ssd")]+[("kv",True)])
def test_half_columns_independent_observations_vjp_master_and_restore(family, positions, implementation, tmp_path):
    device = os.environ.get("TIDE_COLUMN_DEVICE", "cpu")
    resolve_device(device)
    b, spec, opt, execute, placement = owner(family, implementation, device, positions)
    oracle, ospec, oo, _, op = owner(family, "python", "cpu", positions)
    view = place_model(oracle.graph, oracle.model, op)
    base = 2**54+3
    state = Continuation(b.graph.identity, 2, cut=base*spec.stride)
    expected_state = Continuation(oracle.graph.identity, 2, cut=base*spec.stride)
    for step in range(3):
        layout, ports = inputs(torch.float16, material_dtype=torch.float32)
        _, expected_ports = inputs(torch.float16, material_dtype=torch.float32)
        layout = replace(layout, times=layout.times+base+10*step)
        opt.zero_grad(); oo.zero_grad()
        actual = execute.run_ports(spec, layout, ports, stop_position=base+10*(step+1), state=state)
        assert actual.execution['schedule'] == 'settle-columns-v1'
        expected = reference(oracle.graph, view, expected_state, external(ospec, layout, expected_ports, expected_state),
                             (base+10*(step+1))*spec.stride, sealed_until=(base+10*(step+1))*spec.stride)
        a, e = observations(actual.columns, expected, spec)
        half_close(cpu(a), cpu(e))
        assert all(n.controls.dtype == torch.float16 for n in actual.columns.nodes.values())
        assert all(v.dtype == torch.float32 for n in actual.columns.nodes.values() for v in n.descriptors.values())
        opt.backward(objective(a)); oo.backward(objective(e))
        ga = {n:p.grad for n,p in b.model.named_parameters()}
        ge = {n:p.grad for n,p in oracle.model.named_parameters()}
        ga.update({f'input{p}.{n}':getattr(x,n).grad for p,x in ports.items() for n in ('strength','content')})
        ge.update({f'input{p}.{n}':getattr(x,n).grad for p,x in expected_ports.items() for n in ('strength','content')})
        half_close(cpu(ga), cpu(ge))
        opt.step(); oo.step()
        half_close(cpu(b.model.state_dict()), cpu(oracle.model.state_dict()))
        half_close(cpu(opt.state_dict()), cpu(oo.state_dict()))
        state, expected_state = actual.continuation.detach(), expected.continuation.detach()
    save(tmp_path/'half.pt', b.graph, b.model, state, opt)
    fresh, fspec, fo, frun, _ = owner(family, implementation, device, positions)
    restored = load(tmp_path/'half.pt', fresh.graph, fresh.model, fo)
    equivalent(cpu(opt.state_dict()), cpu(fo.state_dict()), atol=0, rtol=0)
    layout, ports = inputs(torch.float16, material_dtype=torch.float32)
    layout = replace(layout, times=layout.times+base+30)
    results = []
    for binding, topology, optimizer, run, old in ((b,spec,opt,execute,state),(fresh,fspec,fo,frun,restored)):
        optimizer.zero_grad()
        r = run.run_ports(topology, layout, ports, stop_position=base+40, state=old)
        optimizer.backward(r.readout(ports[0].content).float().square().sum()*.0001)
        gradients = cpu({n:p.grad for n,p in binding.model.named_parameters()})
        optimizer.step()
        raw = {k:(s.value,s.last_time,s.observations,s.slots) for k,s in r.continuation.states.items()}
        outputs = {k:(p.rows,p.strength,p.content) for k,p in r.outputs.items()}
        results.append(cpu((binding.model.state_dict(),gradients,optimizer.state_dict(),raw,outputs)))
    equivalent(*results, atol=0, rtol=0)


@pytest.mark.parametrize("positional", [False, True])
def test_half_attention_fp32_literal_with_overflowing_half_dot_and_partial_roots(positional):
    q = torch.tensor([[[[300.,300.],[.2,-.4],[.0002,-.0004],[.3,.1]]]], dtype=torch.float16, requires_grad=True)
    k = torch.tensor([[[[300.,300.],[300.,-300.],[.1,.2]]]], dtype=torch.float16, requires_grad=True)
    v = torch.tensor([[[[.1,.3],[.7,-.2],[.4,.6]]]], dtype=torch.float16, requires_grad=True)
    w = torch.tensor([[[.5,1.,0.],[0.,0.,0.],[.4,.7,.1],[0.,0.,.3]]], dtype=torch.float16, requires_grad=True)
    logits = q.float()@k.float().transpose(-1,-2)/(2**.5)
    actual = logit_attention(logits, v, w) if positional else attention(q,k,v,w)
    weights = w.float(); positive = weights>0; active = positive.any(-1)[:,None,:,None]
    l = (logits+torch.where(positive,weights,1).log()[:,None]).masked_fill(~positive[:,None],-torch.inf)
    p = torch.where(active, torch.where(active,l,0).softmax(-1), 0)
    expected = (p@v.float()).half()
    equivalent(actual, expected, atol=0, rtol=0)
    bars = torch.tensor([[[[.3,-.2],[.4,.1],[-.1,.2],[.1,.7]]]],dtype=torch.float16)
    ga = torch.autograd.grad(actual, (q,k,v,w), bars, retain_graph=True)
    ge = torch.autograd.grad(expected, (q,k,v,w), bars)
    half_close(ga,ge)
    with torch.no_grad(): v.add_(1)
    with pytest.raises(RuntimeError, match="modified|inplace"):
        torch.autograd.grad(actual.sum(), (q,k,v,w))


def test_half_columns_require_explicit_solver_dtype():
    binding, spec, _, _, _ = owner("kv", "python", "cpu")
    execute = eager_runner(binding, "python", ExecutionPlacement("cpu"), trace=False)
    with pytest.raises(ValueError, match="FP32/FP64"):
        execute.run_positions(spec, torch.ones(1,2,4,dtype=torch.float16))
