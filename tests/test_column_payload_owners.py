"""Independent scalar schedule, physical messages and raw restart across owners.

CPU runs collapse owners. Set TIDE_COLUMN_PEERS=npu:0,npu:1 for actual distinct
owners; the device gate requires exactly two explicitly available devices.
"""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import Continuation, ExecutionPlacement, place_payloads
from tidegraph.checkpoint import save, load
from tidegraph.comparison import equivalent
from tidegraph.reference import run as reference
from tidegraph.runtime import resolve_device
from tidegraph.weighted import bind, KVConfig, RecurrentConfig, SourceBinding, eager_runner
from tidegraph.weighted.message_columns import MessageColumns
from test_settle_columns import general, inputs, external, observations


def devices():
    selected = os.environ.get("TIDE_COLUMN_PEERS", "cpu,cpu").split(",")
    if len(selected) != 2:
        raise ValueError("TIDE_COLUMN_PEERS must name two owners")
    for d in selected:
        resolve_device(d)
    return selected


def owner(family, selected, implementation, preset):
    c = (KVConfig(width=4, summary_width=2, heads=2, memory_read="qry", window=30, epsilon=.3)
         if family == "kv" else RecurrentConfig(memory_family="ssd", width=4, summary_width=2,
            heads=2, conv_size=3, memory_read="state", epsilon=.3))
    b, spec = general(torch.float32, memory=c)
    # Node-local source parameters permit an SSD region to span owners. KV's
    # actual regional shared base still requires its members to be co-located.
    b = bind(b.graph, memories=c, regions=b.regions, source_binding=SourceBinding("node"),
             alpha_agg=.2, dtype=torch.float32)
    b.model.nodes[1].kernel.summary = b.model.nodes[0].kernel.summary
    owners = [selected[0], selected[0], selected[1] if family == "kv" else selected[0], selected[1]]
    place_payloads(b.graph, b.model, owners)
    spec = replace(spec, graph=b.graph)
    optimizer = torch.optim.AdamW(b.model.parameters(), lr=.0001, foreach=False)
    runner = eager_runner(b, implementation, ExecutionPlacement(preset), trace=False)
    return b, spec, optimizer, runner


def objective(fields):
    return sum(t.square().sum().cpu()*.0001 for t in fields.values() if t.is_floating_point())


def cpu_fields(value):
    return {k: None if v is None else v.detach().cpu() for k, v in value.items()}


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return value


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("family", ["kv", "ssd"])
def test_complete_columns_independent_vjp_updates_and_raw_restore(implementation, family, tmp_path):
    selected = devices()
    preset = "native"  # Valid for CPU/accelerator and two accelerator owners.
    b, spec, opt, run = owner(family, selected, implementation, preset)
    oracle, ospec, oo, _ = owner(family, ("cpu", "cpu"), "python", "cpu")
    q = None; oq = Continuation(oracle.graph.identity, 2)
    for step in range(2):
        layout, ports = inputs(torch.float32)
        layout = replace(layout, times=layout.times+10*step)
        # An input may arrive on CPU even when its physical recipient is remote.
        expected_ports = {p: MessageColumns(x.rows, x.strength.detach().clone().requires_grad_(),
                                           x.content.detach().clone().requires_grad_()) for p, x in ports.items()}
        opt.zero_grad(set_to_none=True); oo.zero_grad(set_to_none=True)
        result = run.run_ports(spec, layout, ports, stop_position=10*(step+1), state=q)
        assert result.execution['schedule'] == 'settle-columns-v1'
        expected = reference(oracle.graph, oracle.model, oq, external(ospec, layout, expected_ports, oq),
                             50*(step+1), sealed_until=50*(step+1))
        a, e = observations(result.columns, expected, spec)
        equivalent(cpu_fields(a), cpu_fields(e), atol=1e-6, rtol=1e-5)
        for edge, payload in result.columns.messages.items():
            assert payload.content.device == b.model.nodes[b.graph.edges[edge].target].bias.device
        for node, state in result.columns.states.items():
            assert state.inert.device == b.model.nodes[node].bias.device
        # Output owners differ; the public numerical projection chooses its owner.
        projected = result.readout(ports[0].content)
        assert projected.device.type == 'cpu'
        with torch.no_grad():
            mass = torch.zeros(len(layout.times))
            numeric = torch.zeros(len(layout.times), 4)
            for key, payload in e.items():
                if key[0] == 'port':
                    row = key[2]; c = payload[0].clamp_min(0)
                    mass[row] += c; numeric[row] += c*payload[1:]
            equivalent(projected, numeric/mass.clamp_min(1)[:, None], atol=1e-6, rtol=1e-5)
        objective(a).backward(); objective(e).backward()
        ga = {n: p.grad for n, p in b.model.named_parameters()}
        ge = {n: p.grad for n, p in oracle.model.named_parameters()}
        ga.update({f'input{p}.{k}': getattr(x, k).grad for p, x in ports.items() for k in ('strength', 'content')})
        ge.update({f'input{p}.{k}': getattr(x, k).grad for p, x in expected_ports.items() for k in ('strength', 'content')})
        equivalent(cpu_fields(ga), cpu_fields(ge), atol=1e-6, rtol=1e-5)
        q, oq = result.continuation.detach(), expected.continuation.detach()
        opt.step(); oo.step()
        equivalent(cpu_fields(b.model.state_dict()), cpu_fields(oracle.model.state_dict()), atol=1e-6, rtol=1e-5)
    save(tmp_path/'checkpoint.pt', b.graph, b.model, q, opt)
    fresh, fspec, fo, frun = owner(family, selected, implementation, preset)
    fq = load(tmp_path/'checkpoint.pt', fresh.graph, fresh.model, fo)
    layout, ports = inputs(torch.float32)
    layout = replace(layout, times=layout.times+20)
    states = []
    for model, topology, optimizer, execute, state in ((b,spec,opt,run,q),(fresh,fspec,fo,frun,fq)):
        optimizer.zero_grad(set_to_none=True)
        result = execute.run_ports(topology, layout, ports, stop_position=30, state=state)
        # All output ports contribute independently to the task loss.
        from tidegraph.experiment.column_task import loss
        loss(result, .0001, training=True).backward()
        gradients = cpu_fields({n:p.grad for n,p in model.model.named_parameters()})
        optimizer.step()
        raw = {(b,v):(s.value,s.slots,s.last_time,s.observations) for (b,v),s in result.continuation.states.items()}
        states.append((cpu_fields(model.model.state_dict()), gradients, result, snapshot(optimizer.state_dict()), snapshot(raw)))
    equivalent(states[0][0], states[1][0], atol=0, rtol=0)
    equivalent(states[0][1], states[1][1], atol=0, rtol=0)
    equivalent(states[0][3], states[1][3], atol=0, rtol=0)
    equivalent(states[0][4], states[1][4], atol=0, rtol=0)
    for key in states[0][2].outputs:
        a, e = states[0][2].outputs[key], states[1][2].outputs[key]
        equivalent(cpu_fields(dict(strength=a.strength, content=a.content)),
                   cpu_fields(dict(strength=e.strength, content=e.content)), atol=0, rtol=0)
