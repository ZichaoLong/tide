"""Column-root training, retained windows and checkpoint-v5 state decoding."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.checkpoint import save, load
from tidegraph.reference import run as reference
from tidegraph.weighted import layered_inputs, bind, RecurrentConfig, KVConfig, RegionalConfig
from tidegraph.weighted.settle_columns import SettleColumnPlan
from test_settle_columns import general, inputs, external, observations


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("positions", [False, True])
def test_raw_checkpoint_and_retained_windows(tmp_path, implementation, positions):
    dtype = torch.float64
    memory = KVConfig(heads=2, memory_read="qry", window=17, position_scale=5,
        read_frequencies=(.17,), full_frequencies=((.11,), (.23,)), distance_slopes=(.04, .03)) if positions else None
    b, spec = general(dtype, memory=memory)
    plan = SettleColumnPlan(b, spec, implementation=implementation)
    domain, ports = inputs(dtype)
    head = plan.run_ports(domain, ports)
    q = plan.export_state(head, stop_position=10)
    oracle = reference(b.graph, b.model, Continuation(b.graph.identity, 2), external(spec, domain, ports), 50, sealed_until=50)
    equivalent(q, oracle.continuation)
    opt = torch.optim.AdamW(b.model.parameters(), lr=.0001)
    loss = head.readout(ports[0].content).square().sum()
    loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    q = q.detach()
    save(tmp_path/"columns.pt", b.graph, b.model, q, opt)
    owner, ospec = general(dtype, memory=memory)
    oopt = torch.optim.AdamW(owner.model.parameters(), lr=.0001)
    restored = load(tmp_path/"columns.pt", owner.graph, owner.model, oopt)
    equivalent(q, restored, atol=0, rtol=0)
    equivalent(opt.state_dict(), oopt.state_dict(), atol=0, rtol=0)
    next_domain = replace(domain, times=domain.times+12)
    resumed_plan = SettleColumnPlan(owner, ospec, implementation=implementation)
    actual = plan.run_ports(next_domain, ports, left=plan.import_state(q))
    resumed = resumed_plan.run_ports(next_domain, ports, left=resumed_plan.import_state(restored))
    equivalent(actual.readout(ports[0].content), resumed.readout(ports[0].content), atol=0, rtol=0)
    equivalent(plan.export_state(actual, stop_position=22, previous=q),
               resumed_plan.export_state(resumed, stop_position=22, previous=restored), atol=0, rtol=0)
    for result, optimizer in ((actual, opt), (resumed, oopt)):
        result.readout(ports[0].content).square().sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    equivalent(b.model.state_dict(), owner.model.state_dict(), atol=0, rtol=0)
    equivalent(opt.state_dict(), oopt.state_dict(), atol=0, rtol=0)


def test_heterogeneous_private_states_share_the_same_column_route():
    from tidegraph.weighted import fully_connected
    dtype = torch.float64
    topology = fully_connected(2, 2)
    memories = (KVConfig(), RecurrentConfig(memory_family="gated_delta", heads=2, conv_size=3),
                RecurrentConfig(memory_family="ssd", heads=2, conv_size=3), KVConfig(memory_read="qry"))
    binding = bind(topology.graph, memories=memories, regions=RegionalConfig(k=2), dtype=dtype)
    spec = replace(topology, graph=binding.graph)
    plan = SettleColumnPlan(binding, spec)
    x = torch.linspace(-.3, .7, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    q = Continuation(binding.graph.identity, 2)
    actual = plan.run(x)
    expected = reference(binding.graph, binding.model, q, layered_inputs(spec, x), 3*spec.stride, sealed_until=3*spec.stride)
    pairs = observations(actual, expected, spec)
    equivalent(*pairs)
    exported = plan.export_state(actual, stop_position=3)
    equivalent(exported, expected.continuation)
    next_actual = plan.run(x+.1, start_position=4, left=actual.states)
    next_expected = reference(binding.graph, binding.model, expected.continuation,
        layered_inputs(spec, x+.1, start_position=4, start_occurrence=3), 7*spec.stride, sealed_until=7*spec.stride)
    a, e = observations(next_actual, next_expected, spec)
    equivalent(a, e)
    from test_settle_columns import objective
    roots = [x, *(p for p in binding.model.parameters() if p.requires_grad)]
    equivalent(torch.autograd.grad(objective(a), roots, allow_unused=True),
               torch.autograd.grad(objective(e), roots, allow_unused=True))
