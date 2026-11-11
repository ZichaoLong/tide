"""Native typed recurrent columns vs independent time-major scalar execution."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import RecurrentConfig, SettleColumnPlan, eager_runner
from test_settle_columns import general, inputs, external, observations, objective
from isolated_cases import vjp


def config(family, read="state"):
    args = dict(memory_family=family, memory_read=read, epsilon=.3)
    if family in {"gated_delta", "ssd"}:
        args.update(heads=2, conv_size=3, conv_stride=5, matrix_rates=(.01, .02))
    return RecurrentConfig(**args)


@pytest.mark.parametrize("family", ["affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("read", ["sum", "state"])
@pytest.mark.parametrize("mode", ["0", "-", "+"])
def test_native_snapshots_fields_and_all_shared_column_roots(family, read, mode):
    binding, spec = general(torch.float64, memory=config(family, read), read_mode=mode)
    domain, ports = inputs(torch.float64)
    domain = replace(domain, times=domain.times+2**54)
    actual = SettleColumnPlan(binding, spec, implementation="native").run_ports(domain, ports)
    stop = (2**54+10)*spec.stride
    q = Continuation(binding.graph.identity, 2, cut=2**54*spec.stride)
    expected = reference(binding.graph, binding.model, q, external(spec, domain, ports), stop, sealed_until=stop)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{n}": getattr(x, n) for p, x in ports.items() for n in ("strength", "content")})
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True))


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_ragged_scan_padding_is_counted_before_execution(implementation):
    from tidegraph.weighted import SequenceLayout, MessageColumns
    binding, spec = general(torch.float64, memory=config("ssd"))
    layout = SequenceLayout(torch.tensor([0]+[5]*16), torch.arange(5))
    payload = MessageColumns(torch.arange(5), torch.ones(5, dtype=torch.float64), torch.ones(5, 4, dtype=torch.float64))
    # Five rows alone would fit this budget; the declared scan uses 16×5 rows.
    with pytest.raises((MemoryError, ValueError, RuntimeError), match="workspace"):
        SettleColumnPlan(binding, spec, implementation=implementation, max_workspace_bytes=20_000).run_ports(layout, {0: payload})


@pytest.mark.parametrize("family", ["affine", "delta", "gated_delta", "ssd"])
def test_native_default_retained_windows_empty_samples_and_capacity(family):
    binding, spec = general(torch.float64, memory=config(family))
    domain, ports = inputs(torch.float64)
    make = lambda **kw: eager_runner(binding, "native", ExecutionPlacement("cpu"), trace=False, **kw)
    candidate, scalar = make(), make(query_execution="legacy")
    left, oracle = None, None
    losses = [0, 0]
    for offset in (0, 12):
        times = replace(domain, times=domain.times+offset)
        actual = candidate.run_ports(spec, times, ports, state=left, stop_position=offset+10)
        expected = scalar.run_ports(spec, times, ports, state=oracle, stop_position=offset+10)
        assert actual.execution['schedule'] == 'settle-columns-v1'
        equivalent(actual.outputs, expected.outputs)
        equivalent(actual.continuation, expected.continuation)
        left, oracle = actual.continuation, expected.continuation
        losses[0] = losses[0]+actual.readout(ports[0].content).square().sum()
        losses[1] = losses[1]+expected.readout(ports[0].content).square().sum()
    roots = {n: p for n, p in binding.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{n}": getattr(x, n) for p, x in ports.items() for n in ("strength", "content")})
    equivalent(vjp(losses[0], roots), vjp(losses[1], roots))
    plan = SettleColumnPlan(binding, spec, implementation="native")
    empty = replace(domain, offsets=torch.tensor([0, 0, 0]), times=torch.empty(0, dtype=torch.int64))
    idle = plan.run_ports(empty, {}, left=plan.import_state(left))
    equivalent(plan.export_state(idle, stop_position=30, previous=left), replace(left, cut=30*spec.stride), atol=0, rtol=0)
    with pytest.raises((MemoryError, ValueError, RuntimeError), match="workspace"):
        SettleColumnPlan(binding, spec, implementation="native", max_workspace_bytes=1).run_ports(domain, ports)


@pytest.mark.parametrize("family", ["affine", "delta", "gated_delta", "ssd"])
def test_native_nonempty_left_state_columns_are_independent_roots(family):
    binding, spec = general(torch.float64, memory=config(family))
    domain, ports = inputs(torch.float64)
    plan = SettleColumnPlan(binding, spec, implementation="native")
    head = plan.run_ports(domain, ports)
    q = plan.export_state(head, stop_position=10).detach()
    left, roots = {}, {}
    for v, s in plan.import_state(q).items():
        values = {n: getattr(s, n).detach().requires_grad_() for n in ("H", "Z", "W")}
        if s.cache_values is not None:
            values['cache_values'] = s.cache_values.detach().requires_grad_()
        left[v] = replace(s, **values)
        roots.update({f"{v}.{n}": t for n, t in values.items()})
        for b, state in enumerate(left[v].states()):
            q.states[b, v] = state
    later = replace(domain, times=domain.times+10)
    actual = plan.run_ports(later, ports, left=left)
    expected = reference(binding.graph, binding.model, q, external(spec, later, ports, q), 100, sealed_until=100)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True))
