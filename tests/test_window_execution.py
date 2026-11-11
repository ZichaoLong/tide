"""Default tensor boundary, general fallback, root ownership and raw recovery."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, ExecutionPlacement
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import eager_runner, SequenceLayout, MessageColumns, SettleColumnPlan
from test_settle_columns import general, inputs, external, observations, objective
from isolated_cases import vjp


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_default_ports_and_explicit_graph_adapter(implementation):
    b, spec = general(torch.float64)
    domain, ports = inputs(torch.float64)
    make = lambda **kw: eager_runner(b, implementation, ExecutionPlacement("cpu"), trace=False, **kw)
    new, graph = make(), make(query_execution="legacy")
    a = new.run_ports(spec, domain, ports, stop_position=10)
    e = graph.run_ports(spec, domain, ports, stop_position=10)
    assert a.execution["schedule"] == "settle-columns-v1" and a.records is None
    assert e.execution["schedule"] == "graph-record-adapter" and e.execution["fallback_reason"]
    equivalent(a.outputs, e.outputs)
    equivalent(a.continuation, e.continuation)
    roots = dict(b.model.named_parameters())
    roots = {n: p for n, p in roots.items() if p.requires_grad}
    roots.update({f"port{p}.{k}": getattr(x, k) for p, x in ports.items() for k in ("strength", "content")})
    equivalent(vjp(a.readout(ports[0].content).square().sum(), roots),
               vjp(e.readout(ports[0].content).square().sum(), roots))
    later = replace(domain, times=domain.times+12)
    continued = new.run_ports(spec, later, ports, stop_position=22, state=a.continuation)
    expected = graph.run_ports(spec, later, ports, stop_position=22, state=e.continuation)
    equivalent(continued.outputs, expected.outputs)
    equivalent(continued.continuation, expected.continuation)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_empty_samples_capacity_and_rejection(implementation):
    b, spec = general(torch.float64)
    plan = SettleColumnPlan(b, spec, implementation=implementation)
    domain = SequenceLayout(torch.tensor([0, 0, 2]), torch.tensor([2, 7]))
    payload = MessageColumns(torch.tensor([0, 1]), torch.ones(2, dtype=torch.float64), torch.ones(2, 4, dtype=torch.float64))
    result = plan.run_ports(domain, {0: payload})
    q = plan.export_state(result, stop_position=8)
    oracle = reference(b.graph, b.model, Continuation(b.graph.identity, 2), external(spec, domain, {0: payload}), 40, sealed_until=40)
    equivalent(q, oracle.continuation)
    empty = SequenceLayout(torch.tensor([0, 0, 0]), torch.empty(0, dtype=torch.int64))
    nothing = plan.run_ports(empty, {}, left=plan.import_state(q))
    equivalent(plan.export_state(nothing, stop_position=11, previous=q), replace(q, cut=55), atol=0, rtol=0)
    with pytest.raises((ValueError, RuntimeError), match="port"):
        plan.run_ports(domain, {99: payload})
    with pytest.raises((ValueError, RuntimeError), match="int64"):
        plan.run_ports(domain, {0: replace(payload, rows=payload.rows.float())})
    with pytest.raises((ValueError, RuntimeError), match="finite|invalid"):
        plan.run_ports(domain, {0: replace(payload, strength=payload.strength*float("nan"))})
    with pytest.raises((MemoryError, ValueError, RuntimeError), match="workspace"):
        SettleColumnPlan(b, spec, implementation=implementation, max_workspace_bytes=1).run_ports(domain, {0: payload})
    with pytest.raises(ValueError, match="pending"):
        plan.import_state(replace(q, pending=[object()]))


@pytest.mark.parametrize("mode", ["constrained", "regional_threshold_hard", "direct_attention", "unit_attention", "pooled_attention"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_remaining_send_candidates(mode, implementation):
    b, spec = general(torch.float64, mode)
    domain, ports = inputs(torch.float64)
    actual = SettleColumnPlan(b, spec, implementation=implementation).run_ports(domain, ports)
    expected = reference(b.graph, b.model, Continuation(b.graph.identity, 2), external(spec, domain, ports), 50, sealed_until=50)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad}
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True))


@pytest.mark.parametrize("kind,alpha", [("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
def test_scoring_key_evidence_and_smooth_regularizers(kind, alpha):
    b, spec = general(torch.float64, regularizer=kind, ent_alpha=alpha, local_score="match",
                      region_score="att", region_key="content", region_evidence="grouped")
    domain, ports = inputs(torch.float64)
    actual = SettleColumnPlan(b, spec).run_ports(domain, ports)
    expected = reference(b.graph, b.model, Continuation(b.graph.identity, 2), external(spec, domain, ports), 50, sealed_until=50)
    a, e = observations(actual, expected, spec)
    equivalent(a, e)
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad}
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
