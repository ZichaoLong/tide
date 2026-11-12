"""Source batching integrates independent schedules and retained raw states."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.reference import run as reference
from tidegraph.compare import equivalent
from tidegraph.weighted import bind, KVConfig, SourceBinding
from tidegraph.weighted.settle_columns import SettleColumnPlan
from tidegraph.experiment.work_observation import observe_work
from isolated_cases import vjp
from test_settle_columns import general, inputs, external, observations, objective


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("mode", ["source", "adapted"])
def test_nonempty_ragged_duplicate_edges_all_roots(mode, implementation):
    # The reference has its own physical-fiber schedule and scalar source MLP.
    original, spec = general(torch.float64)
    classes = original.source_classes
    labels = sorted(set((*classes.edges, *classes.inputs)))
    hidden = {c: 2+i for i, c in enumerate(labels)} if mode == "source" else 3
    b = bind(original.graph, memories=KVConfig(width=4, summary_width=2, heads=2, memory_read="qry", window=30),
        source_classes=classes, source_hidden_width=hidden, source_binding=SourceBinding(mode, 2 if mode == "adapted" else None),
        alpha_agg=.2, dtype=torch.float64)
    spec = replace(spec, graph=b.graph)
    plan = SettleColumnPlan(b, spec, implementation=implementation)
    layout, ports = inputs(torch.float64)
    with observe_work(implementation) as work:
        first = plan.run_ports(layout, ports)
    assert work["metrics"]["packing/bucket/source_columns/calls"] > 0
    expected = reference(b.graph, b.model, Continuation(b.graph.identity, 2), external(spec, layout, ports), 50, sealed_until=50)
    equivalent(*observations(first, expected, spec))
    layout = replace(layout, times=layout.times+11)
    second = plan.run_ports(layout, ports, left=first.states)
    later = reference(b.graph, b.model, expected.continuation, external(spec, layout, ports, expected.continuation), 105, sealed_until=105)
    a, e = observations(second, later, spec)
    equivalent(a, e)
    roots = {name: p for name, p in b.model.named_parameters() if p.requires_grad}
    roots.update({f"input.{i}.{name}": getattr(p, name) for i, p in ports.items() for name in ("strength", "content")})
    equivalent(vjp(objective(a), roots), vjp(objective(e), roots))
    equivalent(vjp(objective(a), roots, zero=True), vjp(objective(e), roots, zero=True))
