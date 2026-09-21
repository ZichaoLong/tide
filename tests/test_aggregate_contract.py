from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, PortLayout, Region
from tidegraph.aggregate import AggregateProgram, AggregateResult, SourceAggregate
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run


class ClockFiber(AggregateProgram):
    profile = "clock-fiber-v1"

    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2., dtype=dtype))

    def step(self, weights, request):
        assert request.slots == 3
        terms = {s.slot: ((s.slot+1)*s.atom.value*s.scale+s.atom.position+s.atom.kind) *
                 (request.time+1)*self.gain for s in request.sources}
        return AggregateResult(sum(terms.values()), dict(sorted(terms.items())))


def test_custom_program_tags_local_slots_time_and_registered_parameter(dtype):
    g = Graph((Node(0, aggregation="clock-fiber-v1"),), (Edge(0, 0, 1),), (Region(1),), (0, 0), (0,),
              PortLayout((0,), (2,), (1, 0), (1,)))
    def case():
        program = ClockFiber(dtype); m = Model(g, width=2, dtype=dtype, aggregate_programs={0: program})
        with torch.no_grad():
            m.nodes[0].weight.zero_(); m.nodes[0].bias.zero_()
            for group in (m.input_scale, m.agg_scale, m.edge_scale, m.output_scale):
                for value in group:
                    value.fill_(1)
        x = torch.ones(2, dtype=dtype, requires_grad=True); y = (x.detach()*3).requires_grad_()
        unused = torch.ones(2, dtype=dtype, requires_grad=True)
        xs = [External(0, 0, 0, 0, x), External(0, 1, 0, 1, y),
              External(1, 0, 0, 0, unused), External(1, 1, 0, 1, unused)]
        return m, xs, (x, y, unused, program.gain)
    m, xs, variables = case(); q = Continuation(g.identity, 2)
    # This cyclic profile exercises independent scalar execution; batching is
    # exercised by a DAG with the same custom program below.
    result = run(g, m, q, xs, 2, sealed_until=2)
    loss = sum(value.sum() for batch, _, _, value in result.outputs if batch == 0)
    equivalent(loss, loss.new_tensor(136.))
    dx, dy, other, gain = torch.autograd.grad(loss, variables, allow_unused=True)
    equivalent(dx, torch.full_like(dx, 52)); equivalent(dy, torch.full_like(dy, 4))
    assert other is None
    equivalent(gain, gain.new_tensor(116.))
    assert dict(m.named_parameters())["nodes.0.aggregate_program.gain"] is variables[-1]
    with pytest.raises(ValueError, match="no native implementation"):
        Native(g, m)
    dag = Graph((Node(0, aggregation="clock-fiber-v1"),), (), (Region(1),), (0, 0, 0), (0,))
    dm = Model(dag, dtype=dtype, aggregate_programs={0: ClockFiber(dtype)})
    dq = Continuation(dag.identity, 2)
    inputs = [External(b, p, t, t, torch.ones(3, dtype=dtype)) for b in range(2) for p in (0, 2) for t in range(3)]
    expected = run(dag, dm, dq, inputs, 3, sealed_until=3)
    actual = frontier(dag, dm, dq, inputs, 3, sealed_until=3)
    equivalent(expected, actual)
    assert actual.stats["aggregate_scalar_fallback_steps"] == 6


@pytest.mark.parametrize("fault", ["absent-slot", "shape", "count", "presence"])
def test_custom_aggregate_batch_invalid_contract(dtype, fault):
    class Broken(ClockFiber):
        def batch(self, w, requests):
            results = super().batch(w, requests)
            if fault == "absent-slot":
                results[0].contributions = {1: results[0].value}
            elif fault == "shape":
                results[0].value = results[0].value[:1]
            elif fault == "count":
                results.pop()
            else:
                results[0].contributions = {}
            return results
    g = Graph((Node(0, aggregation="clock-fiber-v1"),), (), (Region(1),), (0, 0, 0), (0,))
    m = Model(g, dtype=dtype, aggregate_programs={0: Broken(dtype)})
    with pytest.raises(ValueError, match="Aggregate"):
        frontier(g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))], 1, sealed_until=1)


def test_profile_slot_domain_and_native_subclass_guards(dtype):
    g = Graph((Node(0, aggregation="all_softmax"), Node(1, aggregation="all_softmax")), (), (Region(1),)*2, (0, 0, 1), (0, 1))
    m = Model(g, dtype=dtype); m.nodes[1] = m.nodes[0]
    for native in (False, True):
        with pytest.raises(ValueError, match="slot domain"):
            Native(g, m) if native else run(g, m, Continuation(g.identity, 1), [], 0, sealed_until=0)
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,)); m = Model(g, dtype=dtype)
    changed = replace(g, nodes=(Node(0, aggregation="mean"),))
    with pytest.raises(ValueError, match="profile"):
        Native(changed, m)
    class Altered(SourceAggregate):
        pass
    m.nodes[0].aggregate_program = Altered()
    with pytest.raises(ValueError, match="no native implementation"):
        Native(g, m)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_numerically_zero_weighted_mass_fails(dtype, implementation):
    g = Graph((Node(0, aggregation="weighted_mean"),), (), (Region(1),), (0,), (0,)); m = Model(g, dtype=dtype)
    with torch.no_grad():
        m.nodes[0].extra["agg_mass_0"].fill_(-10000)
    q = Continuation(g.identity, 1); xs = [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))]
    with pytest.raises((ValueError, RuntimeError), match="zero mass"):
        frontier(g, m, q, xs, 1, sealed_until=1) if implementation == "python" else Native(
            g, m, packed=True).run(q, xs, 1, sealed_until=1)
