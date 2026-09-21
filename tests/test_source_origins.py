from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.aggregate import AggregateProgram, AggregateResult
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.origins import InputOrigin, view
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from isolated_cases import vjp


class OrderedTags(AggregateProgram):
    profile = "ordered-tags-v1"

    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.2, dtype=dtype))

    def step(self, w, request):
        keys = [s.atom.key() for s in request.sources]
        assert keys == sorted(keys)
        terms = {}; value = 0
        for source in request.sources:
            atom = source.atom
            term = source.scale*atom.value + self.gain*(atom.kind+atom.position+atom.source+request.time)
            terms[source.slot] = term
            value = value*2 + term  # Ordering matters independently of the tags.
        return AggregateResult(value, dict(sorted(terms.items())))


def test_tag_sensitive_custom_aggregate_settle_embedding(dtype):
    g = Graph((Node(0), Node(0), Node(1, aggregation="ordered-tags-v1")),
              (Edge(0, 2, 1), Edge(1, 2, 1)), (Region(2), Region(1)), (0, 1, 2), (2,))
    def case():
        m = Model(g, dtype=dtype, aggregate_programs={2: OrderedTags(dtype)})
        x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3)/30).requires_grad_()
        return m, x, dict(m.named_parameters()) | {"input": x}
    spec = SettleGraph(g, (1, 2)); m, x, variables = case()
    expected = settle(spec, m, Continuation(g.identity, 2), x, mode="hst")
    m, x, actual_variables = case(); eg, em = spec.embed(m)
    assert em.nodes[2] is m.nodes[2]
    assert len(eg.origins) == 3
    actual = spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True),
                                   3*spec.stride, sealed_until=3*spec.stride, mode="hst"))
    equivalent(expected, actual)
    a_event = next(e for e in expected.trace if e["node"] == 2 and e["batch"] == 0)
    b_event = next(e for e in actual.trace if e["node"] == 2 and e["batch"] == 0)
    for a, b in ((objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1]),
                 (a_event["contributions"][0], b_event["contributions"][0])):
        equivalent(vjp(a, variables), vjp(b, actual_variables))


@pytest.mark.parametrize("origin", [InputOrigin(-1, 0, 1), InputOrigin(1, 0, 1), InputOrigin(0, -1, 1),
                                    InputOrigin(0, 0, 0), InputOrigin(0, True, 1)])
def test_invalid_origin_mapping(origin):
    with pytest.raises(ValueError, match="origin"):
        Graph((Node(0),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,), origins=(origin,))


def test_duplicate_origin_and_identity_clock_guards(dtype, tmp_path):
    g = Graph((Node(0),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,))
    origin = InputOrigin(0, 7, 3)
    with pytest.raises(ValueError, match="origin"):
        replace(g, origins=(origin, origin))
    changed = replace(g, origins=(origin,))
    assert changed.identity != g.identity
    m = Model(g, dtype=dtype); path = tmp_path / "origin.pt"
    save(path, g, m, Continuation(g.identity, 1))
    restored = Model(changed, dtype=dtype, seed=53)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, restored)
    equivalent(before, restored.state_dict())
    from tidegraph import Atom
    raw = Atom(0, 0, 4, 1, 0, 3, torch.ones(3, dtype=dtype))
    projected = view(changed, raw)
    assert (projected.kind, projected.source, projected.position, projected.time) == (0, 7, 1, 4)
    assert projected.value is raw.value
    with pytest.raises(ValueError, match="clock"):
        view(changed, replace(raw, position=2))


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_execution_rejects_misaligned_origin_clock(dtype, implementation):
    g = Graph((Node(0),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,), origins=(InputOrigin(0, 7, 3),))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    xs = [External(0, 0, 0, 1, torch.ones(3, dtype=dtype))]
    with pytest.raises((ValueError, RuntimeError), match="origin clock"):
        run(g, m, q, xs, 3, sealed_until=3) if implementation == "python" else Native(
            g, m, packed=True).run(q, xs, 3, sealed_until=3)
