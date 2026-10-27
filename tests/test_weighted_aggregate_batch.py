"""Architecture §2: each physical atom participates once despite shared class.

Literal per-atom formulas anchor projection/reduction; independent schedules
anchor complete routes, source identities, gradients and resumed execution.
"""
import math
import pytest
import torch
from tidegraph import Atom, Continuation, External, Graph, Node, Region
from tidegraph.aggregate import request
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, bind
from isolated_cases import vjp
from test_weighted_execution import setup, variables


@pytest.mark.parametrize("alpha", [0., .5])
def test_ragged_class_batch_literal_and_all_zero(dtype, alpha, monkeypatch):
    g = Graph((Node(0),), (), (Region(1),), (0, 0, 0, 0), (0,))
    classes = SourceClasses((), (0, 0, 1, 1))
    b = bind(g, memories=KVConfig(width=1, summary_width=1, epsilon=1.),
             source_classes=classes, alpha_agg=alpha, source_hidden_width=1, dtype=dtype)
    w = b.model.nodes[0]; program = w.aggregate_program
    with torch.no_grad():
        for label, transform in program.transforms.items():
            transform.a.fill_(.7); transform.b.fill_(-.4)
            transform.embedding.fill_(.2 if label == "0" else -.3)
    assert program.transforms["0"].a is program.transforms["1"].a
    fibers = (((0, .2, 2.), (1, .3, -.4), (2, .5, .7)),
              ((0, 0., 7.), (3, -.2, -9.)), ((2, .4, -.8),), ((1, .2, .5), (3, .8, -.2)))
    requests, expected = [], []
    for i, fiber in enumerate(fibers):
        atoms = [Atom(i, 0, i, 0, port, 0, torch.tensor([mass, value], dtype=dtype)) for port, mass, value in fiber]
        requests.append(request(b.graph, b.model, atoms))
        total = sum(max(0., mass) for _, mass, _ in fiber)
        terms = []
        for port, mass, value in fiber:
            hidden = .7*value/math.sqrt(1+value*value)+(.2 if classes.inputs[port] == 0 else -.3)
            transformed = value-alpha*.4*hidden/(1+math.exp(-hidden))
            terms.append(max(0., mass)*transformed)
        expected.append(torch.tensor([total, sum(terms)/total if total else 0.], dtype=dtype))
    def forbidden(*args, **kw):
        raise AssertionError("joint Aggregate called scalar step")
    monkeypatch.setattr(program, "step", forbidden)
    with torch.no_grad():
        actual = program.batch(w, requests)
    equivalent(expected, [r.value for r in actual])
    assert all(r.contributions == {} for r in actual) and program.joint_batch
    assert program.batch(w, []) == []


@pytest.mark.parametrize("topology", ["pdg", "dag"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("policy", ["replay", "batched"])
def test_packed_sources_complete_records_vjp_and_cuts(dtype, topology, native, policy):
    old, _, inputs, stop = setup(dtype, topology)
    classes = SourceClasses(tuple(i % 2 for i in range(len(old.graph.edges))), (0, 1))
    b = bind(old.graph, memories=old.memories, regions=old.regions, source_classes=classes,
             alpha_agg=.4, source_hidden_width=3, dtype=dtype)
    xs = [External(i, x.port, x.position, x.time, (x.value.detach()+x.value.new_tensor([0., .07*i, .03*i])).requires_grad_())
          for i in range(2) for x in inputs]
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    runner = Native(b.graph, b.model, algorithm="greedy", packed=True, aggregate_autograd=policy).run if native else lambda *a, **kw: greedy(b.graph, b.model, *a, aggregate_autograd=policy, **kw)
    actual = runner(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    roots = variables(b, xs)
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    for i in (0, len(expected.trace)//2):
        equivalent(vjp(expected.trace[i]["content"], roots), vjp(actual.trace[i]["content"], roots))
    equivalent(vjp(expected.outputs[0][-1], roots), vjp(actual.outputs[0][-1], roots))
    assert actual.stats.get("aggregate_scalar_fallback_steps", 0) == 0
    if policy == "batched":
        assert actual.stats.get("semantic_aggregate_replays", 0) == 0
        assert actual.stats["batched_aggregate_events"] > 0
    else:
        assert actual.stats["semantic_aggregate_replays"] > 0
    for cut in (2, 5):
        first = runner(q, [x for x in xs if x.time < cut], cut, sealed_until=cut)
        tail = [x for x in xs if x.time >= cut]
        a = reference(b.graph, b.model, first.continuation, tail, stop, sealed_until=stop)
        c = runner(first.continuation, tail, stop, sealed_until=stop)
        equivalent(a, c)
        equivalent(vjp(objective(a), roots), vjp(objective(c), roots))
    with torch.no_grad():
        equivalent(expected, runner(q, xs, stop, sealed_until=stop))
