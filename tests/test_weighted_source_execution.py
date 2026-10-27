"""Architecture §2.2: independent schedules and roots with source bindings.

Local scalar derivatives live in test_weighted_source_binding. These checks
cover actual class-width buckets, aliases, all returns and pending cuts.
"""
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.reference import run as reference
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import bind
from isolated_cases import vjp
from test_weighted_execution import setup, variables
from test_weighted_source_binding import MODES, binding


def source_setup(dtype, family, mode):
    old, _, xs, stop = setup(dtype, family)
    classes = SourceClasses(tuple(i % 2 for i in range(len(old.graph.edges))),
                            tuple(i % 2 for i in range(len(old.graph.inputs))))
    b = bind(old.graph, memories=old.memories, regions=old.regions, source_classes=classes,
             source_hidden_width={0: 2, 1: 3} if mode == "source" else 3,
             source_binding=binding(mode), alpha_agg=.4, dtype=dtype)
    xs = [External(i, x.port, x.position, x.time,
                   (x.value.detach()+x.value.new_tensor([0., .07*i, -.03*i])).requires_grad_())
          for i in range(2) for x in xs]
    return b, xs, stop


def runner(b, native):
    opts = dict(aggregate_autograd="batched", full_autograd="batched")
    return (Native(b.graph, b.model, algorithm="greedy", packed=True, **opts).run if native else
            lambda *a, **kw: greedy(b.graph, b.model, *a, **opts, **kw))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", ["hb", "pdg"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_shared_sources_complete_records_and_independent_roots(dtype, mode, family, native):
    b, xs, stop = source_setup(dtype, family, mode)
    q = Continuation(b.graph.identity, 2)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    execute = runner(b, native)
    actual = execute(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert actual.stats["batched_aggregate_events"] > 0
    assert actual.stats.get("semantic_aggregate_replays", 0) == 0
    roots = variables(b, xs)
    for a, c in ((objective(expected), objective(actual)),
                 (expected.trace[0]["content"], actual.trace[0]["content"]),
                 (expected.outputs[0][-1], actual.outputs[0][-1])):
        equivalent(vjp(a, roots), vjp(c, roots))
        equivalent(vjp(a*0, roots), vjp(c*0, roots))
    first = execute(q, [x for x in xs if x.time < 2], 2, sealed_until=2)
    assert first.continuation.pending
    tail = [x for x in xs if x.time >= 2]
    a = reference(b.graph, b.model, first.continuation, tail, stop, sealed_until=stop)
    c = execute(first.continuation, tail, stop, sealed_until=stop)
    equivalent(a, c)
    equivalent(vjp(objective(a), roots), vjp(objective(c), roots))
