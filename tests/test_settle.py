import pytest
import torch
from tidegraph import Continuation, Edge, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run as streaming
from tidegraph.frontier import run as frontier
from tidegraph.settle import SettleGraph, run
from tidegraph.records import Result
from cases import gradients


def fixture(dtype, clear):
    g = Graph((Node(0), Node(1, clear=clear), Node(1), Node(2)),
              (Edge(0, 1, 1), Edge(0, 2, 1), Edge(1, 3, 2), Edge(2, 3, 2)),
              (Region(1), Region(1, observe_all=not clear), Region(1)), (0,), (2, 3))
    spec = SettleGraph(g, (1, 2, 4))
    model = Model(g, dtype=dtype)
    x = (torch.arange(30, dtype=dtype).reshape(2, 5, 3) / 50 - 0.2).requires_grad_()
    initial = torch.full((2, 4, 3), 0.11, dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 2, states={(b, v): State(initial[b, v]) for b in range(2) for v in range(4)})
    return spec, model, q, x, initial


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python-streaming", "python-frontier", "native-streaming", "native-frontier"])
def test_settle_embedding_trace_vjp(dtype, clear, mode, implementation):
    spec, model, q, x, initial = fixture(dtype, clear)
    expected = run(spec, model, q, x, mode=mode)
    expected_grad = gradients(objective(expected), model, x, initial)
    spec, model, q, x, initial = fixture(dtype, clear)
    g, em = spec.embed(model)
    eq = spec.embed_initial(q, g)
    xs = spec.external(x, encoded=True)
    stop = spec.stride * x.shape[1]
    if implementation.startswith("python"):
        execute = streaming if implementation.endswith("streaming") else frontier
        actual = execute(g, em, eq, xs, stop, sealed_until=stop, mode=mode)
    else:
        actual = Native(g, em, workers=3, packed=True, mode=mode, algorithm=implementation.split("-")[1]).run(
            eq, xs, stop, sealed_until=stop)
    equivalent(expected, spec.project(actual))
    equivalent(expected_grad, gradients(objective(spec.project(actual)), model, x, initial))
    if implementation.endswith("frontier") and not clear:
        assert actual.stats["frontier_stages"] == 5  # input + three regions + output


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_settle_chunk_decode(dtype, implementation):
    spec, model, q, x, initial = fixture(dtype, True)
    expected = run(spec, model, q, x, mode="hst")
    expected_grad = gradients(objective(expected), model, x, initial)
    spec, model, q, x, initial = fixture(dtype, True)
    g, em = spec.embed(model)
    engine = Native(g, em, algorithm="frontier", workers=3, mode="hst") if implementation == "native" else None
    if engine:
        q = spec.embed_initial(q, g)
    events, outputs, messages = [], [], []
    begin = 0
    for end in (1, 3, 3, 5):
        part = x[:, begin:end]
        if engine:
            piece = engine.run(q, spec.external(part, begin, encoded=True), end * spec.stride, sealed_until=end * spec.stride)
            q = piece.continuation; piece = spec.project(piece)
        else:
            piece = run(spec, model, q, part, mode="hst"); q = piece.continuation
        events += piece.trace; outputs += piece.outputs; messages += piece.messages
        begin = end
    actual = Result(piece.continuation, events, outputs, messages, {})
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), model, x, initial))


def test_settle_rejects_bad_ranks_and_partial_projection(dtype):
    spec, model, q, x, _ = fixture(dtype, False)
    with pytest.raises(ValueError, match="ranks"):
        SettleGraph(spec.graph, (1, 3, 4))
    g, em = spec.embed(model)
    partial = streaming(g, em, spec.embed_initial(q, g), spec.external(x[:, :1], encoded=True), 1, sealed_until=1)
    with pytest.raises(ValueError, match="position cut"):
        spec.project(partial)
