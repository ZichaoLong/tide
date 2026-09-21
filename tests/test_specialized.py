import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from cases import gradients


def fixture(dtype, topology, clear):
    n = 1 if topology == "self_loop" else 3
    edges = (Edge(0, 0, 2),) if n == 1 else (Edge(0, 1, 1), Edge(1, 2, 2))
    g = Graph(tuple(Node(i, clear=clear) for i in range(n)), edges, (Region(1),) * n, (0,), (n - 1,))
    model = Model(g, dtype=dtype)
    x = (torch.arange(24, dtype=dtype).reshape(8, 3) / 50 - 0.2).requires_grad_()
    xs = [External(b, 0, p, t, x[b * 4 + p]) for b in range(2) for p, t in enumerate((0, 1, 3, 6))]
    initial = torch.full((2, n, 3), 0.17, dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 2, states={(b, v): State(initial[b, v]) for b in range(2) for v in range(n)})
    return g, model, q, xs, x, initial


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_specializations_trace_vjp(dtype, topology, clear, mode, implementation):
    g, m, q, xs, x, initial = fixture(dtype, topology, clear)
    expected = run(g, m, q, xs, 9, sealed_until=9, mode=mode)
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = fixture(dtype, topology, clear)
    actual = specialized(g, m, q, xs, 9, sealed_until=9, topology=topology, mode=mode) if implementation == "python" else Native(
        g, m, algorithm=topology, workers=3, mode=mode).run(q, xs, 9, sealed_until=9)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))


@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_settle_chain_anchor(dtype, mode):
    g, m, q, _, x, initial = fixture(dtype, "chain", True)
    spec = SettleGraph(g, (1, 2, 4))
    values = x.reshape(2, 4, 3)
    expected = settle(spec, m, q, values, mode=mode)
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, _, x, initial = fixture(dtype, "chain", True)
    actual = settle_chain(spec, m, q, x.reshape(2, 4, 3), mode=mode)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))
