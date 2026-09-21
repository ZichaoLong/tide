import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.frontier import run as frontier
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from attention_cases import fixture, vjp


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_attention_settle_embedding(dtype, implementation, mode):
    g, m, q, _, x, variables = fixture(dtype, aligned=True)
    spec = SettleGraph(g, (1, 2)); values = x[:, 0]
    expected = settle(spec, m, q, values, mode=mode)
    grad = vjp(expected, variables)
    g, m, q, _, x, variables = fixture(dtype, aligned=True)
    spec = SettleGraph(g, (1, 2)); values = x[:, 0]
    eg, em = spec.embed(m); eq = spec.embed_initial(q, eg)
    inputs = spec.external(values, encoded=True); stop = spec.stride * values.shape[1]
    result = frontier(eg, em, eq, inputs, stop, sealed_until=stop, mode=mode) if implementation == "python" else Native(
        eg, em, algorithm="frontier", packed=True, workers=3, mode=mode).run(eq, inputs, stop, sealed_until=stop)
    actual = spec.project(result)
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))


def chain_case(dtype, topology):
    n = 1 if topology == "self_loop" else 3
    edges = (Edge(0, 0, 2),) if n == 1 else (Edge(0, 1, 1), Edge(1, 2, 1))
    g = Graph(tuple(Node(i, memory="attention", query_heads=2, window=3) for i in range(n)),
              edges, (Region(1),) * n, (0,), (n-1,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 2)
    x = (torch.arange(32, dtype=dtype).reshape(2, 4, 4) / 30).requires_grad_()
    xs = [External(b, 0, p, t, x[b, p]) for b in range(2) for p, t in enumerate((0, 1, 3, 6))]
    return g, m, q, x, xs, dict(m.named_parameters()) | {"input": x}


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_attention_specializations(dtype, topology, implementation):
    g, m, q, _, xs, variables = chain_case(dtype, topology)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode="hst"); grad = vjp(expected, variables)
    g, m, q, _, xs, variables = chain_case(dtype, topology)
    actual = specialized(g, m, q, xs, 10, sealed_until=10, topology=topology, mode="hst") if implementation == "python" else Native(
        g, m, algorithm=topology, packed=True, workers=3, mode="hst").run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))


def test_attention_settle_chain_anchor(dtype):
    g, m, q, x, _, variables = chain_case(dtype, "chain"); spec = SettleGraph(g, (1, 2, 3))
    expected = settle(spec, m, q, x, mode="hst"); grad = vjp(expected, variables)
    g, m, q, x, _, variables = chain_case(dtype, "chain")
    actual = settle_chain(SettleGraph(g, (1, 2, 3)), m, q, x, mode="hst")
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))
