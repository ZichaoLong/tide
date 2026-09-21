import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from isolated_cases import vjp


def fixture(dtype, kind, topology="chain"):
    count = 1 if topology == "self_loop" else 2
    g = Graph(tuple(Node(n, memory=kind, query_heads=2, kv_heads=1, window=3) for n in range(count)),
              (Edge(0, 0 if count == 1 else 1, 1),), (Region(1),) * count, (0,), (count-1,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 2)
    variables = dict(m.named_parameters())
    sources = [torch.full((3, 4), 0.2 + b / 10, dtype=dtype, requires_grad=True) for b in range(2)]
    variables.update({f"source.{b}": x for b, x in enumerate(sources)})
    # This caller-owned stack intentionally connects both source tensors: an
    # unused row of one public packed tensor has a connected-zero cotangent.
    values = torch.stack(sources)
    for b in range(2):
        for n in range(count):
            state = m.nodes[n].initial()
            state.value = torch.full_like(state.value, 0.2, requires_grad=True)
            state.slots = {k: torch.zeros_like(v, requires_grad=True) for k, v in state.slots.items()}
            q.states[b, n] = state
            variables[f"initial.{b}.{n}.value"] = state.value
            variables.update({f"initial.{b}.{n}.{k}": t for k, t in state.slots.items()})
    return g, m, q, values, variables


def check(expected, actual, variables, actual_variables):
    equivalent(expected, actual)
    a = next(x for b, _, _, x in expected.outputs if b == 0)
    b = next(x for b, _, _, x in actual.outputs if b == 0)
    for zero in (False, True):
        grad = vjp(a, variables, zero)
        equivalent(grad, vjp(b, actual_variables, zero))
        assert all(v is None for k, v in grad.items() if k.startswith("initial.1."))
        assert grad["source.1"] is not None and torch.count_nonzero(grad["source.1"]) == 0


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta"])
@pytest.mark.parametrize("topology", ["chain", "self_loop"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_isolated_specialization_roots(dtype, kind, topology, implementation):
    def execute(anchor):
        g, m, q, values, variables = fixture(dtype, kind, topology)
        xs = [External(b, 0, t, 2*t, values[b, t]) for b in range(2) for t in range(3)]
        if anchor:
            result = run(g, m, q, xs, 7, sealed_until=7, mode="hst")
        elif implementation == "python":
            result = specialized(g, m, q, xs, 7, sealed_until=7, topology=topology, mode="hst")
        else:
            result = Native(g, m, algorithm=topology, packed=True, workers=3, mode="hst").run(q, xs, 7, sealed_until=7)
        return result, variables
    expected, variables = execute(True); actual, actual_variables = execute(False)
    check(expected, actual, variables, actual_variables)


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta"])
@pytest.mark.parametrize("implementation", ["direct", "encoded-python", "encoded-native"])
def test_isolated_settle_roots_and_identity_adapters(dtype, kind, implementation):
    def execute(anchor):
        g, m, q, values, variables = fixture(dtype, kind)
        spec = SettleGraph(g, (1, 2))
        if anchor:
            result = settle_chain(spec, m, q, values, mode="hst")
        elif implementation == "direct":
            result = settle(spec, m, q, values, mode="hst")
        else:
            eg, em = spec.embed(m); eq = spec.embed_initial(q, eg)
            xs = spec.external(values, encoded=True); stop = spec.stride * values.shape[1]
            result = frontier(eg, em, eq, xs, stop, sealed_until=stop, mode="hst") if implementation == "encoded-python" else Native(
                eg, em, algorithm="frontier", packed=True, workers=3, mode="hst").run(eq, xs, stop, sealed_until=stop)
            result = spec.project(result)
        return result, variables
    expected, variables = execute(True); actual, actual_variables = execute(False)
    check(expected, actual, variables, actual_variables)
