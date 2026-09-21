import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.checkpoint import load, save


def fixture(dtype, kind, clear=False, cyclic=False, aligned=False):
    kinds = ("linear", "ssm", "delta") if kind == "mixed" else (kind,) * 3
    g = Graph(tuple(Node(0 if i < 2 else 1, clear=clear and i == 1, memory=k, full="swiglu")
                    for i, k in enumerate(kinds)),
              (Edge(0, 2, 1), Edge(1, 2, 1 if aligned else 2)) + ((Edge(2, 0, 1),) if cyclic else ()),
              (Region(1, observe_all=not clear), Region(1)), (0, 1), (2,))
    m = Model(g, dtype=dtype)
    x = torch.sin(torch.arange(48, dtype=dtype) * 0.23).reshape(16, 3).requires_grad_()
    q = Continuation(g.identity, 2)
    variables = dict(m.named_parameters()) | {"input": x}
    for b in range(2):
        for v in range(3):
            s = m.nodes[v].initial()
            s.value = torch.full_like(s.value, 0.11, requires_grad=True)
            s.slots = {k: torch.full_like(t, 0.11, requires_grad=True) for k, t in s.slots.items()}
            q.states[b, v] = s
            variables[f"initial.{b}.{v}.value"] = s.value
            variables.update({f"initial.{b}.{v}.{k}": t for k, t in s.slots.items()})
    xs = [External(b, p, i, t, x[b * 8 + p * 4 + i]) for b in range(2)
          for p in range(2) for i, t in enumerate((0, 2, 5, 6))]
    return g, m, q, xs, x, variables


def vjp(result, variables):
    return dict(zip(variables, torch.autograd.grad(objective(result), list(variables.values()), allow_unused=True)))


@pytest.mark.parametrize("kind", ["linear", "delta", "mixed"])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "native-streaming", "native-frontier"])
def test_matrix_profiles_trace_vjp(dtype, kind, clear, mode, implementation):
    g, m, q, xs, _, variables = fixture(dtype, kind, clear)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode=mode)
    expected_grad = vjp(expected, variables)
    g, m, q, xs, _, variables = fixture(dtype, kind, clear)
    actual = frontier(g, m, q, xs, 10, sealed_until=10, mode=mode) if implementation == "python" else Native(
        g, m, algorithm=implementation.split("-")[1], workers=3, packed=True, mode=mode).run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual); equivalent(expected_grad, vjp(actual, variables))


@pytest.mark.parametrize("kind", ["linear", "delta"])
@pytest.mark.parametrize("packed", [False, True])
def test_matrix_cycles_and_value_checkpoint(dtype, kind, packed, tmp_path):
    g, m, q, xs, _, variables = fixture(dtype, kind, True, cyclic=True)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode="hst")
    expected_grad = vjp(expected, variables)
    g, m, q, xs, _, variables = fixture(dtype, kind, True, cyclic=True)
    actual = Native(g, m, workers=3, packed=packed, mode="hst").run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual); equivalent(expected_grad, vjp(actual, variables))
    path = tmp_path / "matrix.pt"; save(path, g, m, actual.continuation)
    restored = load(path, g, m); equivalent(actual.continuation, restored)
    expected = run(g, m, actual.continuation, [], 12, sealed_until=12)
    actual = Native(g, m, packed=packed).run(restored, [], 12, sealed_until=12)
    equivalent(expected, actual)


@pytest.mark.parametrize("kind", ["ssm", "linear", "delta", "mixed"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_advanced_settle_embedding(dtype, kind, implementation):
    g, m, q, _, x, variables = fixture(dtype, kind, aligned=True)
    spec = SettleGraph(g, (1, 2))
    values = x[:8].reshape(2, 4, 3)
    expected = settle(spec, m, q, values, mode="hst")
    expected_grad = vjp(expected, variables)
    g, m, q, _, x, variables = fixture(dtype, kind, aligned=True)
    spec = SettleGraph(g, (1, 2)); values = x[:8].reshape(2, 4, 3)
    eg, em = spec.embed(m); eq = spec.embed_initial(q, eg)
    inputs = spec.external(values, encoded=True); stop = spec.stride * values.shape[1]
    result = frontier(eg, em, eq, inputs, stop, sealed_until=stop, mode="hst") if implementation == "python" else Native(
        eg, em, algorithm="frontier", workers=3, mode="hst").run(eq, inputs, stop, sealed_until=stop)
    actual = spec.project(result)
    equivalent(expected, actual); equivalent(expected_grad, vjp(actual, variables))


@pytest.mark.parametrize("kind", ["linear", "delta"])
def test_matrix_memory_analytic_vjp(dtype, kind):
    g = Graph((Node(0, memory=kind),), (), (Region(1),), (0,), (0,))
    w = Model(g, width=1, dtype=dtype).nodes[0]
    with torch.no_grad():
        for p in w.extra.values():
            p.zero_()
        for name in (("mem_v", "mem_out") if kind == "linear" else ("mem_q", "mem_k", "mem_v", "mem_out")):
            w.extra[name].fill_(1)
    h = torch.tensor([[1.0], [2.0]], dtype=dtype, requires_grad=True)
    matrix = torch.tensor([[0.2 if kind == "linear" else 0.4]], dtype=dtype, requires_grad=True)
    old = State(torch.zeros(1, dtype=dtype), slots={"matrix": matrix})
    if kind == "linear":
        old.slots["normalizer"] = torch.tensor([0.5], dtype=dtype, requires_grad=True)
    first, _ = w.prepare(old, h[0], 1); second, _ = w.prepare(first, h[1], 4)
    expected = 3.2 / 2.500001 if kind == "linear" else 1.15
    equivalent(second.value, h.new_tensor([expected]))
    dh, dm = torch.autograd.grad(second.value.sum(), (h, matrix))
    equivalent(dh, h.new_tensor([[1 / 2.500001], [1 / 2.500001]]) if kind == "linear" else h.new_tensor([[0.125], [0.5]]))
    equivalent(dm, matrix.new_tensor([[1 / 2.500001 if kind == "linear" else 0.0625]]))
    states, _ = w.prepare_block(old, h, [1, 4]); equivalent(states, [first, second])


def test_zero_message_still_updates_linear_normalizer(dtype):
    g = Graph((Node(0, memory="linear", full="swiglu"),), (Edge(0, 0, 1),), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    xs = [External(0, 0, 0, 0, torch.zeros(3, dtype=dtype))]
    expected = run(g, m, q, xs, 5, sealed_until=5)
    actual = Native(g, m, packed=True).run(q, xs, 5, sealed_until=5)
    equivalent(expected, actual)
    assert len(actual.outputs) == 5 and actual.continuation.history[0, 0].node_maps["selected"][0] == 5
    equivalent(actual.continuation.states[0, 0].slots["normalizer"], torch.full((3,), 5.0, dtype=dtype))
