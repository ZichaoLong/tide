"""Declared FP32 Read precision, independent of payload precision and schedule."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.content import Content
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.readout import NormFloat32Read, ReadInput, validate_program
from isolated_cases import vjp
from read_cases import execute, fixture


def candidate(implementation, g, m, q, xs, stop=6):
    if implementation == "python-greedy":
        return greedy(g, m, q, xs, stop, sealed_until=stop, mode="hst")
    if implementation == "native-greedy":
        return Native(g, m, algorithm="greedy", mode="hst").run(q, xs, stop, sealed_until=stop)
    return execute(implementation, g, m, q, xs, stop=stop)


@pytest.mark.parametrize("implementation", ["python-frontier", "python-causal", "python-greedy",
                                           "native-serial", "native-packed", "native-greedy"])
@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
def test_norm32_complete_observables_vjps_and_continuation(dtype, implementation, read_mode):
    g, m, q, xs, leaves = fixture(dtype, read_mode, observe_all=False, clear=True)
    g = replace(g, nodes=tuple(replace(n, readout="norm-fp32-v1") for n in g.nodes))
    q.identity = g.identity
    for w in m.nodes:
        w.read_program = NormFloat32Read()
    expected = execute("reference", g, m, q, xs)
    actual = candidate(implementation, g, m, q, xs)
    equivalent(actual, expected)
    assert all(e["descriptor"].dtype == torch.float32 and e["control"].dtype == dtype for e in actual.trace)
    for root in (objective, lambda r: r.trace[0]["descriptor"], lambda r: r.trace[-1]["control"]):
        equivalent(vjp(root(actual), leaves), vjp(root(expected), leaves))
    gradients = vjp(actual.trace[0]["descriptor"], leaves)
    assert all(value is None for name, value in gradients.items() if name.endswith(".read"))
    prefix = candidate(implementation, g, m, q, [x for x in xs if x.time < 3], 3)
    suffix = candidate(implementation, g, m, prefix.continuation, [x for x in xs if x.time >= 3])
    equivalent(suffix.continuation, actual.continuation)


@pytest.mark.parametrize("implementation", ["reference", "python-greedy", "native-serial", "native-greedy"])
def test_norm32_explicit_precision_can_change_a_rounded_tie(dtype, implementation):
    def run(profile):
        g = Graph((Node(0, readout=profile), Node(0, readout=profile)), (),
                  (Region(1, count_priority=False, read_mode="content"),), (0, 1), (0, 1))
        m = Model(g, width=2, dtype=dtype)
        with torch.no_grad():
            for scale in m.input_scale:
                scale.fill_(1)
        xs = [External(0, n, 0, 0, torch.tensor([1e6, n+1.], dtype=dtype)) for n in range(2)]
        result = candidate(implementation, g, m, Continuation(g.identity, 1), xs, 1)
        return g, result
    low, a = run("norm-fp32-v1")
    high, b = run("norm-fp64-v1")
    assert low.identity != high.identity
    assert [e["node"] for e in a.trace if e["active"]] == [0]
    assert [e["node"] for e in b.trace if e["active"]] == [1]
    assert a.trace[0]["descriptor"].dtype == torch.float32
    assert b.trace[0]["descriptor"].dtype == torch.float64


@pytest.mark.parametrize("payload", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("values", [[3., 4.], [0., 0.]])
def test_norm32_analytic_conversion_and_connected_zero(payload, values):
    x = torch.tensor(values, dtype=payload, requires_grad=True)
    unused = torch.ones((), dtype=payload, requires_grad=True)
    program = NormFloat32Read()
    result = program.step(None, ReadInput(None, 0, Content(x)))
    expected = torch.tensor([.6, .8] if values[0] else [0., 0.], dtype=torch.float32).to(payload)
    gradient, disconnected = torch.autograd.grad(result, (x, unused), allow_unused=True)
    assert result.dtype == torch.float32 and result.item() == (5. if values[0] else 0.)
    assert gradient is not None and torch.equal(gradient, expected)
    assert disconnected is None


def test_norm32_profile_rejects_undeclared_precision(dtype):
    g = Graph((Node(0, readout="norm-fp32-v1"),), (), (Region(1),), (0,), ())
    m = Model(g, width=2, dtype=dtype)
    m.nodes[0].read_program.precision = "float64"
    with pytest.raises(ValueError, match="precision"):
        validate_program(m.nodes[0], g.nodes[0], native=True)
