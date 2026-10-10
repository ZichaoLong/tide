"""Internal row views must not leak storage/version coupling into public states."""
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State, StateClock
from tidegraph.compare import equivalent
from tidegraph.ops import Model
from read_cases import execute


def case(dtype, clock, rho):
    g = Graph((Node(0, memory="lh-add-repeat-v1", state_clock=clock),), (),
              (Region(1),), (0,), (0,))
    m = Model(g, width=3, dtype=dtype)
    with torch.no_grad():
        m.nodes[0].extra["add_retention"].fill_(rho)
    # Noncontiguous rows and an unconnected row are both intentional.
    initial = [torch.tensor([.13, 7., -.29, 8., .37, 9.], dtype=dtype)[::2].requires_grad_()
               for _ in range(3)]
    inputs = [torch.full((3,), .11+b/13, dtype=dtype, requires_grad=True) for b in range(3)]
    q = Continuation(g.identity, 3, states={(b, 0): State(x) for b, x in enumerate(initial)})
    time = clock.to_global(3)
    xs = [External(b, 0, 0, time, x) for b, x in enumerate(inputs)]
    leaves = initial + inputs + [m.nodes[0].extra["add_retention"], m.nodes[0].decay]
    return g, m, q, xs, time+1, leaves


@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
@pytest.mark.parametrize("clock", [StateClock(), StateClock(4, 3, 1)])
@pytest.mark.parametrize("rho", [0., .71])
def test_add_partial_roots_and_independent_public_versions(dtype, implementation, clock, rho):
    g, m, q, xs, stop, leaves = case(dtype, clock, rho)
    expected = execute("reference", g, m, q, xs, stop=stop)
    actual = execute(implementation, g, m, q, xs, stop=stop)
    equivalent(expected, actual)
    def roots(result):
        states = result.continuation.states
        return states[0, 0].value.sum()*.3 + states[1, 0].value.sum()*0
    a = torch.autograd.grad(roots(expected), leaves, allow_unused=True, retain_graph=True)
    b = torch.autograd.grad(roots(actual), leaves, allow_unused=True, retain_graph=True)
    equivalent(a, b)
    assert b[1] is not None and torch.count_nonzero(b[1]) == 0
    assert b[2] is None and b[5] is None and b[-1] is None
    states = actual.continuation.states
    saved = states[1, 0].value.detach().clone()
    # Keep an outstanding graph which saves row 1. Mutating row 0 must neither
    # alter row 1 nor increment its version counter through shared row storage.
    loss = states[1, 0].value.square().sum()
    with torch.no_grad():
        states[0, 0].value.add_(1)
    assert torch.equal(states[1, 0].value, saved)
    torch.autograd.grad(loss, leaves, allow_unused=True)


@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_add_batch_retains_original_input_version_checks(dtype, implementation):
    g, m, q, xs, stop, leaves = case(dtype, StateClock(), .71)
    actual = execute(implementation, g, m, q, xs, stop=stop)
    with torch.no_grad():
        leaves[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(actual.continuation.states[0, 0].value.sum(), leaves, allow_unused=True)
