import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.compare import equivalent
from tidegraph.lazy_add import decode
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from add_cases import PROFILE, native_decode


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_add_analytic_value_clock_and_physical_vjp(dtype, implementation):
    g = Graph((Node(0, memory=PROFILE),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=1, dtype=dtype); w = m.nodes[0]
    with torch.no_grad():
        w.extra["add_retention"].fill_(.5); m.input_scale[0].fill_(1)
    initial = torch.tensor([2.], dtype=dtype, requires_grad=True)
    x = torch.tensor([[1.], [2.]], dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 1, states={(0, 0): State(initial)})
    xs = [External(0, 0, p, t, x[p]) for p, t in enumerate((2, 5))]
    result = run(g, m, q, xs, 8, sealed_until=8) if implementation == "python" else Native(
        g, m, algorithm="frontier", packed=True).run(q, xs, 8, sealed_until=8)
    equivalent([e["proposal"] for e in result.trace], [x.new_tensor([1.25]), x.new_tensor([2.15625])])
    s = result.continuation.states[0, 0]
    assert (s.last_time, s.observations) == (5, 2)
    fn = decode if implementation == "python" else native_decode
    value = fn(w, s, 8); equivalent(value, x.new_tensor([.5390625]))
    grads = torch.autograd.grad(value.sum(), (x, initial, w.extra["add_retention"], w.decay), allow_unused=True)
    equivalent(grads, (x.new_tensor([[.03125], [.25]]), x.new_tensor([.00390625]), x.new_tensor(2.4375), None))
    # Decoding is a read, never a state/event/count update.
    assert (s.last_time, s.observations) == (5, 2)
    equivalent(s.value, x.new_tensor([2.15625]))


@pytest.mark.parametrize("rho", [0., 1., .99, -.5])
def test_add_repeat_matches_literal_eager_ticks_and_zero_connections(dtype, rho):
    g = Graph((Node(0, memory=PROFILE),), (), (Region(1),), (0,), ())
    w = Model(g, width=3, dtype=dtype).nodes[0]
    with torch.no_grad():
        w.extra["add_retention"].fill_(rho)
    h = torch.tensor([.13, -.23, .37], dtype=dtype, requires_grad=True)
    state = State(h); eager = h
    for time in range(27):
        eager = eager * w.extra["add_retention"]
        if time in (0, 3, 17):
            content = h * (time + 1)
            eager = content + eager
            state = w.propose(state, content, time)
        assert torch.equal(decode(w, state, time+1), eager)
        assert torch.equal(native_decode(w, state, time+1), eager)
    for value in (decode(w, State(h), 0), native_decode(w, State(h), 0)):
        assert torch.autograd.grad(value.sum(), w.extra["add_retention"], allow_unused=True, retain_graph=True)[0] is None
    # A real tick multiplying zero still has a connected zero retention VJP.
    zero = State(torch.zeros(3, dtype=dtype, requires_grad=True))
    for fn in (decode, native_decode):
        grad = torch.autograd.grad(fn(w, zero, 1).sum(), w.extra["add_retention"])[0]
        assert grad is not None and grad.item() == 0


@pytest.mark.parametrize("native", [False, True])
def test_add_rejects_invalid_weights_clocks_and_slots(dtype, native):
    g = Graph((Node(0, memory=PROFILE),), (), (Region(1),), (0,), ())
    w = Model(g, width=1, dtype=dtype).nodes[0]; fn = native_decode if native else decode
    for state, cut in [(State(w.bias), -1), (State(w.bias, 3), 3), (State(w.bias, -2), 2),
                       (State(w.bias, observations=-1), 1), (State(w.bias, slots={"x": w.bias}), 1)]:
        with pytest.raises((ValueError, RuntimeError)):
            fn(w, state, cut)
    for rho in (torch.ones(1, dtype=dtype), torch.tensor(float("nan"), dtype=dtype),
                torch.tensor(.5, dtype=torch.float32 if dtype == torch.float64 else torch.float64)):
        w.extra["add_retention"] = torch.nn.Parameter(rho)
        with pytest.raises((ValueError, RuntimeError), match="retention"):
            fn(w, State(w.bias), 1)


def test_add_old_read_is_stored_representation(dtype):
    g = Graph((Node(0, memory=PROFILE),), (), (Region(1, read_mode="old"),), (0,), ())
    w = Model(g, width=1, dtype=dtype).nodes[0]
    with torch.no_grad():
        w.extra["add_retention"].fill_(.5); w.read.fill_(1)
    old = State(torch.tensor([8.], dtype=dtype))
    prop, desc = w.prepare(old, torch.tensor([1.], dtype=dtype), 2, "old")
    equivalent(prop.value, w.bias.new_tensor([2.])); equivalent(desc, w.bias.new_tensor(8.))
