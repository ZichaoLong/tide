import math
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from cases import gradients


def fixture(dtype, memory, full, clear, cyclic=False):
    nodes = tuple(Node(i, clear=clear and i == 1, memory=memory, full=full) for i in range(3))
    edges = (Edge(0, 1, 1), Edge(1, 2, 1)) + ((Edge(2, 0, 1),) if cyclic else ())
    g = Graph(nodes, edges, (Region(1),) * 3, (0,), (2,))
    m = Model(g, dtype=dtype)
    x = (torch.arange(24, dtype=dtype).reshape(8, 3) / 40 - 0.2).requires_grad_()
    xs = [External(b, 0, i, time, x[b * 4 + i]) for b in range(2) for i, time in enumerate((0, 1, 4, 6))]
    initial = torch.full((2, 3, 2, 3), 0.17, dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 2)
    for b in range(2):
        for v in range(3):
            q.states[b, v] = State(initial[b, v, 0], slots={"memory": initial[b, v, 1]} if memory == "ssm" else {})
    return g, m, q, xs, x, initial


@pytest.mark.parametrize("memory,full", [("ema", "swiglu"), ("ssm", "tanh"), ("ssm", "swiglu")])
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-streaming", "native-frontier", "native-chain"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_state_program_cross_executor_vjp(dtype, memory, full, clear, implementation, mode):
    g, m, q, xs, x, initial = fixture(dtype, memory, full, clear)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode=mode)
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = fixture(dtype, memory, full, clear)
    actual = frontier(g, m, q, xs, 10, sealed_until=10, mode=mode) if implementation == "python-frontier" else Native(
        g, m, workers=3, packed=True, algorithm=implementation.split("-")[1], mode=mode).run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))


@pytest.mark.parametrize("workers,packed", [(1, False), (1, True), (3, True)])
def test_ssm_cyclic_streaming(dtype, workers, packed):
    g, m, q, xs, x, initial = fixture(dtype, "ssm", "swiglu", True, cyclic=True)
    expected = run(g, m, q, xs, 10, sealed_until=10, mode="hst")
    expected_grad = gradients(objective(expected), m, x, initial)
    g, m, q, xs, x, initial = fixture(dtype, "ssm", "swiglu", True, cyclic=True)
    actual = Native(g, m, workers=workers, packed=packed, mode="hst").run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))


def test_ssm_hand_computed_state_and_vjp(dtype):
    g = Graph((Node(0, memory="ssm"),), (), (Region(1),), (0,), (0,))
    w = Model(g, width=1, dtype=dtype).nodes[0]
    with torch.no_grad():
        w.extra["ssm_dt"].zero_(); w.extra["ssm_a"].fill_(math.log(math.expm1(1)))
        w.extra["ssm_b"].fill_(1 / math.log(2)); w.extra["ssm_c"].fill_(1); w.extra["ssm_skip"].zero_()
    h = torch.tensor([[1.0], [2.0]], dtype=dtype, requires_grad=True)
    memory = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    old = State(torch.zeros_like(memory), slots={"memory": memory})
    s1, _ = w.prepare(old, h[0], 2)
    s2, _ = w.prepare(s1, h[1], 9)
    equivalent(s1.slots["memory"], memory.new_tensor([1.15]))
    equivalent(s2.slots["memory"], memory.new_tensor([2.575]))
    equivalent(s2.value, memory.new_tensor([5.15]))
    dh, dm = torch.autograd.grad(s2.value.sum(), (h, memory))
    equivalent(dh, h.new_tensor([[1.0], [4.575]])); equivalent(dm, memory.new_tensor([0.5]))
    states, _ = w.prepare_block(old, h, [2, 9])
    equivalent(states, [s1, s2])


def test_swiglu_hand_computed_vjp(dtype):
    g = Graph((Node(0, full="swiglu"),), (), (Region(1),), (0,), (0,))
    w = Model(g, width=2, dtype=dtype).nodes[0]
    with torch.no_grad():
        for p in w.extra.values():
            p.zero_()
        w.extra["ffn_gate"][:, :2].copy_(torch.eye(2, dtype=dtype))
        w.extra["ffn_up"][:, :2].copy_(torch.eye(2, dtype=dtype))
        w.extra["ffn_down"][:2].copy_(torch.eye(2, dtype=dtype))
    h = torch.tensor([0.1, -0.2], dtype=dtype, requires_grad=True)
    s = torch.tensor([0.3, -0.7], dtype=dtype, requires_grad=True)
    value = w.full(s, h, h.new_tensor(0.2), "hard", 1)
    sig = s.sigmoid()
    equivalent(value, h + s.square() * sig)
    dh, ds = torch.autograd.grad(value.sum(), (h, s))
    equivalent(dh, torch.ones_like(h)); equivalent(ds, 2 * s * sig + s.square() * sig * (1 - sig))


def test_slot_clear_checkpoint_and_detach(dtype, tmp_path):
    g, m, q, xs, _, _ = fixture(dtype, "ssm", "swiglu", True, cyclic=True)
    part = run(g, m, q, [a for a in xs if a.time < 5], 5, sealed_until=5)
    for event in part.trace:
        if event["node"] == 1:
            assert torch.count_nonzero(event["next_slots"]["memory"]) == 0
            assert event["comparison_slots"]["memory"].abs().sum() > 0
    path = tmp_path / "slots.pt"
    save(path, g, m, part.continuation)
    restored = load(path, g, m)
    equivalent(part.continuation, restored)
    detached = part.continuation.detach()
    assert all(not s.slots["memory"].requires_grad for s in detached.states.values())
    follow = [a for a in xs if a.time >= 5]
    expected = run(g, m, part.continuation, follow, 10, sealed_until=10)
    actual = Native(g, m, workers=3, packed=True).run(restored, follow, 10, sealed_until=10)
    equivalent(expected, actual)
