import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run


def fixture(dtype, kind):
    g = Graph(tuple(Node(0 if i < 2 else 1, memory=kind, full="swiglu") for i in range(3)),
              (Edge(0, 2, 2), Edge(1, 2, 3)), (Region(1), Region(1)), (0, 1), (2,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 3)
    x = torch.sin(torch.arange(120, dtype=dtype) * 0.17).reshape(3, 2, 5, 4).requires_grad_()
    variables = dict(m.named_parameters()) | {"input": x}
    for b in range(3):
        for v in range(3):
            state = m.nodes[v].initial()
            state.value = torch.full_like(state.value, 0.03 * (b+1), requires_grad=True)
            state.slots = {name: torch.full_like(t, 0.12 * (b+1), requires_grad=True) for name, t in state.slots.items()}
            q.states[b, v] = state
            variables[f"initial.{b}.{v}.value"] = state.value
            variables.update({f"initial.{b}.{v}.{k}": t for k, t in state.slots.items()})
    xs = [External(b, p, i, t, x[b, p, i]) for b in range(3) for p in range(2)
          for i, t in enumerate((0, 1, 3, 6, 7) if b < 2 else (0, 5, 7))]
    return g, m, q, xs, variables


def gradients(result, variables):
    return dict(zip(variables, torch.autograd.grad(objective(result), list(variables.values()), allow_unused=True)))


@pytest.mark.parametrize("kind", ["ema", "ssm"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "native-unpacked", "native-packed", "native-step"])
def test_joint_memory_scan_trace_vjp(dtype, kind, mode, implementation):
    g, m, q, xs, variables = fixture(dtype, kind)
    expected = run(g, m, q, xs, 12, sealed_until=12, mode=mode)
    grad = gradients(expected, variables)
    g, m, q, xs, variables = fixture(dtype, kind)
    actual = frontier(g, m, q, xs, 12, sealed_until=12, mode=mode) if implementation == "python" else Native(
        g, m, algorithm="frontier", workers=3, packed=implementation == "native-packed",
        prefill=implementation != "native-step", mode=mode).run(q, xs, 12, sealed_until=12)
    equivalent(expected, actual); equivalent(grad, gradients(actual, variables))


@pytest.mark.parametrize("kind", ["ema", "ssm"])
@pytest.mark.parametrize("implementation", ["python", "native-packed", "native-unpacked"])
def test_joint_memory_scan_shapes_and_compact_state(dtype, kind, implementation):
    g = Graph((Node(0, memory=kind),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=4, dtype=dtype); q = Continuation(g.identity, 4)
    lengths = (7, 4, 7)
    xs = [External(b, 0, t, 2*t, torch.full((4,), (b+1) * (t+1) / 10, dtype=dtype))
          for b, length in enumerate(lengths) for t in range(length)]
    with torch.inference_mode():
        expected = run(g, m, q, xs, 15, sealed_until=15)
        actual = frontier(g, m, q, xs, 15, sealed_until=15) if implementation == "python" else Native(
            g, m, algorithm="frontier", packed=implementation == "native-packed").run(q, xs, 15, sealed_until=15)
    equivalent(expected, actual)
    assert actual.stats["state_blocks"] == 3
    assert actual.stats["state_sequence_calls"] == (3 if implementation == "native-unpacked" else 2)
    assert (3, 0) not in actual.continuation.states
    if implementation.startswith("native"):
        assert actual.stats["max_state_batch"] == (2 if implementation == "native-packed" else 1)
        assert actual.stats["max_state_sequence"] == 7
    for state in actual.continuation.states.values():
        for t in (state.value, *state.slots.values()):
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
