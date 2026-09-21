import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from emit_cases import fixture
from isolated_cases import vjp


@pytest.mark.parametrize("memory", ["ema", "ssm", "attention"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "python-step", "native-stream", "native-serial", "native-frontier", "native-step"])
def test_per_slot_phase_emissions_trace_and_isolated_vjps(dtype, memory, mode, implementation):
    g, m, q, xs, variables = fixture(dtype, memory)
    expected = run(g, m, q, xs, 5, sealed_until=5, mode=mode)
    g, m, q, xs, actual_variables = fixture(dtype, memory)
    if implementation.startswith("python"):
        actual = frontier(g, m, q, xs, 5, sealed_until=5, mode=mode, prefill=implementation == "python")
    else:
        actual = Native(g, m, algorithm="streaming" if implementation in {"native-stream", "native-serial"} else "frontier",
                        workers=1 if implementation == "native-serial" else 3, packed=implementation != "native-serial",
                        prefill=implementation != "native-step", mode=mode).run(q, xs, 5, sealed_until=5)
    equivalent(expected, actual)
    pairs = [(objective(expected), objective(actual)), (expected.outputs[0][-1], actual.outputs[0][-1]),
             (expected.trace[0]["emitted"][0], actual.trace[0]["emitted"][0]),
             (expected.continuation.pending[0].value, actual.continuation.pending[0].value)]
    for a, b in pairs:
        for zero in (False, True):
            grad = vjp(a, variables, zero)
            equivalent(grad, vjp(b, actual_variables, zero))
            assert grad["nodes.0.extra.emit_w_1"] is None
            assert grad["edge_scale.0"] is None
    assert not any(a.source == 0 for a in actual.messages)
    assert any(a.source == 1 for a in actual.messages)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_zero_message_is_present_and_absent_slot_does_not_touch_destination(dtype, implementation):
    g = Graph((Node(0, emission="slot_affine", emit_phases=(-1, -2)), Node(1), Node(2)),
              (Edge(0, 1, 1), Edge(0, 2, 1)), (Region(1),) * 3, (0,), (1, 2))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        m.nodes[0].extra["emit_w_0"].zero_(); m.nodes[0].extra["emit_b_0"].zero_()
    q = Continuation(g.identity, 1); xs = [External(0, 0, 0, 0, torch.ones(2, dtype=dtype))]
    result = frontier(g, m, q, xs, 2, sealed_until=2) if implementation == "python" else Native(
        g, m, algorithm="frontier", packed=True).run(q, xs, 2, sealed_until=2)
    assert [(e["node"], e["time"]) for e in result.trace] == [(0, 0), (1, 1)]
    assert len(result.messages) == 1 and torch.count_nonzero(result.messages[0].value) == 0
    assert (0, 1) in result.continuation.states and (0, 2) not in result.continuation.states
    assert [p for _, _, p, _ in result.outputs] == [0]


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_full_reads_preclear_comparison_for_distinct_slot_values(dtype, implementation):
    g = Graph((Node(0, clear=True, emission="slot_affine"),), (), (Region(1),), (0,), (0, 0))
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        m.nodes[0].weight.copy_(torch.eye(2, dtype=dtype)); m.nodes[0].bias.zero_()
        for group in (m.input_scale, m.output_scale):
            for p in group:
                p.fill_(1)
        for slot in range(2):
            m.nodes[0].extra[f"emit_w_{slot}"].copy_(torch.eye(2, dtype=dtype) * (slot+1))
            m.nodes[0].extra[f"emit_b_{slot}"].fill_(slot / 10)
    x = torch.tensor([0.2, -0.3], dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 1); xs = [External(0, 0, 0, 0, x)]
    result = frontier(g, m, q, xs, 1, sealed_until=1) if implementation == "python" else Native(
        g, m, packed=True).run(q, xs, 1, sealed_until=1)
    assert torch.count_nonzero(result.continuation.states[0, 0].value) == 0
    for _, _, port, value in result.outputs:
        expected = (x + x.tanh()) * (port+1) + port / 10
        equivalent(value, expected)
        a, = torch.autograd.grad(value.sum(), x, retain_graph=True)
        b, = torch.autograd.grad(expected.sum(), x, retain_graph=True)
        equivalent(a, b)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_never_emitted_slot_is_skipped_by_optimizer(dtype, implementation):
    def update(anchor):
        g, m, q, xs, _ = fixture(dtype, "ssm")
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.1)
        if anchor:
            result = run(g, m, q, xs, 5, sealed_until=5, mode="hst")
        elif implementation == "python":
            result = frontier(g, m, q, xs, 5, sealed_until=5, mode="hst")
        else:
            result = Native(g, m, packed=True, workers=3, mode="hst").run(q, xs, 5, sealed_until=5)
        never = m.nodes[0].extra["emit_w_1"]; before = never.detach().clone()
        objective(result).backward(); assert never.grad is None
        optimizer.step(); assert never not in optimizer.state
        assert torch.equal(before, never)
        return m.state_dict(), optimizer.state_dict()
    equivalent(update(True), update(False))
