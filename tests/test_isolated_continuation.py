import pytest
import torch
from tidegraph.blocks import canonicalize
from tidegraph.compare import equivalent
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.records import Result
from tidegraph.reference import run
from isolated_cases import fixture, vjp


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta"])
@pytest.mark.parametrize("implementation", ["python", "native", "cursor"])
def test_isolated_roots_across_cuts_and_pending_delivery(dtype, kind, implementation):
    g, m, q, xs, variables = fixture(dtype, kind)
    expected = run(g, m, q, xs, 9, sealed_until=9, mode="hst")
    g, m, q, xs, actual_variables = fixture(dtype, kind)
    engine = Native(g, m, workers=3, packed=True, mode="hst") if implementation != "python" else None
    cursor = engine.cursor(q) if implementation == "cursor" else None
    events, outputs, messages = [], [], []
    start = 0
    for stop in (3, 4, 9):
        inputs = [a for a in xs if start <= a.time < stop]
        if cursor:
            part = cursor.advance(inputs, stop, sealed_until=stop)
            q = cursor.snapshot()
        elif engine:
            part = engine.run(q, inputs, stop, sealed_until=stop); q = part.continuation
        else:
            part = frontier(g, m, q, inputs, stop, sealed_until=stop, mode="hst"); q = part.continuation
        if stop == 3:
            assert q.pending
        events += part.trace; outputs += part.outputs; messages += part.messages
        start = stop
    actual = canonicalize(g, Result(q, events, outputs, messages, {}))
    equivalent(expected, actual)
    for a, b in ((expected.outputs[0][-1], actual.outputs[0][-1]),
                 (expected.continuation.states[0, 2].value, actual.continuation.states[0, 2].value)):
        for zero in (False, True):
            expected_grad = vjp(a, variables, zero)
            equivalent(expected_grad, vjp(b, actual_variables, zero))
            assert expected_grad["upstream.1"] is None


@pytest.mark.parametrize("implementation", ["python", "native-stream", "native-frontier"])
@pytest.mark.parametrize("zero", [False, True])
def test_isolated_loss_optimizer_skips_disconnected_parameters(dtype, implementation, zero):
    def update(native):
        g, m, q, xs, variables = fixture(dtype, "ssm")
        parameters = dict(m.named_parameters()) | {k: v for k, v in variables.items() if k.startswith("upstream.")}
        optimizer = torch.optim.AdamW(parameters.values(), lr=0.01, weight_decay=0.1)
        if not native:
            result = run(g, m, q, xs, 4, sealed_until=4, mode="hst")
        elif implementation == "python":
            result = frontier(g, m, q, xs, 4, sealed_until=4, mode="hst")
        else:
            result = Native(g, m, workers=3, packed=True, mode="hst",
                            algorithm="streaming" if implementation == "native-stream" else "frontier").run(
                                q, xs, 4, sealed_until=4)
        before = parameters["upstream.1"].detach().clone()
        loss = result.outputs[0][-1].square().sum() * (0.0 if zero else 0.7)
        loss.backward()
        assert parameters["upstream.1"].grad is None
        assert parameters["upstream.0"].grad is not None
        optimizer.step()
        assert torch.equal(before, parameters["upstream.1"])
        return {k: v.detach().clone() for k, v in parameters.items()}, optimizer.state_dict()
    equivalent(update(False), update(True))
