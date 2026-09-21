from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent
from tidegraph.token_window import token_inputs
from isolated_cases import vjp
from read_cases import execute
from pronounce_cases import KINDS, fixture, direct, logits


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_pronounce_direct_token_anchor_and_isolated_roots(dtype, kind, implementation):
    g, m, q, body, leaves = fixture(dtype, kind)
    expected, states = direct(m, body, 3, kind)
    result = execute(implementation, g, m, q, token_inputs(body, q, 3, 4, body_cut=12), stop=4, mode="hard")
    actual = logits(m, result.outputs)
    equivalent(expected, actual); equivalent(states, result.continuation.states)
    direction = m.nodes[0].bias.new_tensor([.3, -.2, .5, .7, -.9])
    for a, b in ((expected[0, 0]@direction, actual[0, 0]@direction),
                 (sum(x@direction for x in expected.values()), sum(x@direction for x in actual.values()))):
        equivalent(vjp(a, leaves), vjp(b, leaves))
    for owner, state in states.items():
        other = result.continuation.states[owner]
        equivalent(vjp(state.value, leaves), vjp(other.value, leaves))
        for name, value in state.slots.items(): equivalent(vjp(value, leaves), vjp(other.slots[name], leaves))
    assert all(t in range(4) and b < 2 for t, b in actual)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("implementation", ["python-frontier", "native-frontier"])
def test_pronounce_token_cuts_detach_and_checkpoint(dtype, kind, implementation, tmp_path):
    g, m, q, body, leaves = fixture(dtype, kind)
    whole = execute(implementation, g, m, q, token_inputs(body, q, 3, 4, body_cut=12), stop=4, mode="hard")
    first = execute(implementation, g, m, q, token_inputs([o for o in body if o[1] < 6], q, 3, 2, body_cut=6),
                    stop=2, mode="hard")
    q = first.continuation
    xs = token_inputs([o for o in body if o[1] >= 6], q, 3, 4, body_cut=12)
    second = execute(implementation, g, m, q, xs, stop=4, mode="hard")
    equivalent(whole, replace(second, outputs=first.outputs+second.outputs, trace=first.trace+second.trace))
    tail = execute(implementation, g, m, q.detach(), xs, stop=4, mode="hard")
    grads = vjp(next(iter(logits(m, tail.outputs).values())), leaves)
    assert grads["input.0.0.0"] is None
    path = tmp_path/"pronounce.pt"; save(path, g, m, q)
    _, restored, _, _, _ = fixture(dtype, kind); loaded = load(path, g, restored)
    equivalent(loaded, q); equivalent(m.state_dict(), restored.state_dict())
    resumed = execute(implementation, g, restored, loaded, xs, stop=4, mode="hard")
    equivalent(logits(restored, resumed.outputs), logits(m, second.outputs))


def train(dtype, kind, implementation, path):
    g, m, q, body, _ = fixture(dtype, kind)
    optimizer = torch.optim.SGD(m.parameters(), lr=.002, momentum=.8, weight_decay=.01)
    records = []
    for stop in (2, 4):
        optimizer.zero_grad(set_to_none=True)
        part = [o for o in body if q.cut*3 <= o[1] < stop*3]
        result = execute(implementation, g, m, q, token_inputs(part, q, 3, stop, body_cut=stop*3),
                         stop=stop, mode="hard")
        direction = m.nodes[0].bias.new_tensor([.3, -.2, .5, .7, -.9])
        sum(x@direction for x in logits(m, result.outputs).values()).backward()
        gradients = {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()}
        optimizer.step(); q = result.continuation.detach()
        records.append((q, gradients, {k: v.clone() for k, v in m.state_dict().items()}))
        if stop == 2:
            save(path, g, m, q, optimizer)
            _, restored, _, _, _ = fixture(dtype, kind)
            resumed = torch.optim.SGD(restored.parameters(), lr=.002, momentum=.8, weight_decay=.01)
            loaded = load(path, g, restored, resumed)
            equivalent(resumed.state_dict(), optimizer.state_dict())
            q, m, optimizer = loaded, restored, resumed
    return records, optimizer.state_dict()


@pytest.mark.parametrize("kind", ["add", "all-softmax"])
def test_pronounce_optimizer_resume_with_rectangular_head(dtype, kind, tmp_path):
    expected = train(dtype, kind, "reference", tmp_path/"reference.pt")
    actual = train(dtype, kind, "native-frontier", tmp_path/"native.pt")
    equivalent(expected, actual)
