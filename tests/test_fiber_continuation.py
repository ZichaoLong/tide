from dataclasses import replace
import pytest
import torch
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from fiber_cases import fixture, physical
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-packed", "native-frontier"])
def test_fiber_cuts_idle_prefix_suffix_and_detach(dtype, implementation):
    g, m, q, xs, leaves = fixture(dtype)
    whole = execute(implementation, g, m, q, xs, stop=12)
    outputs, trace, messages = [], [], []
    for stop in (0, 2, 4, 8, 12):
        part = execute(implementation, g, m, q, [x for x in xs if q.cut <= x.time < stop], stop=stop)
        q = part.continuation; outputs += part.outputs; trace += part.trace; messages += part.messages
    joined = replace(part, outputs=outputs, trace=trace, messages=messages)
    equivalent(whole, joined); equivalent(vjp(objective(whole), leaves), vjp(objective(joined), leaves))
    equivalent(physical(whole, m), physical(joined, m))
    start = fixture(dtype)[2]
    prefix = execute(implementation, g, m, start, [x for x in xs if x.time < 4], stop=4)
    suffix = execute(implementation, g, m, prefix.continuation.detach(), [x for x in xs if x.time >= 4], stop=12)
    grads = vjp(objective(suffix), leaves)
    assert grads["input.0.0.0"] is None and grads["input.1.0.0"] is None


def test_fiber_owned_cursor_keeps_idle_cache_lazy(dtype):
    g, m, q, xs, leaves = fixture(dtype)
    engine = Native(g, m, workers=3, packed=True, mode="hst")
    expected = engine.run(q, xs, 12, sealed_until=12)
    cursor = engine.cursor(q); cursor.advance(xs, 12, sealed_until=12)
    snapshot = cursor.snapshot(); equivalent(expected.continuation, snapshot)
    equivalent(vjp(objective(expected), leaves), vjp(objective(replace(expected, continuation=snapshot)), leaves))
    stats = cursor.advance([], 15, sealed_until=15).stats
    later = cursor.snapshot(); equivalent(snapshot.states, later.states)
    assert stats.get("candidate_events", 0) == 0 and later.states[2, 0].last_time == -1
    assert not torch.equal(physical(expected, m)[2, 0], physical(replace(expected, continuation=later), m)[2, 0])


def train(dtype, implementation):
    g, m, q, xs, _ = fixture(dtype)
    m.nodes[1] = m.nodes[0]
    optimizer = torch.optim.AdamW(m.parameters(), lr=.002)
    records = []
    for stop in (4, 12):
        optimizer.zero_grad(set_to_none=True)
        result = execute(implementation, g, m, q, [x for x in xs if q.cut <= x.time < stop], stop=stop)
        objective(result).backward(); optimizer.step()
        q = result.continuation.detach()
        records.append((q, {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()},
                        {k: t.clone() for k, t in m.state_dict().items()}))
    return records, g, m, q, optimizer


def test_fiber_shared_training_checkpoint_and_rejection(dtype, tmp_path):
    expected, _, _, _, _ = train(dtype, "reference")
    actual, g, m, q, optimizer = train(dtype, "native-frontier")
    equivalent(expected, actual)
    path = tmp_path/"fiber.pt"; save(path, g, m, q, optimizer)
    restored = fixture(dtype)[1]; restored.nodes[1] = restored.nodes[0]
    ro = torch.optim.AdamW(restored.parameters(), lr=.002)
    loaded = load(path, g, restored, ro)
    equivalent(loaded, q); equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())
    assert restored.nodes[0] is restored.nodes[1]
    record = torch.load(path, weights_only=True)
    bad_state = list(record["states"][0, 0]); bad_state[1] = 12; record["states"][0, 0] = tuple(bad_state)
    bad = tmp_path/"bad.pt"; torch.save(record, bad)
    before = {k: t.clone() for k, t in restored.state_dict().items()}
    with pytest.raises(ValueError, match="clock"):
        load(bad, g, restored, ro)
    equivalent(before, restored.state_dict())
