from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from add_cases import PROFILE, fixture, physical
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-packed", "native-frontier"])
def test_add_cuts_idle_prefix_suffix_and_detach(dtype, implementation):
    g, m, q, xs, leaves = fixture(dtype)
    whole = execute(implementation, g, m, q, xs, stop=12)
    outputs, trace, messages = [], [], []
    for stop in (2, 4, 8, 12):
        part = execute(implementation, g, m, q, [x for x in xs if q.cut <= x.time < stop], stop=stop)
        q = part.continuation; outputs += part.outputs; trace += part.trace; messages += part.messages
    joined = replace(part, outputs=outputs, trace=trace, messages=messages)
    equivalent(whole, joined)
    equivalent(vjp(objective(whole), leaves), vjp(objective(joined), leaves))
    equivalent(physical(whole, m), physical(joined, m))
    start = fixture(dtype)[2]
    prefix = execute(implementation, g, m, start, [x for x in xs if x.time < 4], stop=4)
    suffix = execute(implementation, g, m, prefix.continuation.detach(), [x for x in xs if x.time >= 4], stop=12)
    grads = vjp(objective(suffix), leaves)
    assert grads["input.0.0.0"] is None and grads["input.1.0.0"] is None


def test_add_owned_cursor_idle_advance_preserves_sparse_representation(dtype):
    g, m, q, xs, leaves = fixture(dtype)
    engine = Native(g, m, workers=3, packed=True, mode="hst")
    expected = engine.run(q, xs, 12, sealed_until=12)
    cursor = engine.cursor(q)
    cursor.advance(xs, 12, sealed_until=12)
    snapshot = cursor.snapshot(); equivalent(expected.continuation, snapshot)
    equivalent(vjp(objective(expected), leaves), vjp(objective(replace(expected, continuation=snapshot)), leaves))
    stats = cursor.advance([], 15, sealed_until=15).stats
    later = cursor.snapshot()
    equivalent(snapshot.states, later.states); equivalent(snapshot.history, later.history)
    assert stats.get("candidate_events", 0) == 0
    assert later.states[2, 0].last_time == -1
    assert not torch.equal(physical(expected, m)[2, 0], physical(replace(expected, continuation=later), m)[2, 0])


def test_add_sharing_adamw_and_checkpoint_clock(dtype, tmp_path):
    g = Graph((Node(0, memory=PROFILE), Node(1, memory=PROFILE)), (), (Region(1),)*2, (0, 1), (0, 1))
    m = Model(g, width=2, dtype=dtype); m.nodes[1] = m.nodes[0]
    optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
    xs = [External(0, p, i, t, torch.full((2,), .2+p/10+i/20, dtype=dtype))
          for p in range(2) for i, t in enumerate((2, 7))]
    q = Continuation(g.identity, 1)
    expected = run(g, m, q, xs, 10, sealed_until=10)
    actual = Native(g, m, algorithm="frontier", packed=True, workers=2).run(q, xs, 10, sealed_until=10)
    equivalent(expected, actual)
    leaves = dict(m.named_parameters())
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    objective(actual).backward(); optimizer.step()
    path = tmp_path / "add.pt"; save(path, g, m, actual.continuation, optimizer)
    restored = Model(g, width=2, dtype=dtype); restored.nodes[1] = restored.nodes[0]
    ro = torch.optim.AdamW(restored.parameters(), lr=.001)
    loaded = load(path, g, restored, ro)
    equivalent(loaded, actual.continuation.detach()); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), ro.state_dict())
    assert (loaded.states[0, 0].last_time, loaded.states[0, 0].observations) == (7, 2)
    equivalent(physical(actual, m), physical(replace(actual, continuation=loaded), restored, native=True))
    record = torch.load(path, weights_only=True)
    bad_state = list(record["states"][0, 0]); bad_state[1] = 10; record["states"][0, 0] = tuple(bad_state)
    bad = tmp_path / "bad.pt"; torch.save(record, bad)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="clock"):
        load(bad, g, restored, ro)
    equivalent(before, restored.state_dict())
