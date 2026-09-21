from dataclasses import replace
import pytest
import torch
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.ops import Model
from fiber_cases import fixture
from fiber_pool_cases import POOLS, profile
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("implementation", ["python-frontier", "native-frontier"])
def test_pooling_complete_cuts_and_explicit_detach(dtype, kind, implementation):
    g, m, q, xs, leaves = fixture(dtype, profile=profile(kind))
    whole = execute(implementation, g, m, q, xs, stop=12)
    outputs, trace, messages = [], [], []
    for stop in (2, 4, 8, 12):
        part = execute(implementation, g, m, q, [x for x in xs if q.cut <= x.time < stop], stop=stop)
        q = part.continuation; outputs += part.outputs; trace += part.trace; messages += part.messages
    joined = replace(part, outputs=outputs, trace=trace, messages=messages)
    equivalent(whole, joined); equivalent(vjp(objective(whole), leaves), vjp(objective(joined), leaves))
    start = fixture(dtype, profile=profile(kind))[2]
    prefix = execute(implementation, g, m, start, [x for x in xs if x.time < 4], stop=4)
    suffix = execute(implementation, g, m, prefix.continuation.detach(), [x for x in xs if x.time >= 4], stop=12)
    grads = vjp(objective(suffix), leaves)
    assert grads["input.0.0.0"] is None and grads["input.1.0.0"] is None


def train(dtype, kind, implementation):
    g, m, q, xs, _ = fixture(dtype, profile=profile(kind))
    m.nodes[1] = m.nodes[0]  # Same incoming slot domain; vector/attention/FFN parameters all share.
    optimizer = torch.optim.AdamW(m.parameters(), lr=.002); records = []
    for stop in (4, 12):
        optimizer.zero_grad(set_to_none=True)
        result = execute(implementation, g, m, q, [x for x in xs if q.cut <= x.time < stop], stop=stop)
        objective(result).backward(); optimizer.step(); q = result.continuation.detach()
        records.append((q, {k: None if p.grad is None else p.grad.clone() for k, p in m.named_parameters()},
                        {k: t.clone() for k, t in m.state_dict().items()}))
    return records, g, m, q, optimizer


@pytest.mark.parametrize("kind", POOLS)
def test_pooling_shared_optimizer_and_checkpoint(dtype, kind, tmp_path):
    expected, _, _, _, _ = train(dtype, kind, "reference")
    actual, g, m, q, optimizer = train(dtype, kind, "native-frontier"); equivalent(expected, actual)
    path = tmp_path/"pool.pt"; save(path, g, m, q, optimizer)
    restored = Model(g, width=4, dtype=dtype); restored.nodes[1] = restored.nodes[0]
    ro = torch.optim.AdamW(restored.parameters(), lr=.002)
    loaded = load(path, g, restored, ro)
    equivalent(loaded, q); equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())
    assert restored.nodes[0] is restored.nodes[1]
    # A different profile has different graph identity even when initial pooling
    # happens to produce the same numerical mean.
    changed = replace(g, nodes=tuple(replace(n, memory=profile("linear" if kind != "linear" else "mean")) for n in g.nodes))
    with pytest.raises(ValueError, match="identity|graph"):
        load(path, changed, Model(changed, width=4, dtype=dtype))
