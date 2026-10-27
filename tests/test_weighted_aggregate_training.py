"""Architecture §2: batched Aggregate shared-owner training and exact resume."""
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.reference import run as reference
from test_weighted_execution import setup


@pytest.mark.parametrize("family", ["hb", "pdg"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_aggregate_adamw_and_pending_checkpoint(dtype, family, native, tmp_path):
    a, _, xa, stop = setup(dtype, family); b, _, xb, _ = setup(dtype, family)
    def runner(binding):
        if native:
            return Native(binding.graph, binding.model, algorithm="greedy", packed=True, aggregate_autograd="batched").run
        return lambda *args, **kw: greedy(binding.graph, binding.model, *args, aggregate_autograd="batched", **kw)
    execute = runner(b)
    oa = torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad], lr=.002)
    ob = torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad], lr=.002)
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        expected = reference(a.graph, a.model, Continuation(a.graph.identity, 1), xa, stop, sealed_until=stop)
        actual = execute(Continuation(b.graph.identity, 1), xb, stop, sealed_until=stop)
        equivalent(expected, actual)
        assert actual.stats.get("semantic_aggregate_replays", 0) == 0
        assert actual.stats["batched_aggregate_events"] > 0
        objective(expected).backward(); objective(actual).backward()
        for x, y in zip(a.model.parameters(), b.model.parameters()):
            equivalent(x.grad, y.grad)
        oa.step(); ob.step()
        equivalent(a.model.state_dict(), b.model.state_dict())
        equivalent(oa.state_dict(), ob.state_dict())
    cut = 2
    first = execute(Continuation(b.graph.identity, 1), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    assert first.continuation.pending
    save(tmp_path/"aggregate.pt", b.graph, b.model, first.continuation, ob)
    c, _, _, _ = setup(dtype, family)
    oc = torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad], lr=.002)
    restored = load(tmp_path/"aggregate.pt", c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored)
    equivalent(ob.state_dict(), oc.state_dict())
    tail = [x for x in xb if x.time >= cut]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    expected = execute(first.continuation.detach(), tail, stop, sealed_until=stop)
    actual = runner(c)(restored, tail, stop, sealed_until=stop)
    equivalent(expected, actual)
    objective(expected).backward(); objective(actual).backward()
    ob.step(); oc.step()
    equivalent(b.model.state_dict(), c.model.state_dict())
    equivalent(ob.state_dict(), oc.state_dict())
