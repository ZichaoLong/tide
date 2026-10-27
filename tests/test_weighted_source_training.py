"""Architecture §2.2: all shared users contribute to AdamW and exact resume."""
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.reference import run as reference
from test_weighted_source_execution import source_setup, runner


@pytest.mark.parametrize("mode,family", [("region", "hb"), ("node", "pdg"), ("source", "hb"),
                                         ("adapted", "pdg"), ("source", "pdg"), ("adapted", "hb")])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_source_adamw_and_pending_checkpoint(dtype, mode, family, native, tmp_path):
    a, xa, stop = source_setup(dtype, family, mode)
    b, xb, _ = source_setup(dtype, family, mode)
    execute = runner(b, native)
    def optimizer(model):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
    oa, ob = optimizer(a.model), optimizer(b.model)
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        expected = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        actual = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        equivalent(expected, actual)
        objective(expected).backward(); objective(actual).backward()
        for x, y in zip(a.model.parameters(), b.model.parameters()):
            equivalent(x.grad, y.grad)
        oa.step(); ob.step()
        equivalent(a.model.state_dict(), b.model.state_dict())
        equivalent(oa.state_dict(), ob.state_dict())
    first = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < 2], 2, sealed_until=2)
    assert first.continuation.pending
    path = tmp_path/"sources.pt"
    save(path, b.graph, b.model, first.continuation, ob)
    c, _, _ = source_setup(dtype, family, mode); oc = optimizer(c.model)
    restored = load(path, c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored)
    equivalent(ob.state_dict(), oc.state_dict())
    tail = [x for x in xb if x.time >= 2]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    expected = execute(first.continuation.detach(), tail, stop, sealed_until=stop)
    actual = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(expected, actual)
    objective(expected).backward(); objective(actual).backward()
    ob.step(); oc.step()
    equivalent(b.model.state_dict(), c.model.state_dict())
    equivalent(ob.state_dict(), oc.state_dict())
    # Same shapes do not license loading a different binding or alias graph.
    other, _, _ = source_setup(dtype, family, "node" if mode != "node" else "region")
    with pytest.raises(ValueError, match="graph mismatch"):
        load(path, other.graph, other.model)
    c, _, _ = source_setup(dtype, family, mode)
    if mode in {"region", "adapted", "source"}:
        seen, t = set(), None
        for node in c.model.nodes:
            for transform in node.aggregate_program.transforms.values():
                if id(transform.a) in seen:
                    t = transform
                seen.add(id(transform.a))
        assert t is not None
        t.a = torch.nn.Parameter(t.a.detach().clone())
        with pytest.raises(ValueError, match="sharing"):
            load(path, c.graph, c.model)
