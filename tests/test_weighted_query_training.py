"""Architecture §3/4: actual batch VJPs, tied owners, AdamW and exact cuts.

Independent scalar schedules anchor whole records. Local formula and directional
checks live beside the query VJP tests; no reference trajectory feeds a candidate.
"""
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.native import Native
from tidegraph.reference import run as reference
from test_weighted_scoring_execution import configured
from test_weighted_execution import variables
from isolated_cases import vjp


def fixture(dtype, family):
    case = 3 if family == "hb" else 0
    b, xs, stop = configured(dtype, case, family, tie=case == 3)
    xs = [External(i, x.port, x.position, x.time,
                   (x.value.detach()+x.value.new_tensor([0., .04*i, -.02*i])).requires_grad_())
          for i in range(2) for x in xs]
    return b, xs, stop


def runner(b, native):
    policy = dict(packed=True, full_autograd="batched", aggregate_autograd="batched")
    if native:
        return Native(b.graph, b.model, algorithm="greedy", **policy).run
    return lambda *a, **kw: greedy(b.graph, b.model, *a, **kw, **policy)


def stats(result):
    for kind in ("state", "read", "full", "aggregate"):
        assert result.stats.get(f"semantic_{kind}_replays", 0) == 0
        assert result.stats[f"batched_{kind}_events"] > 0


@pytest.mark.parametrize("family", ["hb", "pdg"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_adamw_and_pending_checkpoint(dtype, family, native, tmp_path):
    a, xa, stop = fixture(dtype, family); b, xb, _ = fixture(dtype, family)
    execute = runner(b, native)
    oa = torch.optim.AdamW([p for p in a.model.parameters() if p.requires_grad], lr=.001)
    ob = torch.optim.AdamW([p for p in b.model.parameters() if p.requires_grad], lr=.001)
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        expected = reference(a.graph, a.model, Continuation(a.graph.identity, 2), xa, stop, sealed_until=stop)
        actual = execute(Continuation(b.graph.identity, 2), xb, stop, sealed_until=stop)
        equivalent(expected, actual); stats(actual)
        objective(expected).backward(); objective(actual).backward()
        for x, y in zip(a.model.parameters(), b.model.parameters()):
            equivalent(x.grad, y.grad)
        oa.step(); ob.step()
        equivalent(a.model.state_dict(), b.model.state_dict())
        equivalent(oa.state_dict(), ob.state_dict())
    cut = 2
    first = execute(Continuation(b.graph.identity, 2), [x for x in xb if x.time < cut], cut, sealed_until=cut)
    assert first.continuation.pending
    save(tmp_path/"query.pt", b.graph, b.model, first.continuation, ob)
    c, _, _ = fixture(dtype, family)
    oc = torch.optim.AdamW([p for p in c.model.parameters() if p.requires_grad], lr=.001)
    restored = load(tmp_path/"query.pt", c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored); equivalent(ob.state_dict(), oc.state_dict())
    tail = [x for x in xb if x.time >= cut]
    ob.zero_grad(set_to_none=True); oc.zero_grad(set_to_none=True)
    expected = execute(first.continuation.detach(), tail, stop, sealed_until=stop)
    actual = runner(c, native)(restored, tail, stop, sealed_until=stop)
    equivalent(expected, actual)
    objective(expected).backward(); objective(actual).backward(); ob.step(); oc.step()
    equivalent(b.model.state_dict(), c.model.state_dict()); equivalent(ob.state_dict(), oc.state_dict())
    if family == "hb":
        for node in c.model.nodes:
            read = node.read_program
            assert read.key is read.evidence.encoders["mem"] is read.score.encoders["mem"]


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_retained_cut_and_independent_roots(dtype, native, monkeypatch):
    b, xs, stop = fixture(dtype, "pdg")
    q = Continuation(b.graph.identity, 2); cut = 3
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    execute = runner(b, native)
    head = execute(q, [x for x in xs if x.time < cut], cut, sealed_until=cut)
    actual = execute(head.continuation, [x for x in xs if x.time >= cut], stop, sealed_until=stop)
    equivalent(expected.continuation, actual.continuation)
    ref_tail = reference(b.graph, b.model, head.continuation, [x for x in xs if x.time >= cut], stop, sealed_until=stop)
    equivalent(ref_tail, actual); roots = variables(b, xs)
    equivalent(vjp(objective(ref_tail), roots), vjp(objective(actual), roots))
    for name in ref_tail.trace[0]["descriptor"]:
        equivalent(vjp(ref_tail.trace[0]["descriptor"][name], roots), vjp(actual.trace[0]["descriptor"][name], roots))
    first = next(e for e in actual.trace if e["active"])
    ref = next(e for e in ref_tail.trace if e["active"])
    for slot in first["emitted"]:
        equivalent(vjp(ref["emitted"][slot], roots), vjp(first["emitted"][slot], roots))
    if not native:
        def forbidden(*a, **kw):
            raise AssertionError("batched training replayed scalar Read or Full")
        for node in b.model.nodes:
            monkeypatch.setattr(node.read_program, "step", forbidden)
            monkeypatch.setattr(node.full_program, "step", forbidden)
        result = execute(q, xs, stop, sealed_until=stop)
        equivalent(expected, result); stats(result)
        equivalent(vjp(objective(expected), roots), vjp(objective(result), roots))
