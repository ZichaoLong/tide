from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.lh_full import PROFILES
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import settle_chain, run as specialized
from isolated_cases import vjp
from lh_full_cases import fixture
from read_cases import execute


@pytest.mark.parametrize("profile", ["lh-relu-rms-v1", "lh-silu-layer-v1"])
@pytest.mark.parametrize("topology", ["self_loop", "chain"])
def test_lh_full_independent_fixed_topologies(dtype, profile, topology):
    from tidegraph.reference import run
    count = 1 if topology == "self_loop" else 2
    g = Graph(tuple(Node(i, memory="lh-add-repeat-v1", full=profile, clear=True, emission="slot_affine")
                    for i in range(count)), (Edge(0, 0, 2),) if count == 1 else (Edge(0, 1, 2),),
              (Region(1),)*count, (0,), (count-1,))
    m = Model(g, width=3, dtype=dtype); q = Continuation(g.identity, 2)
    x = torch.tensor([[.2, -.1, .3], [.1, .2, -.3]], dtype=dtype, requires_grad=True)
    xs = [External(b, 0, 0, b, x[b]) for b in range(2)]
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = run(g, m, q, xs, 7, sealed_until=7, mode="hst")
    for actual in (specialized(g, m, q, xs, 7, sealed_until=7, topology=topology, mode="hst"),
                   Native(g, m, algorithm=topology, mode="hst").run(q, xs, 7, sealed_until=7)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_lh_full_schedule_roots(dtype, profile, clear, implementation):
    g, m, q, xs, leaves = fixture(dtype, profile, clear)
    expected = execute("reference", g, m, q, xs, stop=5)
    actual = execute(implementation, g, m, q, xs, stop=5)
    equivalent(expected, actual)
    for root in ("output", "pending", "state"):
        equivalent(vjp(objective(expected, root), leaves), vjp(objective(actual, root), leaves))
    if clear:
        assert any(e["active"] and e["full"].abs().sum() > 0 for e in actual.trace)
        assert all(torch.count_nonzero(e["next"]) == 0 for e in actual.trace if e["active"])


@pytest.mark.parametrize("profile", ["lh-relu-rms-v1", "lh-silu-layer-v1"])
def test_lh_full_cut_detach_checkpoint_and_shared_norm(dtype, profile, tmp_path):
    g, m, q, xs, leaves = fixture(dtype, profile, clear=True)
    engine = Native(g, m, workers=3, packed=True, mode="softp")
    whole = engine.run(q, xs, 8, sealed_until=8)
    prefix = engine.run(q, [x for x in xs if x.time < 3], 3, sealed_until=3)
    suffix_x = [x for x in xs if x.time >= 3]
    suffix = engine.run(prefix.continuation, suffix_x, 8, sealed_until=8)
    joined = replace(suffix, outputs=prefix.outputs+suffix.outputs, trace=prefix.trace+suffix.trace,
                     messages=prefix.messages+suffix.messages)
    equivalent(whole, joined); equivalent(vjp(objective(whole), leaves), vjp(objective(joined), leaves))
    detached = engine.run(prefix.continuation.detach(), suffix_x, 8, sealed_until=8)
    assert vjp(objective(detached), leaves)["input.0.0.0"] is None
    optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
    objective(whole).backward(); optimizer.step()
    path = tmp_path / "lh-full.pt"; save(path, g, m, whole.continuation, optimizer)
    _, restored, _, _, _ = fixture(dtype, profile, clear=True)
    ro = torch.optim.AdamW(restored.parameters(), lr=.001)
    equivalent(load(path, g, restored, ro), whole.continuation.detach())
    equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())


@pytest.mark.parametrize("profile", ["lh-relu-rms-v1", "lh-silu-layer-v1"])
def test_lh_full_settle_embedding_and_specialization(dtype, profile):
    g = Graph((Node(0, memory="lh-add-repeat-v1", full=profile, emission="slot_affine"),
               Node(1, memory="lh-add-repeat-v1", full=profile, emission="slot_affine")),
              (Edge(0, 1, 2),), (Region(1),)*2, (0,), (1,))
    m = Model(g, width=3, dtype=dtype); spec = SettleGraph(g, (1, 3)); eg, em = spec.embed(m)
    q = Continuation(g.identity, 2)
    x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3)/20-.3).requires_grad_()
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode="hst")
    encoded = Native(eg, em, algorithm="frontier", workers=2, packed=True, mode="hst").run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 3*spec.stride, sealed_until=3*spec.stride)
    for actual in (spec.project(encoded), settle_chain(spec, m, q, x, mode="hst")):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
