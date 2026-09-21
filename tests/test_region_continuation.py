from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, History, Node, Region
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph, run as settle
from isolated_cases import vjp
from read_cases import execute
from region_cases import fixture


@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_region_cut_composition_and_detach(dtype, implementation):
    g, m, q, xs, leaves = fixture(dtype)
    whole = execute(implementation, g, m, q, xs)
    prefix = execute(implementation, g, m, q, [x for x in xs if x.time < 3], stop=3)
    suffix_inputs = [x for x in xs if x.time >= 3]
    suffix = execute(implementation, g, m, prefix.continuation, suffix_inputs)
    equivalent(whole.continuation, suffix.continuation); equivalent(whole.trace[4:], suffix.trace)
    equivalent(vjp(objective(whole), leaves), vjp(objective(replace(suffix, outputs=whole.outputs)), leaves))
    detached = execute(implementation, g, m, prefix.continuation.detach(), suffix_inputs)
    gradients = vjp(objective(detached), leaves)
    for name in leaves:
        if name.startswith("history.") or name.startswith("input.") and name.endswith(".0"):
            assert gradients[name] is None
    fork = prefix.continuation.fork()
    fork.history[0, 0].node_maps["selected"].clear()
    fork.history[0, 0].tensors.clear()
    assert prefix.continuation.history[0, 0].node_maps["selected"] == {1: 1}
    assert "memory" in prefix.continuation.history[0, 0].tensors


def test_native_cursor_owns_history_and_preserves_its_vjp(dtype):
    g, m, q, xs, leaves = fixture(dtype)
    engine = Native(g, m, workers=3, packed=True, mode="hst")
    expected = engine.run(q, xs, 6, sealed_until=6)
    cursor = engine.cursor(q)
    # Import cloned tensor storage. Only mutate an idle sample to avoid invalidating
    # an autograd operand subsequently used by the expected run.
    with torch.no_grad():
        q.history[2, 0].tensors["memory"].fill_(100)
    assert cursor.snapshot().history[2, 0].tensors["memory"].item() == 7
    for stop in (3, 6):
        cursor.advance([x for x in xs if cursor.cut <= x.time < stop], stop, sealed_until=stop)
        snapshot = cursor.snapshot()
        saved = snapshot.history[0, 0].tensors["memory"].clone()
        with torch.no_grad():
            snapshot.history[0, 0].tensors["memory"].fill_(99)
        equivalent(cursor.snapshot().history[0, 0].tensors["memory"], saved)
    actual = cursor.snapshot()
    equivalent(expected.continuation.history[0, 0], actual.history[0, 0])
    equivalent(vjp(expected.continuation.history[0, 0].tensors["memory"], leaves),
               vjp(actual.history[0, 0].tensors["memory"], leaves))
    cursor = engine.cursor(fixture(dtype)[2])
    cursor.advance([x for x in xs if x.time < 3], 3, sealed_until=3)
    cursor.detach()
    assert all(not t.requires_grad for h in cursor.snapshot().history.values() for t in h.tensors.values())
    cursor.advance([x for x in xs if x.time >= 3], 6, sealed_until=6)
    gradients = vjp(cursor.snapshot().history[0, 0].tensors["memory"], leaves)
    assert gradients["input.0.0.0"] is None and gradients["input.0.1.0"] is None


def test_region_checkpoint_optimizer_identity_and_history_validation(dtype, tmp_path):
    g, m, q, xs, leaves = fixture(dtype)
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    result = execute("native-frontier", g, m, q, xs)
    objective(result).backward(); optimizer.step()
    path = tmp_path / "region.pt"; save(path, g, m, result.continuation, optimizer)
    _, restored, _, _, _ = fixture(dtype)
    ro = torch.optim.AdamW(restored.parameters(), lr=.01)
    loaded = load(path, g, restored, ro)
    equivalent(loaded, result.continuation.detach()); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), ro.state_dict())
    assert not loaded.history[0, 0].tensors["memory"].requires_grad
    record = torch.load(path, weights_only=True)
    assert record["schema"] == "tide-continuation-v4" and isinstance(record["history"][0, 0], tuple)
    record["weights"]["regions.0.alpha"].fill_(float("nan"))
    bad_weights = tmp_path / "bad-weights.pt"; torch.save(record, bad_weights)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="parameter"):
        load(bad_weights, g, restored, ro)
    equivalent(before, restored.state_dict())
    record = torch.load(path, weights_only=True)
    record["history"][0, 0][3]["memory"] = torch.ones(2, dtype=dtype)
    invalid = tmp_path / "malformed.pt"; torch.save(record, invalid)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="history"):
        load(invalid, g, restored, ro)
    equivalent(before, restored.state_dict())
    changed = replace(g, regions=(replace(g.regions[0], selector="count-v1"), g.regions[1]))
    changed_model = Model(changed, width=1, dtype=dtype)
    before = {k: v.clone() for k, v in changed_model.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, changed_model)
    equivalent(before, changed_model.state_dict())


def test_region_parameter_sharing_local_membership_and_checkpoint(dtype, tmp_path):
    g = Graph(tuple(Node(r) for r in (0, 1, 0, 1)), (),
              tuple(Region(1, selector="tensor-history-v1") for _ in range(2)), (0, 1, 2, 3), (0, 1, 2, 3))
    m = Model(g, width=1, dtype=dtype); m.regions[1] = m.regions[0]
    q = Continuation(g.identity, 1)
    xs = [External(0, v, i, i*2, torch.tensor([v+i+1.], dtype=dtype)) for v in range(4) for i in range(2)]
    leaves = dict(m.named_parameters())
    expected = execute("reference", g, m, q, xs)
    for impl in ("python-frontier", "native-packed", "native-frontier"):
        actual = execute(impl, g, m, q, xs)
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    assert expected.continuation.history[0, 0].tensors["memory"].item() != expected.continuation.history[0, 1].tensors["memory"].item()
    path = tmp_path / "sharing.pt"; save(path, g, m, expected.continuation)
    restored = Model(g, width=1, dtype=dtype)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="sharing"):
        load(path, g, restored)
    equivalent(before, restored.state_dict())
    restored.regions[1] = restored.regions[0]
    equivalent(load(path, g, restored), expected.continuation.detach())


def test_settle_initial_tensor_history_embedding(dtype):
    g = Graph((Node(0),), (), (Region(1, selector="tensor-history-v1"),), (0,), (0,))
    m = Model(g, dtype=dtype); spec = SettleGraph(g, (1,)); eg, em = spec.embed(m)
    h = torch.tensor(2., dtype=dtype, requires_grad=True)
    q = Continuation(g.identity, 1, history={(0, 0): History(node_maps={"selected": {}}, tensors={"memory": h})})
    eq = spec.embed_initial(q, eg)
    invalid = q.fork(); invalid.history[0, 1] = History(node_maps={"selected": {}})
    with pytest.raises(ValueError, match="body initial"):
        spec.embed_initial(invalid, eg)
    x = torch.ones((1, 2, 3), dtype=dtype, requires_grad=True)
    expected = settle(spec, m, q, x, mode="hst")
    actual = spec.project(Native(eg, em, algorithm="frontier", workers=2, mode="hst").run(
        eq, spec.external(x, encoded=True), 6, sealed_until=6))
    equivalent(expected, actual)
    leaves = dict(m.named_parameters()) | {"input": x, "history": h}
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
