from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.next import ControlBlendNext
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.specialized import run as specialized, settle_chain
from isolated_cases import fixture as memory_fixture, vjp
from next_cases import blend_fixture
from read_cases import execute


IMPLEMENTATIONS = ["reference", "python-frontier", "python-causal", "native-serial", "native-packed", "native-frontier"]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_control_blend_analytic_recurrence_and_vjp(dtype, implementation):
    g, m, q, xs, leaves = blend_fixture(dtype)
    result = execute(implementation, g, m, q, xs)
    equivalent(result.continuation.states[0, 0].value, torch.tensor([2.5], dtype=dtype))
    equivalent(result.continuation.states[0, 1].value, torch.tensor([3.375], dtype=dtype))
    loss = result.continuation.states[0, 0].value.sum() + result.continuation.states[0, 1].value.sum()
    equivalent(loss, loss.new_tensor(5.875))
    gradients = dict(zip(leaves, torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)))
    expected = {"initial.0.0": .5625, "initial.0.1": .5625, "input.0.0.0": .375, "input.0.1.0": .375,
                "input.0.0.1": .5, "input.0.1.1": .5, "nodes.0.decay": .4375, "nodes.1.decay": .9375,
                "nodes.0.read": 1.4375, "nodes.1.read": .5625}
    for name, value in expected.items():
        equivalent(gradients[name], torch.full_like(leaves[name], value))
    for name in leaves:
        if name.startswith(("input.1.", "initial.1.", "nodes.2.")) or name == "initial.0.2":
            assert gradients[name] is None
    assert not any(e["node"] == 2 for e in result.trace)
    equivalent(q.states[0, 2], result.continuation.states[0, 2])
    first_passive = next(e for e in result.trace if (e["batch"], e["node"], e["time"]) == (0, 1, 1))
    assert not first_passive["active"]
    equivalent(first_passive["next"], torch.tensor([4.5], dtype=dtype))
    last_full = next(e for e in result.trace if (e["batch"], e["node"], e["time"]) == (0, 0, 4))
    equivalent(last_full["full"], torch.tensor([2.], dtype=dtype) + torch.tensor([.6], dtype=dtype).tanh())
    if "frontier" in implementation:
        assert result.stats.get("state_blocks", 0) == 0
        assert result.stats["state_prefill_blocked_next"] == 8
        assert result.stats["full_blocks"] > 0 and result.stats["next_steps"] == 8


@pytest.mark.parametrize("policy", ["all", "selected", "clear"])
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS[1:])
def test_control_blend_schedules_gradients_and_inference(dtype, implementation, policy):
    g, m, q, xs, leaves = blend_fixture(dtype, clear=policy == "clear", observe_all=policy != "selected")
    expected = execute("reference", g, m, q, xs)
    g, m, q, xs, actual_leaves = blend_fixture(dtype, clear=policy == "clear", observe_all=policy != "selected")
    actual = execute(implementation, g, m, q, xs)
    equivalent(expected, actual)
    for root in (lambda r: objective(r), lambda r: r.outputs[0][-1], lambda r: r.continuation.states[0, 0].value):
        equivalent(vjp(root(expected), leaves), vjp(root(actual), actual_leaves))
    with torch.no_grad():
        equivalent(actual, execute(implementation, g, m, q, xs))
    if policy == "clear":
        assert all(e["next"].eq(0).all() for e in actual.trace if e["active"])


@pytest.mark.parametrize("kind", ["ssm", "linear", "delta"])
@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_control_blend_preserves_memory_slots_and_isolated_roots(dtype, kind, implementation):
    g, m, q, xs, leaves = memory_fixture(dtype, kind)
    g = replace(g, nodes=tuple(replace(n, next_state="control-blend-v1") for n in g.nodes)); q.identity = g.identity
    for w in m.nodes:
        w.next_program = ControlBlendNext()
    expected = execute("reference", g, m, q, xs, stop=10)
    actual = execute(implementation, g, m, q, xs, stop=10)
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    for slot in expected.continuation.states[0, 0].slots:
        equivalent(vjp(expected.continuation.states[0, 0].slots[slot], leaves),
                   vjp(actual.continuation.states[0, 0].slots[slot], leaves))


@pytest.mark.parametrize("implementation", ["python-frontier", "native-packed", "native-frontier"])
def test_control_blend_cuts_detach_checkpoint_and_optimizer(dtype, implementation, tmp_path):
    g, m, q, xs, leaves = blend_fixture(dtype)
    full = execute(implementation, g, m, q, xs)
    prefix = execute(implementation, g, m, q, [x for x in xs if x.time < 3], stop=3)
    tail = execute(implementation, g, m, prefix.continuation, [x for x in xs if x.time >= 3])
    equivalent(full.continuation, tail.continuation)
    equivalent(full.trace[4:], tail.trace)
    equivalent(vjp(full.continuation.states[0, 0].value, leaves), vjp(tail.continuation.states[0, 0].value, leaves))
    detached = execute(implementation, g, m, prefix.continuation.detach(), [x for x in xs if x.time >= 3])
    grads = vjp(objective(detached), leaves)
    assert all(grad is None for name, grad in grads.items() if name.startswith("initial.") or
               name.startswith("input.") and name.endswith(".0"))
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    objective(full).backward(); optimizer.step()
    path = tmp_path / "next.pt"; save(path, g, m, prefix.continuation, optimizer)
    _, restored, _, _, _ = blend_fixture(dtype); ro = torch.optim.AdamW(restored.parameters(), lr=.01)
    loaded = load(path, g, restored, ro)
    equivalent(prefix.continuation.detach(), loaded); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), ro.state_dict())
    changed = replace(g, nodes=tuple(replace(n, next_state="adopt-v1") for n in g.nodes))
    changed_model = Model(changed, width=1, dtype=dtype); before = {k: v.clone() for k, v in changed_model.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, changed_model)
    equivalent(before, changed_model.state_dict())


@pytest.mark.parametrize("topology", ["self_loop", "chain"])
def test_control_blend_topology_settle_and_native_cursor(dtype, topology):
    n = 1 if topology == "self_loop" else 2
    edges = (Edge(0, 0, 1),) if n == 1 else (Edge(0, 1, 1),)
    g = Graph(tuple(Node(v, next_state="control-blend-v1") for v in range(n)), edges,
              tuple(Region(1) for _ in range(n)), (0,), (n-1,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 2)
    x = torch.ones((2, 2, 3), dtype=dtype, requires_grad=True); leaves = dict(m.named_parameters()) | {"input": x}
    xs = [External(b, 0, i, i*3, x[b, i]) for b in range(2) for i in range(2)]
    expected = run(g, m, q, xs, 6, sealed_until=6, mode="hst")
    for actual in (specialized(g, m, q, xs, 6, sealed_until=6, topology=topology, mode="hst"),
                   Native(g, m, algorithm=topology, workers=2, packed=True, mode="hst").run(q, xs, 6, sealed_until=6)):
        equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    cursor = Native(g, m, workers=2, packed=True, mode="hst").cursor(q)
    cursor.advance([e for e in xs if e.time < 3], 3, sealed_until=3)
    cursor.advance([e for e in xs if e.time >= 3], 6, sealed_until=6)
    equivalent(expected.continuation, cursor.snapshot())
    if topology == "chain":
        spec = SettleGraph(g, (1, 2)); eg, em = spec.embed(m)
        expected = settle(spec, m, q, x, mode="hst")
        for actual in (settle_chain(spec, m, q, x, mode="hst"),
                       spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True), 8,
                                             sealed_until=8, mode="hst")),
                       spec.project(Native(eg, em, algorithm="frontier", packed=True, workers=2, mode="hst").run(
                           Continuation(eg.identity, 2), spec.external(x, encoded=True), 8, sealed_until=8))):
            equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
