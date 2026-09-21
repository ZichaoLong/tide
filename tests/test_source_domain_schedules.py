from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.settle import SettleGraph, run as settle
from isolated_cases import vjp
from read_cases import execute
from source_domain_cases import PROFILES, fixture, project


def loss_grad(result, leaves):
    return dict(zip(leaves, torch.autograd.grad(objective(result), list(leaves.values()),
                                                allow_unused=True, retain_graph=True)))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
def test_phase_aliases_preserve_complete_projection_and_isolated_vjps(dtype, profile, implementation):
    base, model, q, g, m, eq, xs, leaves, mapping = fixture(dtype, profile)
    expected = execute("reference", base, model, q, xs, stop=8)
    actual = project(execute(implementation, g, m, eq, xs, stop=8), base, g, mapping)
    equivalent(expected, actual)
    # Root independent public tensors. objective() is already a quadratic loss;
    # differentiate it directly below instead of squaring that sum a second time.
    def public_roots(result):
        return (next(x for b, _, p, x in result.outputs if b == 0 and p == 2),
                result.continuation.states[0, 2].value, result.continuation.pending[0].value)
    for a, b in zip(public_roots(expected), public_roots(actual)):
        for zero in (False, True):
            equivalent(vjp(a, leaves, zero), vjp(b, leaves, zero))
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))
    first = next(e for e in expected.trace if e["node"] == 2)
    other = next(e for e in actual.trace if e["node"] == 2)
    equivalent(vjp(first["contributions"][1], leaves), vjp(other["contributions"][1], leaves))
    if profile.startswith("fiber-"):
        for name, value in expected.continuation.states[0, 2].slots.items():
            equivalent(vjp(value, leaves), vjp(actual.continuation.states[0, 2].slots[name], leaves))


@pytest.mark.parametrize("profile", ["all_softmax", "fiber-all-softmax"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("clear", [False, True])
def test_cycle_parallel_cursor_every_cut(dtype, profile, mode, clear):
    base, model, q, g, m, eq, xs, leaves, mapping = fixture(dtype, profile, cyclic=True, clear=clear)
    cursor = Native(g, m, packed=True, workers=3, mode=mode).cursor(eq)
    trace, outputs, messages = [], [], []
    for stop in range(1, 10):
        part = cursor.advance([x for x in xs if x.time == stop-1], stop, sealed_until=stop)
        trace += part.trace; outputs += part.outputs; messages += part.messages
        from tidegraph.records import Result
        joined = Result(cursor.snapshot(), trace, outputs, messages, {})
        actual = project(joined, base, g, mapping)
        expected = execute("reference", base, model, q, [x for x in xs if x.time < stop], stop=stop, mode=mode)
        equivalent(expected, actual)
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))


@pytest.mark.parametrize("profile", ["all_softmax", "fiber-all-softmax"])
def test_settle_embedding_preserves_logical_domain_and_parameter_aliases(dtype, profile):
    _, _, _, g, m, q, _, leaves, _ = fixture(dtype, profile)
    spec = SettleGraph(g, (1, 3)); eg, em = spec.embed(m)
    assert eg.source_counts[:3] == g.source_counts
    assert eg.domain.edge_target[:len(g.edges)] == g.domain.edge_target
    assert eg.domain.edge_target[len(g.edges):len(g.edges)+len(g.inputs)] == g.domain.input
    assert em.agg_scale[0] is em.agg_scale[3]
    x = (torch.arange(36, dtype=dtype).reshape(3, 3, 4)/30).requires_grad_()
    leaves = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, q, x, mode="hst")
    raw = Native(eg, em, algorithm="frontier", packed=True, workers=3, mode="hst").run(
        spec.embed_initial(q, eg), spec.external(x, encoded=True), 3*spec.stride, sealed_until=3*spec.stride)
    actual = spec.project(raw); equivalent(expected, actual)
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))


def train(dtype, profile, aliases):
    base, model, q, encoded, em, eq, xs, _, mapping = fixture(dtype, profile, cyclic=True)
    g, m, q = (encoded, em, eq) if aliases else (base, model, q)
    optimizer = torch.optim.AdamW(m.parameters(), lr=.003)
    records = []
    for stop in (3, 9):
        optimizer.zero_grad(set_to_none=True)
        raw = execute("native-packed" if aliases else "reference", g, m, q,
                      [x for x in xs if q.cut <= x.time < stop], stop=stop)
        result = project(raw, base, encoded, mapping) if aliases else raw
        objective(result).backward(); optimizer.step(); q = raw.continuation.detach()
        records.append(({k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()},
                        {k: p.clone() for k, p in model.state_dict().items()}))
    return records, g, m, q, optimizer


@pytest.mark.parametrize("profile", ["all_softmax", "fiber-all-softmax"])
def test_shared_alias_optimizer_checkpoint_and_resume(dtype, profile, tmp_path):
    expected, *_ = train(dtype, profile, False)
    actual, g, m, q, optimizer = train(dtype, profile, True)
    equivalent(expected, actual)
    path = tmp_path/"aliases.pt"; save(path, g, m, q, optimizer)
    _, _, _, _, restored, _, _, _, _ = fixture(dtype, profile, cyclic=True)
    ro = torch.optim.AdamW(restored.parameters(), lr=.003)
    loaded = load(path, g, restored, ro)
    equivalent(q, loaded); equivalent(m.state_dict(), restored.state_dict())
    equivalent(optimizer.state_dict(), ro.state_dict())
    assert restored.agg_scale[0] is restored.agg_scale[3]
    assert restored.edge_scale[1] is restored.edge_scale[2]
    expected = execute("reference", g, m, q, [], stop=12)
    actual = execute("native-packed", g, restored, loaded, [], stop=12)
    equivalent(expected, actual)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_alias_packed_inference_does_not_replay(dtype, context):
    with context():
        base, model, q, g, m, eq, xs, _, mapping = fixture(dtype, "fiber-all-softmax")
        expected = execute("reference", base, model, q, xs, stop=8)
        result = execute("native-frontier", g, m, eq, xs, stop=8)
        actual = project(result, base, g, mapping)
    equivalent(expected, actual)
    assert result.stats["state_blocks"] > 0
    assert not any(value for name, value in result.stats.items() if name.startswith("semantic_") and "replay" in name)
