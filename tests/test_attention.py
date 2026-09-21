import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.records import Result
from tidegraph.blocks import canonicalize
from tidegraph.checkpoint import load, save
from attention_cases import fixture, vjp


@pytest.mark.parametrize("window", [0, 3])
@pytest.mark.parametrize("adoption", ["all", "selected", "clear"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("implementation", ["python", "native-streaming", "native-frontier"])
def test_attention_trace_and_cache_vjp(dtype, window, adoption, mode, implementation):
    g, m, q, xs, _, variables = fixture(dtype, window, adoption)
    expected = run(g, m, q, xs, 12, sealed_until=12, mode=mode)
    grad = vjp(expected, variables)
    g, m, q, xs, _, variables = fixture(dtype, window, adoption)
    actual = frontier(g, m, q, xs, 12, sealed_until=12, mode=mode) if implementation == "python" else Native(
        g, m, workers=3, packed=True, algorithm=implementation.split("-")[1], mode=mode).run(q, xs, 12, sealed_until=12)
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))
    if adoption == "clear":
        cleared = [e for e in actual.trace if e["node"] < 2 and e["active"]]
        assert cleared and all(len(e["next_slots"]["key"]) == 0 for e in cleared)
        assert all(len(e["comparison_slots"]["key"]) > 0 for e in cleared)


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_attention_chunk_checkpoint_detach(dtype, cyclic, packed, tmp_path):
    g, m, q, xs, _, variables = fixture(dtype, cyclic=cyclic)
    expected = run(g, m, q, xs, 12, sealed_until=12, mode="hst")
    grad = vjp(expected, variables)
    g, m, q, xs, _, variables = fixture(dtype, cyclic=cyclic)
    engine = Native(g, m, algorithm="streaming" if cyclic else "frontier", workers=3 if packed else 1,
                    packed=packed, mode="hst")
    trace, outputs, messages = [], [], []
    for stop in (0, 3, 7, 12):
        piece = engine.run(q, [a for a in xs if q.cut <= a.time < stop], stop, sealed_until=stop)
        trace += piece.trace; outputs += piece.outputs; messages += piece.messages; q = piece.continuation
    actual = canonicalize(g, Result(q, trace, outputs, messages, {}))
    equivalent(expected, actual); equivalent(grad, vjp(actual, variables))
    truncated = q.detach()
    assert all(not t.requires_grad for s in truncated.states.values() for t in (s.value, *s.slots.values()))
    assert all(not a.value.requires_grad for a in truncated.pending)
    path = tmp_path / "attention.pt"; save(path, g, m, q)
    restored = load(path, g, m); equivalent(q, restored)
    equivalent(run(g, m, truncated, [], 14, sealed_until=14, mode="hst"),
               engine.run(restored, [], 14, sealed_until=14))


@pytest.mark.parametrize("window", [0, 2])
def test_attention_analytic_mean_vjp(dtype, window):
    g = Graph((Node(0, memory="attention", window=window),), (), (Region(1),), (0,), (0,))
    w = Model(g, width=1, dtype=dtype).nodes[0]
    with torch.no_grad():
        w.extra["attn_q"].zero_(); w.extra["attn_k"].zero_()
        w.extra["attn_v"].fill_(1); w.extra["attn_out"].fill_(1)
    h = torch.tensor([[1.0], [3.0]], dtype=dtype, requires_grad=True)
    k = torch.tensor([[[0.7]]], dtype=dtype, requires_grad=True)
    v = torch.tensor([[[0.4]]], dtype=dtype, requires_grad=True)
    old = State(torch.zeros(1, dtype=dtype), observations=1, slots={"key": k, "value": v})
    first, _ = w.prepare(old, h[0], 0); second, _ = w.prepare(first, h[1], 10)
    equivalent(second.value, h.new_tensor([2.0 if window else 4.4 / 3]))
    gh, gk, gv = torch.autograd.grad(second.value.sum(), (h, k, v))
    equivalent(gh, torch.full_like(h, 0.5 if window else 1/3))
    equivalent(gk, torch.zeros_like(k)); equivalent(gv, torch.full_like(v, 0 if window else 1/3))
    states, _ = w.prepare_block(old, h, [0, 10]); equivalent(states, [first, second])


def test_gqa_head_assignment_analytic(dtype):
    g = Graph((Node(0, memory="attention", query_heads=4, kv_heads=2),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=4, dtype=dtype); w = m.nodes[0]
    with torch.no_grad():
        w.extra["attn_q"].zero_(); w.extra["attn_k"].zero_()
        w.extra["attn_v"].copy_(torch.eye(4, dtype=dtype)[:, :2]); w.extra["attn_out"].copy_(torch.eye(4, dtype=dtype))
        m.input_scale[0].fill_(1)
    xs = [External(0, 0, 0, 0, torch.tensor([1., 2., 3., 4.], dtype=dtype))]
    q = Continuation(g.identity, 1)
    for result in (run(g, m, q, xs, 1, sealed_until=1), Native(g, m, packed=True).run(q, xs, 1, sealed_until=1)):
        equivalent(result.trace[0]["proposal"], xs[0].value.new_tensor([1., 1., 2., 2.]))
