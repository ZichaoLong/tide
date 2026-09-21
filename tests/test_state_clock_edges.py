from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, SourceDomain, StateClock
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import Result
from clock_cases import project, loss_grad
from isolated_cases import vjp
from read_cases import execute


@pytest.mark.parametrize("memory", ["lh-add-repeat-v1", "lh-fiber-attention-all-softmax-repeat-v1"])
@pytest.mark.parametrize("implementation", ["reference", "native-serial", "native-packed", "cursor"])
def test_reserved_phase_edges_and_pending_projection_at_every_cut(dtype, memory, implementation):
    clock = StateClock(4, 0, 3)
    node = Node(0, memory=memory, aggregation="sum" if "fiber" in memory else "all_softmax")
    base = Graph((node,), (Edge(0, 0, 1),), (Region(1),), (0,), (0,))
    g = replace(base, nodes=(replace(node, state_clock=clock, emit_period=4, emit_phases=(0, 1, 2, -1)),),
                edges=(Edge(0, 0, 1), Edge(0, 0, 1), Edge(0, 0, 2)),
                source_domain=SourceDomain((1, 1, 1), (0,)))
    m, em = Model(base, width=2, dtype=dtype), Model(g, width=2, dtype=dtype)
    for name in ("decay", "weight", "bias", "read", "extra"):
        setattr(em.nodes[0], name, getattr(m.nodes[0], name))
    for name in ("regions", "input_scale", "output_scale"):
        setattr(em, name, getattr(m, name))
    em.agg_scale = torch.nn.ParameterList([m.agg_scale[0]]*3)
    em.edge_scale = torch.nn.ParameterList([m.edge_scale[0]]*3)
    q = Continuation(base.identity, 2); eq = Continuation(g.identity, 2)
    x = torch.tensor([[.2, -.1], [.1, .3]], dtype=dtype, requires_grad=True)
    xs = [External(b, 0, pos, t, x[b]) for b in range(2) for pos, t in enumerate((0, 4))]
    exs = [replace(a, time=clock.to_global(a.time)) for a in xs]
    leaves = dict(m.named_parameters()) | {"input": x}
    cursor = Native(g, em, packed=True, workers=3, mode="hst").cursor(eq) if implementation == "cursor" else None
    traces, outputs, messages = [], [], []
    for cut in range(1, 11):
        expected = execute("reference", base, m, q, [a for a in xs if a.time < clock.cut(cut)], stop=clock.cut(cut))
        if cursor is not None:
            part = cursor.advance([a for a in exs if cursor.cut <= a.time < cut], cut, sealed_until=cut)
            traces += part.trace; outputs += part.outputs; messages += part.messages
            raw = Result(cursor.snapshot(), traces, outputs, messages, {})
        else:
            raw = execute(implementation, g, em, eq, [a for a in exs if a.time < cut], stop=cut)
        actual = project(raw, base, clock, edge_map=(0, 0, 0), emit_map=(0, 0, 0, 1))
        equivalent(expected, actual)
        assert not any(e["time"] % 4 == 3 for e in raw.trace)
        assert len(raw.continuation.pending) == 2
    equivalent(loss_grad(expected, leaves), loss_grad(actual, leaves))
    for zero in (False, True):
        equivalent(vjp(expected.continuation.pending[0].value, leaves, zero),
                   vjp(actual.continuation.pending[0].value, leaves, zero))


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-packed", "native-frontier"])
def test_full_history_and_input_ledger_retain_global_clock(dtype, implementation):
    clock = StateClock(4, 3, 1)
    g = Graph((Node(0, memory="lh-add-repeat-v1", state_clock=clock, emit_period=4, emit_phases=(3,)),),
              (), (Region(1),), (0,), (0,))
    m = Model(g, width=1, dtype=dtype)
    xs = [External(0, 0, i, t, torch.ones(1, dtype=dtype)) for i, t in enumerate((3, 7))]
    result = execute(implementation, g, m, Continuation(g.identity, 1), xs, stop=8)
    assert [o[1] for o in result.outputs] == [3, 7]
    assert result.continuation.states[0, 0].last_time == result.continuation.history[0, 0].last_time == 7
    assert result.continuation.ledger[0, 0] == (1, 7)


@pytest.mark.parametrize("memory", ["lh-add-repeat-v1", "lh-fiber-attention-all-softmax-repeat-v1"])
def test_native_physical_decode_with_explicit_clock(dtype, memory):
    import _tide_native as core
    from tidegraph.lazy_add import decode
    from tidegraph.fiber_attention import decode_bias
    from clock_cases import fixture
    clock = StateClock(4, 3, 1)
    _, _, _, g, em, eq, _, exs, _ = fixture(dtype, clock, memory)
    result = execute("reference", g, em, eq, [a for a in exs if a.time < 14], stop=14)
    w = em.nodes[0]; native = core.NodeWeights(w.decay, w.weight, w.bias, w.read)
    native.extra = dict(w.extra.items())
    fn = core.decode_add_repeat if "add" in memory else core.decode_fiber_bias
    py = decode if "add" in memory else decode_bias
    state = result.continuation.states[0, 0]
    s = core.State(state.value, state.last_time, state.observations, state.slots)
    actual = fn(native, s, 14, core.StateClock(4, 3, 1))
    equivalent(py(w, state, 14), actual)
