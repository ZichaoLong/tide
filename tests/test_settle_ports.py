"""Core-5 Settle §2, §6, §9–11: ranks, named slots and exact identity embedding.

Literal path coefficients check formulas; independent time/region/native orders
check records and VJPs. See docs/settle-embedding.md for the acceptance mapping.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, Graph, Node, Region, State
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.reference import run as reference
from tidegraph.settle import SettleGraph
from tidegraph.settle_ports import decode_slots, run_ports
from isolated_cases import vjp


def fixture(dtype, identity=False):
    g = Graph(tuple(Node(r, identity=identity) for r in range(3)),
              (Edge(0, 2, 2), Edge(0, 2, 2), Edge(1, 2, 2)),
              (Region(1), Region(1), Region(1)), (0, 1, 2, 2), (0, 2, 2))
    spec = SettleGraph(g, (1, 1, 3), binding="ports", time_stride=5)
    m = Model(g, width=2, dtype=dtype)
    if not identity:
        m.nodes[1] = m.nodes[0]  # Shared parameters, separate states and regions.
        m.output_scale[2] = m.input_scale[0]
    q = Continuation(g.identity, 1)
    variables = dict(m.named_parameters())
    for node in range(3):
        s = torch.full((2,), .1+.03*node, dtype=dtype, requires_grad=True)
        q.states[0, node] = State(s)
        variables[f"initial.{node}"] = s
    xs = []
    for port, positions in enumerate(((0, 2), (0, 1), (0, 2), (1,))):
        for occurrence, position in enumerate(positions):
            x = torch.tensor([.2+.07*port, -.1+.03*occurrence], dtype=dtype, requires_grad=True)
            variables[f"input.{port}.{occurrence}"] = x
            xs += spec.port_external(x[None], (position,), port=port, start_occurrence=occurrence)
    return spec, m, q, xs, variables


def engine(name, spec, model):
    if name == "reference":
        return lambda q, xs, stop: reference(spec.graph, model, q, xs, stop, sealed_until=stop, mode="hst")
    if name.startswith("python"):
        return lambda q, xs, stop: run_ports(spec, model, q, xs, stop, sealed_until=stop, mode="hst",
                                            prefill=name == "python-prefill", packed=name == "python-prefill")
    import _tide_native as core
    from tidegraph.native import Native
    from tidegraph.native_records import from_continuation, to_continuation, window_records
    body = Native(spec.graph, model)
    compiled = core.SettleGraph(body.compiled, spec.ranks, spec.binding, spec.stride)
    assert compiled.encoded_graph.identity == body.compiled.identity
    assert len(compiled.embed_model(body.weights).nodes) == len(spec.graph.nodes)
    options = core.Options(); options.mode = "hst"; options.packed = True
    executor = core.SettleExecutor(compiled, body.weights, options, name.removeprefix("native-"))
    def run(q, xs, stop):
        cq = to_continuation(core, spec.graph, body.compiled, q)
        rows = [core.External(x.batch, x.port, x.position, x.time, x.value) for x in xs]
        out = compiled.project(executor.run_ports(cq, rows, stop, stop))
        return Result(from_continuation(spec.graph, out.continuation), *window_records(out))
    return run


def test_named_slots_literal_parallel_paths(dtype):
    spec, m, q, xs, variables = fixture(dtype, identity=True)
    with torch.no_grad():
        for bank, constants in ((m.input_scale, (1, 2, 3, 4)), (m.edge_scale, (2, 3, 4)),
                                (m.agg_scale, (5, 7, 11)), (m.output_scale, (13, 17, 19))):
            for parameter, value in zip(bank, constants):
                parameter.fill_(value)
    # Zero input remains a present port and contributes a distinct atom.
    with torch.no_grad():
        variables["input.2.1"].zero_()
    out = run_ports(spec, m, q, xs, 15, sealed_until=15)
    edges, ports = decode_slots(spec, out, 0, 3)
    x, y, z, w = (variables[f"input.{p}.0"] for p in range(4))
    h0 = 31*x+88*y+3*z  # Two physical edges from node 0: 2*5 + 3*7.
    h1 = 88*variables["input.1.1"]+4*w
    h2 = 31*variables["input.0.1"]+3*variables["input.2.1"]
    for t, h in enumerate((h0, h1, h2)):
        equivalent(ports[0, t, 1], 17*h)
        equivalent(ports[0, t, 2], 19*h)
    assert ports[0, 1, 0] is None and edges[0, 1, 0] is None
    equivalent(edges[0, 0, 0], 2*x); equivalent(edges[0, 0, 1], 3*x)
    grads = torch.autograd.grad(ports[0, 0, 1].sum(), (x, y, z, w), allow_unused=True)
    equivalent(grads, (torch.full_like(x, 17*31), torch.full_like(y, 17*88),
                       torch.full_like(z, 17*3), None))
    assert out.continuation.ledger[0, 0] == (1, 11)  # occurrence 1, global position 2
    assert {e["node"] for e in out.trace if e["time"] == 1} == {0, 1}
    fiber = next(e["fiber"] for e in out.trace if (e["node"], e["time"]) == (2, 3))
    assert len(fiber) == 4 and sum(a.kind == 1 for a in fiber) == 3
    assert spec.embed(m) == (spec.graph, m)


@pytest.mark.parametrize("implementation", ["python-streaming", "python-prefill", "native-streaming", "native-frontier", "native-greedy"])
def test_every_cut_records_independent_roots_and_continuation(dtype, implementation):
    spec, model, q, xs, variables = fixture(dtype)
    expected = engine("reference", spec, model)
    other, om, oq, oxs, ov = fixture(dtype)
    actual = engine(implementation, other, om)
    # Every cut includes empty gaps, nonempty pending and equal-rank events.
    for cut in range(16):
        left = expected(q, [x for x in xs if x.time < cut], cut)
        right = actual(oq, [x for x in oxs if x.time < cut], cut)
        equivalent(left, right)
        roots = lambda r: ([v for *_, v in r.outputs] + [s.value for s in r.continuation.states.values()]
                           + [a.value for a in r.continuation.pending])
        for a, b in zip(roots(left), roots(right)):
            for zero in (False, True):
                equivalent(vjp(a, variables, zero), vjp(b, ov, zero))
        a = expected(left.continuation, [x for x in xs if x.time >= cut], 15)
        b = actual(right.continuation, [x for x in oxs if x.time >= cut], 15)
        equivalent(a, b)
        whole = actual(oq, oxs, 15)
        joined = Result(b.continuation, right.trace+b.trace, right.outputs+b.outputs, right.messages+b.messages, {})
        equivalent(whole, joined)
        equivalent(vjp(objective(whole), ov), vjp(objective(joined), ov))


@pytest.mark.parametrize("implementation", ["python-prefill", "native-greedy"])
def test_shared_optimizer_and_checkpoint_with_pending(dtype, implementation, tmp_path):
    spec, m, q, xs, _ = fixture(dtype)
    other, om, oq, oxs, _ = fixture(dtype)
    optim = torch.optim.AdamW(m.parameters(), lr=.001)
    ooptim = torch.optim.AdamW(om.parameters(), lr=.001)
    expected = engine("reference", spec, m)
    actual = engine(implementation, other, om)
    for step in range(3):
        # Fresh initial tensor roots per optimizer graph, preserving shared owners.
        a, b = expected(q.detach(), xs, 15), actual(oq.detach(), oxs, 15)
        equivalent(a, b)
        optim.zero_grad(set_to_none=True); ooptim.zero_grad(set_to_none=True)
        objective(a).backward(); objective(b).backward()
        equivalent([p.grad for p in m.parameters()], [p.grad for p in om.parameters()])
        optim.step(); ooptim.step()
        equivalent(m.state_dict(), om.state_dict())
    part = actual(oq.detach(), [x for x in oxs if x.time < 2], 2)
    assert len(part.continuation.pending) == 3
    path = tmp_path / "settle.pt"
    save(path, other.graph, om, part.continuation, ooptim)
    rs, rm, _, rxs, _ = fixture(dtype)
    roptim = torch.optim.AdamW(rm.parameters(), lr=.001)
    restored = load(path, rs.graph, rm, roptim)
    equivalent(part.continuation.detach(), restored)
    equivalent(ooptim.state_dict(), roptim.state_dict())
    assert rm.nodes[0] is rm.nodes[1] and rm.output_scale[2] is rm.input_scale[0]
    tail = actual(part.continuation.detach(), [x for x in oxs if x.time >= 2], 15)
    replay = engine(implementation, rs, rm)(restored, [x for x in rxs if x.time >= 2], 15)
    equivalent(tail, replay)
    ooptim.zero_grad(set_to_none=True); roptim.zero_grad(set_to_none=True)
    objective(tail).backward(); objective(replay).backward()
    ooptim.step(); roptim.step()
    equivalent(om.state_dict(), rm.state_dict())


def test_time_code_validation_and_sparse_int64(dtype):
    spec, model, q, xs, _ = fixture(dtype)
    for ranks in ((True, 1, 3), (1.0, 1, 3), (1, 1, 2**63), (1, 2, 3)):
        with pytest.raises(ValueError):
            SettleGraph(spec.graph, ranks, binding="ports")
    with pytest.raises(ValueError, match="stride"):
        replace(spec, time_stride=3)
    with pytest.raises(ValueError, match="time code"):
        run_ports(spec, model, q, [replace(xs[0], time=2)], 3, sealed_until=3)
    for positions in ((2, 1), (0, 0), (-1,), (True,), (2**63-1,)):
        with pytest.raises(ValueError):
            spec.port_external(torch.ones(len(positions), 2, dtype=dtype), positions, port=0)
    large = 2**54+7  # Must not be packed as floating point or scanned densely.
    inputs = spec.port_external(torch.ones(1, 2, dtype=dtype), (large,), port=0)
    result = run_ports(spec, model, q, inputs, large*5+4, sealed_until=large*5+4)
    assert [e["time"] for e in result.trace] == [large*5+1, large*5+3]
    assert result.continuation.ledger[0, 0] == (0, large*5+1)


@pytest.mark.parametrize("implementation", ["python-prefill", "native-greedy"])
def test_partial_full_slots_zero_messages_and_empty_selection(dtype, implementation):
    # A sparse Full has one present-zero edge, one absent parallel edge and an
    # independent present-zero output. Another candidate is not selected.
    g = Graph((Node(0, emission="slot_affine", emit_phases=(-1, -2, -1)), Node(0), Node(1)),
              (Edge(0, 2, 2), Edge(0, 2, 2), Edge(1, 2, 2)),
              (Region(1, count_priority=False, selector="positive-v1"), Region(1)),
              (0, 1), (0, 2))
    spec = SettleGraph(g, (1, 3), binding="ports", time_stride=5)
    m = Model(g, width=2, dtype=dtype)
    with torch.no_grad():
        for node in m.nodes: node.read.fill_(1)
        for value in m.nodes[0].extra.values(): value.zero_()
    inputs = spec.port_external(torch.ones(1, 2, dtype=dtype), (0,), port=0)
    inputs += spec.port_external(-torch.ones(1, 2, dtype=dtype), (0,), port=1)
    q = Continuation(g.identity, 1)
    out = engine(implementation, spec, m)(q, inputs, 5)
    equivalent(engine("reference", spec, m)(q, inputs, 5), out)
    edges, ports = decode_slots(spec, out, 0, 1)
    equivalent(edges[0, 0, 0], torch.zeros(2, dtype=dtype))
    assert edges[0, 0, 1] is edges[0, 0, 2] is None
    equivalent(ports[0, 0, 0], torch.zeros(2, dtype=dtype))
    assert [e["node"] for e in out.trace if e["active"]] == [0, 2]
    assert out.continuation.states[0, 1].observations == 1
    negative = [replace(x, value=-x.value.abs()) for x in inputs]
    empty = engine(implementation, spec, m)(q, negative, 5)
    assert not empty.messages and not empty.outputs
    assert len(empty.trace) == 2 and all(not e["active"] for e in empty.trace)
