"""Column composition against an independently scheduled scalar graph.

Tests decode candidate-computed fields only, including physical multiplicity,
zero inputs, missing ports, tied ranks and ragged original-time schedules.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Graph, Node, Edge, Region
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.settle import SettleGraph
from tidegraph.weighted import bind, KVConfig, RegionalConfig, Payload
from tidegraph.weighted.column_layout import SequenceLayout
from tidegraph.weighted.message_columns import MessageColumns
from tidegraph.weighted.settle_columns import SettleColumnPlan
from isolated_cases import vjp
from dense_checks import Checks
from column_numeric_witnesses import settle as witness


def general(dtype, mode="regional_total", material_dtype=None, *, memory=None, **overrides):
    g = Graph((Node(0), Node(1), Node(2), Node(2)),
              (Edge(0, 2, 3), Edge(0, 2, 3), Edge(1, 3, 3), Edge(0, 3, 3)),
              (Region(1), Region(1), Region(2)), (0, 1, 1, 2), (0, 2, 3))
    region = RegionalConfig(selection_mode=mode, **overrides)
    b = bind(g, memories=memory or KVConfig(width=4, summary_width=2, heads=2, memory_read="qry", window=30),
             regions=region, alpha_agg=.2, dtype=material_dtype or dtype)
    b.model.to(dtype=dtype)
    # Parameter sharing must collect every user; private states stay distinct.
    b.model.nodes[1].kernel.summary = b.model.nodes[0].kernel.summary
    spec = SettleGraph(b.graph, (1, 1, 4), binding="ports", time_stride=5)
    return b, spec


def inputs(dtype, material_dtype=None):
    layout = SequenceLayout(torch.tensor([0, 4, 7]), torch.tensor([0, 2, 5, 9, 1, 4, 8]))
    ports = {}
    for p, rows in enumerate(([0, 2, 3, 4, 6], [1, 3, 5], [0, 5, 6], [0, 1, 4])):
        values = torch.linspace(-.25+p*.01, .5, len(rows)*4, dtype=material_dtype or dtype).to(dtype).reshape(-1, 4).requires_grad_()
        strength = torch.linspace(.2, 1.2, len(rows), dtype=material_dtype or dtype).to(dtype)
        if p == 0:
            strength[0] = 0  # Present zero remains a candidate and a ledger input.
        if p == 2:
            strength[0] = -.1  # Architecture uses max(0,m), not an absence test.
        ports[p] = MessageColumns(torch.tensor(rows), strength.requires_grad_(), values)
    return layout, ports


def external(spec, layout, ports, old=None):
    rows = []
    for p, payload in ports.items():
        owners = layout.owners()[payload.rows]
        for b in range(layout.samples):
            ids = (owners == b).nonzero().flatten()
            positions = layout.times[payload.rows[ids]].tolist()
            start = 0 if old is None else old.ledger.get((b, p), (-1, -1))[0]+1
            rows += spec.port_external(Payload(payload.strength[ids], payload.content[ids]).encode(),
                                       positions, port=p, batch=b, start_occurrence=start)
    return rows


def observations(actual, expected, spec):
    owners = actual.layout.owners()
    columns, scalar = {}, {}
    events = {(e["batch"], e["node"], e["time"]): e for e in expected.trace}
    for v, record in actual.nodes.items():
        for j, row in enumerate(record.rows.tolist()):
            time = int(actual.layout.times[row])*spec.stride+spec.rank(v)
            e = events.pop((int(owners[row]), v, time))
            for name, a, b in (("control", record.controls[j], e["control"]),
                                ("content", torch.cat((record.window.mass[j:j+1], record.window.content[j])), e["content"])):
                columns[v, row, name], scalar[v, row, name] = a, b
            for name, values in record.descriptors.items():
                columns[v, row, name], scalar[v, row, name] = values[j], e["descriptor"][name]
            assert bool(record.controls[j] > 0) == e["active"]
    assert not events
    for name, actual_messages, originals in (
        ("edge", actual.messages, {(a.batch, a.source, a.position): a.value for a in expected.messages}),
        ("port", actual.outputs, {(b, p, t): v for b, t, p, v in expected.outputs})):
        for source, payload in actual_messages.items():
            node = spec.graph.edges[source].source if name == "edge" else spec.graph.outputs[source]
            for j, row in enumerate(payload.rows.tolist()):
                time = int(actual.layout.times[row])*spec.stride+spec.rank(node)
                columns[name, source, row] = torch.cat((payload.strength[j:j+1], payload.content[j]))
                scalar[name, source, row] = originals.pop((int(owners[row]), source, time))
        assert not originals
    for v, state in actual.states.items():
        for b, decoded in enumerate(state.states()):
            old = expected.continuation.states.get((b, v))
            if old is None:
                continue
            assert (decoded.last_time, decoded.observations) == (old.last_time, old.observations)
            for name, x in decoded.slots.items():
                columns["state", b, v, name] = x
                scalar["state", b, v, name] = old.slots[name]
    return columns, scalar


def objective(fields):
    return sum((value*torch.linspace(-.4, .7, value.numel(), dtype=value.dtype,
                                    device=value.device).reshape(value.shape)).sum()
               for value in fields.values() if value.is_floating_point())


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("mode", ["regional_total", "regional_total_zero", "proportional", "regional_threshold", "hard_selection"])
def test_general_ports_messages_states_and_gradients(dtype, mode, implementation, material_dtype=None):
    b, spec = general(dtype, mode, material_dtype)
    layout, ports = inputs(dtype, material_dtype)
    actual = SettleColumnPlan(b, spec, implementation=implementation).run_ports(layout, ports)
    expected = reference(b.graph, b.model, Continuation(b.graph.identity, 2),
                         external(spec, layout, ports), 50, sealed_until=50)
    pairs = observations(actual, expected, spec)
    equivalent(*pairs)
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad}
    roots.update({f"{p}.{name}": getattr(x, name) for p, x in ports.items() for name in ("strength", "content")})
    check = Checks(f"settle-columns:{dtype}:{mode}:{implementation}",
                   known_numeric=witness(dtype, mode, implementation),
                   failure_reason="column FP32 reduction witness; same material FP64 check is separate")
    check(vjp(objective(pairs[0]), roots), vjp(objective(pairs[1]), roots), "gradient")
    equivalent(vjp(objective(pairs[0]), roots, zero=True), vjp(objective(pairs[1]), roots, zero=True))
    check.finish()


@pytest.mark.parametrize("mode", ["regional_total", "regional_total_zero", "proportional", "regional_threshold", "hard_selection"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_same_fp32_material_promoted(mode, implementation):
    test_general_ports_messages_states_and_gradients(torch.float64, mode, implementation, material_dtype=torch.float32)


def test_dense_fc_against_independent_direct_formula(dtype):
    from test_fc_tensor_baseline import fixture
    from fc_tensor_baseline import run
    b, spec = fixture(dtype)
    values = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    actual = SettleColumnPlan(b, spec).run(values)
    direct = run(b, spec, values)
    equivalent(actual.readout(values), direct["hidden"])
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad}
    roots["inputs"] = values
    gradients = [vjp(x.square().sum(), roots) for x in (actual.readout(values), direct["hidden"])]
    gradients = [{n: torch.zeros_like(roots[n]) if x is None else x for n, x in row.items()} for row in gradients]
    equivalent(*gradients)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_default_region_count_uses_real_cardinality_not_member_tuple(implementation):
    # Four different pairs at four positions, all in one region. Arbitrary
    # member subsets must not force four numerical calls or fabricated ranks.
    g = Graph(tuple(Node(0) for _ in range(4)), (), (Region(2),), tuple(range(4)), tuple(range(4)))
    b = bind(g, memories=KVConfig(width=2, summary_width=2, heads=1, ffn_width=4),
             regions=RegionalConfig(k=2), alpha_agg=.2, dtype=torch.float64)
    spec = SettleGraph(b.graph, (1,), binding="ports", time_stride=3)
    layout = SequenceLayout(torch.tensor([0, 4]), torch.tensor([0, 2, 5, 8]))
    ports = {v: MessageColumns(torch.tensor(rows), torch.tensor([.4, .7], dtype=torch.float64, requires_grad=True),
        torch.tensor([[.2+.01*v, -.3], [-.1, .6-.02*v]], dtype=torch.float64, requires_grad=True))
        for v, rows in enumerate(([0, 1], [0, 2], [1, 3], [2, 3]))}
    actual = SettleColumnPlan(b, spec, implementation=implementation).run_ports(layout, ports)
    assert actual.stats["region_column_calls"] == 1
    expected = reference(b.graph, b.model, Continuation(b.graph.identity, 1),
                         external(spec, layout, ports), 30, sealed_until=30)
    pairs = observations(actual, expected, spec)
    equivalent(*pairs)
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad} | {f"{v}.{name}": getattr(p, name)
              for v, p in ports.items() for name in ("strength", "content")}
    equivalent(vjp(objective(pairs[0]), roots), vjp(objective(pairs[1]), roots))
