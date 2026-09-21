from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, PortLayout, Region
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.frontier import run as frontier


def graph():
    return Graph((Node(0), Node(0), Node(1), Node(2)),
                 (Edge(0, 2, 1), Edge(0, 2, 1), Edge(1, 2, 1), Edge(2, 3, 1)),
                 (Region(2), Region(1), Region(1)), (0, 1, 2), (0, 2, 3),
                 PortLayout((2, 0, 0, 1), (3, 0, 1, 0), (0, 0, 2), (1, 0, 0)))


def check_indexes(incoming, outgoing):
    assert incoming == ((0, 1, 2, 6, 7), ((0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (1, 0), (1, 3)))
    assert outgoing == ((0, 3, 4, 6, 7), ((1, 1), (0, 0), (1, 0), (1, 2), (0, 1), (1, 3), (0, 2)))


def test_python_local_slot_bijection_and_parallel_edge_identity():
    g = graph()
    check_indexes(*((i.offsets, i.bindings) for i in g.port_indexes))
    assert g.ports.incoming_slot(0, 2) == 2
    assert g.ports.incoming_slot(1, 0) == 3
    default = replace(g, layout=None)
    assert default.identity == replace(default, layout=default.ports).identity
    assert default.identity != g.identity
    # Replacing an automatically laid-out topology recalculates the slots.
    changed = replace(default, edges=default.edges + (Edge(3, 0, 1),))
    assert len(changed.ports.edge_source) == 5


def test_native_slot_indexes_and_default_canonical_identity(dtype):
    g = graph(); engine = Native(g, Model(g, dtype=dtype))
    check_indexes(*((tuple(i.offsets), tuple((b.kind, b.id) for b in i.bindings))
                    for i in (engine.compiled.incoming_ports, engine.compiled.outgoing_ports)))
    different = replace(g, layout=replace(g.ports, edge_source=(0, 2, 0, 1)))
    assert different.identity != g.identity
    assert Native(different, Model(different, dtype=dtype)).compiled.identity != engine.compiled.identity
    default = replace(g, layout=None)
    native = Native(default, Model(default, dtype=dtype)).compiled
    identity = native.identity
    native.layout = None; native.compile()
    assert native.identity == identity
    for name in ("edge_source", "edge_target", "input", "output"):
        assert tuple(getattr(native.layout, name)) == getattr(default.ports, name)


@pytest.mark.parametrize("field,values", [("edge_source", (2, 2, 0, 1)), ("edge_target", (3, 0, 1)),
                                           ("input", (0, 0, 4)), ("output", (2, 0, 0)),
                                           ("edge_target", (-1, 0, 1, 0))])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_malformed_layout_rejected_before_execution(dtype, field, values, implementation):
    g = graph()
    if implementation == "python":
        with pytest.raises(ValueError, match="slot"):
            replace(g, layout=replace(g.ports, **{field: values}))
    else:
        native = Native(g, Model(g, dtype=dtype)).compiled
        layout = native.layout; setattr(layout, field, values); native.layout = layout
        with pytest.raises(ValueError, match="slot"):
            native.compile()


def test_layout_change_rejects_checkpoint_and_continuation(dtype, tmp_path):
    g = graph(); m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    path = tmp_path / "layout.pt"; save(path, g, m, q)
    changed = replace(g, layout=replace(g.ports, edge_source=(0, 2, 0, 1)))
    restored = Model(changed, dtype=dtype, seed=29)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, restored)
    equivalent(before, restored.state_dict())
    with pytest.raises(ValueError, match="identity"):
        Native(changed, restored).run(q, [], 0, sealed_until=0)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_settle_embedding_preserves_slots_sharing_and_mixed_fibers(dtype, implementation):
    def fixture():
        g = graph(); spec = SettleGraph(g, (1, 2, 3)); m = Model(g, dtype=dtype)
        m.nodes[1] = m.nodes[0]
        values = (torch.arange(18, dtype=dtype).reshape(2, 3, 3) / 30).requires_grad_()
        return g, spec, m, values, dict(m.named_parameters()) | {"input": values}
    g, spec, m, values, variables = fixture()
    expected = settle(spec, m, Continuation(g.identity, 2), values, mode="hst")
    grad = dict(zip(variables, torch.autograd.grad(objective(expected), list(variables.values()), allow_unused=True)))
    g, spec, m, values, variables = fixture(); encoded, em = spec.embed(m)
    assert em.nodes[0] is em.nodes[1] is m.nodes[0]
    for e in range(len(g.edges)):
        assert (encoded.ports.edge_source[e], encoded.ports.edge_target[e]) == (g.ports.edge_source[e], g.ports.edge_target[e])
    for p, slot in enumerate(g.ports.input):
        assert encoded.ports.edge_target[len(g.edges) + p] == slot
    for p, slot in enumerate(g.ports.output):
        assert encoded.ports.edge_source[len(g.edges) + len(g.inputs) + p] == slot
    stop = spec.stride * values.shape[1]; xs = spec.external(values, encoded=True)
    q = Continuation(encoded.identity, 2)
    result = frontier(encoded, em, q, xs, stop, sealed_until=stop, mode="hst") if implementation == "python" else Native(
        encoded, em, algorithm="frontier", workers=3, packed=True, mode="hst").run(q, xs, stop, sealed_until=stop)
    actual = spec.project(result)
    equivalent(expected, actual)
    equivalent(grad, dict(zip(variables, torch.autograd.grad(objective(actual), list(variables.values()), allow_unused=True))))
