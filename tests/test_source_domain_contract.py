from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region, SourceDomain
from tidegraph.aggregate import request
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.ops import Model
from read_cases import execute
from source_domain_cases import fixture


def test_domain_default_and_physical_layout_are_distinct(dtype):
    base, model, _, encoded, em, *_ = fixture(dtype)
    assert base.source_counts == encoded.source_counts == (1, 1, 3)
    assert len(encoded.port_indexes[0].row(2)) == 5
    assert base.identity == replace(base, source_domain=base.domain).identity
    assert encoded.identity != replace(encoded, source_domain=None).identity
    engine = Native(encoded, em)
    assert tuple(engine.compiled.source_counts) == encoded.source_counts
    assert engine.compiled.identity.startswith("tide-graph-v13;")
    default = Native(base, model).compiled
    identity = default.identity; default.source_domain = None; default.compile()
    assert default.identity == identity
    assert tuple(default.source_domain.edge_target) == base.domain.edge_target
    changed = replace(encoded, edges=encoded.edges + (Edge(2, 1, 1),), source_domain=None)
    assert len(changed.domain.edge_target) == 5


@pytest.mark.parametrize("edge,input", [((0,), (0, 0, 0)), ((1, 4), (0, 0, 0)),
                                       ((-1, 2), (0, 0, 0)), ((1, 2), (0,)),
                                       ((1, 2), (0, 0, 2**63-1))])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_invalid_domains_fail_before_execution(dtype, edge, input, implementation):
    g, m, *_ = fixture(dtype)
    if implementation == "python":
        with pytest.raises(ValueError, match="source domain"):
            replace(g, source_domain=SourceDomain(edge, input))
    else:
        native = Native(g, m).compiled
        domain = native.source_domain; domain.edge_target = edge; domain.input = input
        native.source_domain = domain
        with pytest.raises(ValueError, match="source domain"):
            native.compile()


@pytest.mark.parametrize("value", [True, 1.0, 2**63, -1])
def test_domain_integer_guards(value):
    with pytest.raises(ValueError, match="source domain"):
        SourceDomain((value,), ())


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
@pytest.mark.parametrize("kind", ["edges", "ports", "mixed"])
def test_logical_collision_is_not_silently_summed(dtype, implementation, kind):
    if kind == "edges":
        g = Graph((Node(0), Node(1)), (Edge(0, 1, 1),)*2, (Region(1),)*2, (0,), (1,),
                  source_domain=SourceDomain((0, 0), (0,)))
        records = [(0, 0)]
    elif kind == "mixed":
        g = Graph((Node(0), Node(1)), (Edge(0, 1, 1),), (Region(1),)*2, (0, 1), (1,),
                  source_domain=SourceDomain((0,), (0, 0)))
        records = [(0, 0), (1, 1)]
    else:
        g = Graph((Node(0),), (), (Region(1),), (0, 0), (0,), source_domain=SourceDomain((), (0, 0)))
        records = [(0, 0), (1, 0)]
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    xs = [External(0, p, 0, t, torch.zeros(3, dtype=dtype)) for p, t in records]
    with pytest.raises((ValueError, RuntimeError), match="duplicate logical source"):
        execute(implementation, g, m, q, xs, stop=3)
    assert not q.states and not q.history and not q.ledger and not q.pending


def test_domain_identity_blocks_checkpoint_and_continuation_before_mutation(dtype, tmp_path):
    _, _, _, g, m, q, *_ = fixture(dtype)
    path = tmp_path/"domain.pt"; save(path, g, m, q)
    # Permute two logical source roles without changing tensor shapes.
    changed = replace(g, source_domain=replace(g.domain, edge_target=(1, 2, 2, 1)))
    new = Model(changed, width=4, dtype=dtype, seed=91)
    before = {k: v.clone() for k, v in new.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, new)
    equivalent(before, new.state_dict())
    with pytest.raises(ValueError, match="identity"):
        Native(changed, new).run(q, [], 0, sealed_until=0)


def test_logical_slots_do_not_rewrite_physical_tags_or_receive_parameters(dtype):
    _, _, _, g, m, _, _, _, _ = fixture(dtype)
    from tidegraph import Atom
    a = Atom(0, 2, 3, 1, 0, 1, torch.ones(4, dtype=dtype))
    r = request(g, m, [a])
    assert r.slots == 3 and r.sources[0].slot == 2
    assert r.sources[0].atom is a and r.sources[0].scale is m.agg_scale[0]


@pytest.mark.parametrize("kind", ["active_softmax", "all_softmax"])
@pytest.mark.parametrize("implementation", ["reference", "native-packed"])
def test_logical_softmax_denominator_and_absent_source_vjp(dtype, kind, implementation):
    g = Graph((Node(0, aggregation=kind),), (), (Region(1),), (0, 0, 0), (0,),
              source_domain=SourceDomain((), (0, 1, 0)))
    m = Model(g, width=1, dtype=dtype); w = m.nodes[0]
    with torch.no_grad():
        w.extra["agg_logit_0"].fill_(2.0); w.extra["agg_logit_1"].fill_(2.0)
        m.input_scale[0].fill_(1.0)
    x = torch.tensor([4.0], dtype=dtype, requires_grad=True)
    r = execute(implementation, g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, x)], stop=1)
    h = r.trace[0]["content"]
    equivalent(h, x.new_tensor([2.0 if kind == "all_softmax" else 4.0]))
    grads = torch.autograd.grad(h.sum(), [w.extra["agg_logit_0"], w.extra["agg_logit_1"], m.input_scale[2]], allow_unused=True)
    equivalent(grads, (x.new_tensor(1.0), x.new_tensor(-1.0), None) if kind == "all_softmax" else (x.new_tensor(0.0), None, None))
