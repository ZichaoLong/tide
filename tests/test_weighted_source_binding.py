"""0770593 architecture §2.2: parameter groups, widths and low-rank formula.

Literal one-dimensional derivatives are independent of production autograd;
alias assertions distinguish shared parameters from private node state.
"""
import math
import pytest
import torch
from tidegraph import Atom, Graph, Node, Region
from tidegraph.aggregate import request
from tidegraph.compare import equivalent
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, SourceBinding, bind


MODES = ("region", "node", "source", "adapted")


def binding(mode):
    return SourceBinding(mode, 1 if mode == "adapted" else None)


def fixture(dtype, mode="region", widths=3, inputs=(0, 0, 1, 1, 2, 2), labels=(0, 1, 0, 1, 0, 1)):
    g = Graph((Node(0), Node(0), Node(1)), (), (Region(2), Region(1)), inputs, (0, 1, 2))
    return bind(g, memories=KVConfig(width=2, summary_width=2), source_binding=binding(mode),
                source_classes=SourceClasses((), labels), source_hidden_width=widths, alpha_agg=.4, dtype=dtype)


@pytest.mark.parametrize("mode", MODES)
def test_parameter_groups_and_private_state(dtype, mode):
    b = fixture(dtype, mode)
    transforms = {(v, int(a)): t for v, w in enumerate(b.model.nodes)
                  for a, t in w.aggregate_program.transforms.items()}
    for (v, a), x in transforms.items():
        for (u, c), y in transforms.items():
            j, k = b.graph.nodes[v].region, b.graph.nodes[u].region
            share = v == u if mode == "node" else (j, a) == (k, c) if mode == "source" else j == k
            assert (x.a is y.a) == share and (x.b is y.b) == share
            embeddings = (v, a) == (u, c) if mode in {"node", "adapted"} else (j, a) == (k, c)
            assert (x.embedding is y.embedding) == embeddings
            if mode == "adapted":
                for name in ("a_left", "a_right", "b_left", "b_right"):
                    assert (getattr(x, name) is getattr(y, name)) == ((v, a) == (u, c))
    states = [w.kernel.initial(w) for w in b.model.nodes]
    assert len({id(q) for q in states}) == len(states)
    for name in ("summary", "adapter_u", "adapter_v"):
        assert getattr(b.model.nodes[0].kernel, name) is not getattr(b.model.nodes[1].kernel, name)


@pytest.mark.parametrize("mode", MODES)
def test_class_width_domain_and_shared_matrix_compatibility(dtype, mode):
    widths = {0: 2, 1: 5}
    if mode == "source":
        b = fixture(dtype, mode, widths)
        assert b.model.nodes[0].aggregate_program.transforms["1"].a.shape == (5, 2)
        # Copy/freeze user configuration; later edits cannot change the binding.
        widths[1] = 7
        b.model.validate_binding(b.graph)
    else:
        with pytest.raises(ValueError, match="sharing A/B"):
            fixture(dtype, mode, widths)
    # Node groups may differ in width if each node receives one class only.
    if mode in {"node", "source"}:
        b = fixture(dtype, mode, widths, (0, 1, 2), (0, 1, 0))
        assert b.model.nodes[1].aggregate_program.transforms["1"].b.shape == (2, 7 if mode == "source" else 5)
    # Region/adapted may differ between regions with disjoint class sets.
    if mode in {"region", "adapted"}:
        b = fixture(dtype, mode, widths, (0, 1, 2), (0, 0, 1))
        assert b.model.nodes[2].aggregate_program.transforms["1"].a.shape == (5, 2)
    for invalid in ({0: 2}, {0: 2, 1: 3, 2: 4}, {0: 0, 1: 3}, {0: True, 1: 3}, {"0": 2, 1: 3}):
        with pytest.raises(ValueError):
            fixture(dtype, mode, invalid)


def test_rank_shape_and_binding_rejections(dtype):
    for mode, rank in (("unknown", None), ("node", 1), ("adapted", None), ("adapted", 0), ("adapted", True)):
        with pytest.raises(ValueError):
            SourceBinding(mode, rank)
    b = fixture(dtype, "adapted")
    t = b.model.nodes[0].aggregate_program.transforms["0"]
    t.a_left = torch.nn.Parameter(torch.zeros(3, 2, dtype=dtype))
    with pytest.raises(ValueError, match="shape mismatch"):
        b.model.validate_binding(b.graph)
    b = fixture(dtype)
    del b.model.nodes[0].aggregate_program.transforms["1"]
    with pytest.raises(ValueError, match="class domain"):
        b.model.validate_binding(b.graph)


def test_adapted_formula_and_all_factor_derivatives(dtype):
    g = Graph((Node(0),), (), (Region(1),), (0, 0), (0,))
    b = bind(g, memories=KVConfig(width=1, summary_width=1, epsilon=1.),
             source_binding=SourceBinding("adapted", 1), source_hidden_width=1, alpha_agg=.4, dtype=dtype)
    w = b.model.nodes[0]; p = w.aggregate_program; t = p.transforms["0"]
    values = dict(a=.7, b=-.4, embedding=.2, a_left=.3, a_right=-.5, b_left=.6, b_right=.2)
    with torch.no_grad():
        for name, value in values.items():
            getattr(t, name).fill_(value)
    x = torch.tensor([.3, .8], dtype=dtype, requires_grad=True)
    req = request(b.graph, b.model, [Atom(0, 0, 0, 0, port, 0, x) for port in (0, 1)])
    # One normalized content, two physical mass contributions. All derivatives
    # below follow ordinary scalar calculus, without calling a model helper.
    a, c = .7+.3*(-.5), -.4+.6*.2
    n = .8/math.sqrt(1+.8**2); h = a*n+.2; s = 1/(1+math.exp(-h))
    df = c*s*(1+h*(1-s)); da = .4*df*n; db = .4*h*s
    derivatives = dict(a=da, b=db, embedding=.4*df, a_left=-.5*da,
                       a_right=.3*da, b_left=.2*db, b_right=.6*db)
    roots = list(t.parameters())+[x]
    for evaluate in (lambda: p.step(w, req), lambda: p.batch(w, [req])[0], lambda: p.batch_grad(w, [req])[0]):
        result = evaluate().value
        equivalent(result, x.new_tensor([.6, .8+.4*c*h*s]))
        grads = torch.autograd.grad(result[1], roots, retain_graph=True)
        for (name, parameter), grad in zip(t.named_parameters(), grads):
            equivalent(grad, torch.full_like(parameter, derivatives[name]))
        equivalent(grads[-1], x.new_tensor([0., 1+.4*df*a/(1+.8**2)**1.5]))
        equivalent(torch.autograd.grad(result[0], x)[0], x.new_tensor([2., 0.]))


def test_default_region_profile_keeps_published_identity(dtype):
    from tidegraph.weighted.config import identity
    b = fixture(dtype)
    p = b.model.nodes[0].aggregate_program
    assert b.graph.nodes[0].aggregation == identity("weighted-aggregate-v1", b.source_classes, .4, 3, 1e-5)
    assert p.source_binding == SourceBinding()
