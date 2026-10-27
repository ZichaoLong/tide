"""Architecture §2: independent fiber roots, absent transforms, shared owners.

Literal mean derivatives and directional differences anchor the local formula;
scalar autograd checks the structural None/zero contract of the batch VJP.
"""
import pytest
import torch
from tidegraph import Atom, Graph, Node, Region
from tidegraph.aggregate import request
from tidegraph.compare import equivalent
from tidegraph.source_class import SourceClasses
from tidegraph.weighted import KVConfig, bind
from isolated_cases import vjp


def fixture(dtype, alpha):
    graph = Graph((Node(0),), (), (Region(1),), (0, 0, 0, 0), (0,))
    b = bind(graph, memories=KVConfig(width=2, summary_width=2),
             source_classes=SourceClasses((), (0, 0, 1, 2)), alpha_agg=alpha,
             source_hidden_width=3, dtype=dtype)
    # Same-class duplicate atoms, different fibers, absent class 2, zero totals,
    # positive total with a zero-mass class, and mixed requires_grad metadata.
    fibers = (((0, .2), (1, .4), (2, .3)), ((0, 0.), (2, -.2)),
              ((0, .7), (2, 0.)), ((0, .3), (1, .6), (2, .2)),
              ((0, .1), (1, .2), (2, .3)))
    requests, roots = [], {}
    for i, fiber in enumerate(fibers):
        atoms = []
        for j, (port, mass) in enumerate(fiber):
            value = torch.tensor([mass, .2+.04*i, -.3+.07*j], dtype=dtype, requires_grad=i != 4 or j == 0)
            atoms.append(Atom(i, 0, i, 0, port, 0, value))
            if value.requires_grad:
                roots[f"input.{i}.{j}"] = value
        requests.append(request(b.graph, b.model, atoms))
    w = b.model.nodes[0]
    roots.update(dict(w.aggregate_program.named_parameters(prefix="transform")))
    roots.update({f"unused.scale.{i}": s.requires_grad_() for i, s in enumerate(b.model.input_scale)})
    return w, requests, roots


@pytest.mark.parametrize("alpha", [0., .4])
def test_fiber_roots_zero_branches_and_aliases(dtype, alpha, monkeypatch):
    w, requests, roots = fixture(dtype, alpha)
    p = w.aggregate_program
    expected = [p.step(w, r).value for r in requests]
    def forbidden(*a, **kw):
        raise AssertionError("batch VJP replayed scalar Aggregate")
    monkeypatch.setattr(p, "step", forbidden)
    actual = [r.value for r in p.batch_grad(w, requests)]
    equivalent(expected, actual)
    for a, b in zip(expected, actual):
        for root in (lambda t: t, lambda t: t[1:], lambda t: t*0):
            equivalent(vjp(root(a), roots), vjp(root(b), roots))
    equivalent(vjp(sum(t.square().sum() for t in expected), roots),
               vjp(sum(t.square().sum() for t in actual), roots))
    grads = vjp(actual[1], roots)
    assert all(value is None for name, value in grads.items() if name.startswith("transform"))
    assert grads["input.0.0"] is None and grads["input.1.0"] is not None
    assert p.batch_grad(w, []) == []


def test_literal_mean_jacobian_and_repeated_input(dtype):
    w, requests, _ = fixture(dtype, 0.)
    p = w.aggregate_program
    r = requests[0]
    values = [s.atom.value for s in r.sources]
    result = p.batch_grad(w, [r])[0].value
    bar = result.new_tensor([.3, .8, -.2])
    grads = torch.autograd.grad(result, values, bar)
    total = sum(float(v[0].detach()) for v in values)
    mean = sum(float(v[0].detach())*v[1:].detach() for v in values)/total
    for value, grad in zip(values, grads):
        expected = torch.cat(((bar[0]+(bar[1:]*(value[1:].detach()-mean)).sum()/total).reshape(1),
                              bar[1:]*value[0].detach()/total))
        equivalent(expected, grad)
    # Both physical atoms use the same public Tensor. Their cotangents add.
    from dataclasses import replace
    repeated = replace(r, sources=tuple(replace(s, atom=replace(s.atom, value=values[0])) for s in r.sources))
    result = p.batch_grad(w, [repeated])[0].value
    grad, = torch.autograd.grad(result, values[0], torch.ones_like(result))
    equivalent(grad, grad.new_tensor([3., 1., 1.]))


def test_aggregate_transform_directional_difference():
    w, requests, roots = fixture(torch.float64, .4)
    p, r = w.aggregate_program, requests[0]
    x = roots["input.0.0"]
    direction = x.new_tensor([.13, -.22, .31])
    output = p.batch_grad(w, [r])[0].value
    bar = x.new_tensor([.6, -.4, .9])
    gradient, = torch.autograd.grad(output, x, bar)
    delta = 1e-5
    from dataclasses import replace
    def at_shift(sign):
        source = r.sources[0]
        shifted = replace(source, atom=replace(source.atom, value=x.detach()+sign*delta*direction))
        shifted_request = replace(r, sources=(shifted, *r.sources[1:]))
        # The scalar formula is independently evaluated at the perturbed input.
        return (p.step(w, shifted_request).value*bar).sum()
    numeric = (at_shift(1)-at_shift(-1))/(2*delta)
    torch.testing.assert_close((gradient*direction).sum(), numeric, atol=1e-9, rtol=1e-7)


def test_aggregate_versions_and_first_order(dtype):
    w, requests, roots = fixture(dtype, .4)
    output = w.aggregate_program.batch_grad(w, requests)[0].value
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(output.sum(), roots["input.0.0"], create_graph=True)
    output = w.aggregate_program.batch_grad(w, requests)[0].value
    with torch.no_grad():
        w.aggregate_program.transforms["0"].embedding.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
