"""Architecture §2 physical means: independent formulas and regional roots."""
import pytest
import torch
from tidegraph.weighted.aggregate_tensor import assemble_many
from tidegraph.weighted.column_aggregate import assemble
from tidegraph.weighted.message_columns import MessageColumns
from tidegraph.weighted.packing_observation import observe
from tidegraph.compare import equivalent
from isolated_cases import vjp


def fixture(dtype):
    roots = {}
    def message(name, rows, masses):
        m = torch.tensor(masses, dtype=dtype, requires_grad=True)
        z = torch.sin(torch.arange(len(rows)*3, dtype=dtype).reshape(-1, 3)*.3).requires_grad_()
        roots[name+".mass"], roots[name+".value"] = m, z
        return MessageColumns(torch.tensor(rows), m, z)
    a = message("a", [0, 2, 5], [.3, 0., .7])
    b = message("b", [1, 2], [.5, .4])
    c = message("c", [0, 2], [0., 0.])
    d = message("d", [3, 7], [.2, .6])
    # Duplicate physical atoms count twice; equal class labels do not merge.
    groups = [[(a, 0), (b, 0), (a, 0)], [(d, 1)], [(c, 2)]]
    return [(g, [p.content for p, _ in g]) for g in groups], roots


def literal(entry):
    sources, transformed = entry
    rows = sorted({int(r) for p, _ in sources for r in p.rows})
    masses, values = [], []
    for r in rows:
        atoms = [(p.strength[(p.rows == r).nonzero()[0, 0]], z[(p.rows == r).nonzero()[0, 0]])
                 for (p, _), z in zip(sources, transformed) if r in p.rows.tolist()]
        mass = sum(m.clamp_min(0) for m, _ in atoms)
        masses.append(mass)
        values.append(sum(m.clamp_min(0)*z for m, z in atoms)/mass if mass.detach() > 0
                      else transformed[0].new_zeros(3))
    return torch.tensor(rows), torch.stack(masses), torch.stack(values)


@pytest.mark.parametrize("field", ["mass", "value", "both", "zero"])
def test_independent_literal_fields_ragged_duplicates_and_none(dtype, field):
    entries, roots = fixture(dtype)
    actual = assemble_many(entries, 2**24, assemble)
    expected = [literal(entry) for entry in entries]
    equivalent(actual, expected)
    def loss(rows):
        m, u = rows[0][1:]
        if field == "mass":
            return (m*torch.arange(1, 1+len(m), dtype=dtype)).sum()
        value = (u*torch.tensor([.3, -.7, .9], dtype=dtype)).sum()
        return value+(m.sum() if field == "both" else 0) if field != "zero" else value*0
    a, e = vjp(loss(actual), roots), vjp(loss(expected), roots)
    equivalent(a, e)
    assert a["d.mass"] is None and a["c.value"] is None
    if field == "mass":
        assert a["a.value"] is None and a["b.value"] is None
    if field == "zero":
        assert a["a.value"] is not None and not a["a.value"].count_nonzero()
    assert not actual[2][2].requires_grad


def test_singleton_centered_mass_transpose_and_zero_fiber_mass_root():
    entries, roots = fixture(torch.float64)
    actual = assemble_many(entries, 2**24, assemble)
    names = tuple(roots)
    gradients = torch.autograd.grad(actual[1][2].sum(), tuple(roots.values()), allow_unused=True, retain_graph=True)
    g = dict(zip(names, gradients))
    assert g["d.mass"] is not None and g["d.mass"].eq(0).all()
    assert g["d.value"].eq(1).all() and g["a.value"] is None
    gradients = torch.autograd.grad(actual[2][1].sum(), tuple(roots.values()), allow_unused=True)
    g = dict(zip(names, gradients))
    assert g["c.mass"].eq(1).all() and g["c.value"] is None


def test_actual_batch_and_budget_fallback():
    entries, _ = fixture(torch.float64)
    with observe() as counter:
        got = assemble_many(entries, 2**24, lambda *_: pytest.fail("unexpected scalar assembly"))
    assert counter.metrics()["packing/bucket/aggregate_tensor/calls"] == 1
    assert counter.metrics()["packing/bucket/aggregate_tensor/rows"] == 3
    with observe() as counter:
        small = assemble_many(entries, 1, assemble)
    assert "packing/bucket/aggregate_tensor/calls" not in counter.metrics()
    equivalent(got, small)


@pytest.mark.parametrize("root", ["a.mass", "a.value", "c.value"])
def test_original_versions_and_first_order(root):
    entries, roots = fixture(torch.float64)
    actual = assemble_many(entries, 2**24, assemble)
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(actual[0][2].sum(), roots["a.mass"], create_graph=True, retain_graph=True)
    with torch.no_grad():
        roots[root].add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(actual[0][2].sum(), roots["a.mass"])


def test_frozen_no_grad_and_smooth_directional_derivative():
    entries, roots = fixture(torch.float64)
    # Frozen fields keep metadata ownership but cannot gain roots by batching.
    roots["d.value"].requires_grad_(False); roots["d.mass"].requires_grad_(False)
    got = assemble_many(entries, 2**24, assemble)
    assert not got[1][1].requires_grad and not got[1][2].requires_grad
    with torch.no_grad():
        expected = assemble_many(entries, 2**24, assemble)
    equivalent(got, expected)
    assert all(not x.requires_grad for row in expected for x in row)
    mass, value = roots["a.mass"], roots["a.value"]
    dm, dv = torch.autograd.grad(got[0][2].square().sum(), (mass, value))
    md = torch.tensor([.03, 0., -.02], dtype=mass.dtype)
    vd = torch.full_like(value, .02)
    epsilon = 1e-5
    original_m, original_v = mass.detach().clone(), value.detach().clone()
    samples = []
    with torch.no_grad():
        for sign in (1, -1):
            mass.copy_(original_m+sign*epsilon*md); value.copy_(original_v+sign*epsilon*vd)
            samples.append(assemble_many(entries, 2**24, assemble)[0][2].square().sum())
    equivalent((dm*md).sum()+(dv*vd).sum(), (samples[0]-samples[1])/(2*epsilon))
