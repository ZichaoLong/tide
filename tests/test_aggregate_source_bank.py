"""Architecture §2: source ownership, exact incidence and independent roots."""
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.aggregate_layout import SourceIncidence
from tidegraph.weighted.aggregate_tensor import assemble_many, _rows
from tidegraph.weighted.column_aggregate import assemble
from tidegraph.weighted.column_sources import source_blocks
from tidegraph.weighted.message_columns import MessageColumns
from isolated_cases import vjp
from test_aggregate_tensor import fixture, literal
from test_column_sources import fixture as source_fixture, literal as source_formula


def gradients(loss, roots):
    return dict(zip(roots, torch.autograd.grad(loss, tuple(roots.values()), retain_graph=True, allow_unused=True)))


def precise_literal(entries):
    # Literal quotient in FP64 on the exact stored input values. The root cast
    # remains in the oracle graph. FP32 quotient autograd can subtract two
    # large equal terms at invariant mass roots, obscuring layout correctness.
    result = []
    for sources, values in entries:
        high = [(replace(p, strength=p.strength.double(), content=p.content.double()), c) for p, c in sources]
        rows, mass, value = literal((high, [z.double() for z in values]))
        result.append((rows, mass.to(values[0].dtype), value.to(values[0].dtype)))
    return result


@pytest.mark.parametrize("adapted", [False, True])
def test_source_owners_are_shared_without_publishing_physical_use_tensors(dtype, adapted):
    requests, roots = source_fixture(dtype, adapted)
    requests = [requests[0], requests[0], requests[2], requests[1], requests[0]]
    block = source_blocks(requests, max_workspace_bytes=2**24)
    assert block.owners == (0, 0, 1, 2, 0) and len(block.values) == 3
    got = [block.values[i] for i in block.owners]
    assert got[0] is got[1] is got[4] and got[0] is not got[2]
    expected = [source_formula(r) for r in requests]
    equivalent(got, expected)
    loss = lambda x: x[0].square().sum()+.3*x[1].sum()-.2*x[4].sum()+x[3].sin().sum()
    equivalent(vjp(loss(got), roots), vjp(loss(expected), roots))
    projected = block.independent_uses()
    assert projected[0] is not projected[1]
    assert projected[0].data_ptr() == projected[1].data_ptr()
    equivalent(vjp(projected[1].sum()*0, roots), vjp(expected[1].sum()*0, roots))


def test_unique_bank_storage_is_independent_of_receiver_multiplicity(dtype):
    entries, roots = fixture(dtype)
    entries = [entries[0]]*8+[entries[1], entries[2]]
    plan = SourceIncidence.build([(a, z, _rows(a)) for a, z in entries])
    assert len(plan.sources) == 4 and len(plan.receivers) == 10
    assert plan.incidence[0] == (2, 1, 0, 0)
    index, inverse, counts, publish, singleton = plan.tensors(torch.device("cpu"))
    assert index.shape == (4, 6) and counts.shape == (10, 4)
    assert index.dtype == inverse.dtype == counts.dtype == torch.int64
    assert all(x.dtype == torch.int64 for x in publish)
    assert singleton.dtype == torch.bool
    actual = assemble_many(entries, 2**26, assemble)
    saved = actual[0][2].grad_fn.saved_tensors
    # The shared value bank is S,R,D. Output means G,R,D remain necessary;
    # there is no receiver x source payload or retained per-physical-use graph.
    assert any(tuple(t.shape) == (4, 6, 3) for t in saved)
    assert all(t.ndim <= 3 for t in saved)
    expected = [literal(entry) for entry in entries]
    objective = lambda x: sum((i+.3)*row[2].square().sum() for i, row in enumerate(x[:8]))+x[9][1].sum()
    equivalent(actual, expected)
    equivalent(gradients(objective(actual), roots), gradients(objective(precise_literal(entries)), roots))


def test_row_domains_and_equal_independent_inputs_do_not_alias(dtype):
    mass = torch.tensor([.2, .7], dtype=dtype, requires_grad=True)
    z = torch.tensor([[.3, -.1, .8], [.6, .9, -.5]], dtype=dtype, requires_grad=True)
    a = MessageColumns(torch.tensor([0, 2]), mass, z)
    b = replace(a, rows=torch.tensor([1, 3]))
    c = replace(a, strength=mass.detach().clone().requires_grad_(), content=z.detach().clone().requires_grad_())
    groups = [[a, b, a], [c], [b]]
    entries = [([(p, 0) for p in g], [p.content for p in g]) for g in groups]
    plan = SourceIncidence.build([(a, z, _rows(a)) for a, z in entries])
    assert len(plan.sources) == 3 and plan.incidence[0] == (2, 1, 0)
    roots = dict(mass=mass, value=z, independent_mass=c.strength, independent_value=c.content)
    actual = assemble_many(entries, 2**24, assemble)
    expected = [literal(entry) for entry in entries]
    equivalent(actual, expected)
    for index in (0, 1, 2):
        equivalent(gradients(actual[index][2].sum(), roots), gradients(precise_literal(entries)[index][2].sum(), roots))


@pytest.mark.parametrize("budget", [1, 10000, 2**24])
def test_budget_group_boundaries_keep_all_roots_and_ordered_uses(dtype, budget):
    entries, roots = fixture(dtype)
    entries = [entries[0], entries[1], entries[0], entries[2], entries[0]]
    actual = assemble_many(entries, budget, assemble)
    expected = [literal(entry) for entry in entries]
    equivalent(actual, expected)
    objective = lambda x: x[0][2].square().sum()+.7*x[2][2].sum()+x[4][1].sum()*0
    first = gradients(objective(actual), roots)
    equivalent(first, gradients(objective(precise_literal(entries)), roots))
    second = gradients(objective(actual), roots)
    assert all(a is None and second[k] is None or a is not None and torch.equal(a, second[k]) for k, a in first.items())


def test_centered_mass_vjp_keeps_large_common_offset_invariance():
    z = torch.full((3, 3), 2**23, dtype=torch.float32, requires_grad=True)
    a = MessageColumns(torch.arange(3), torch.ones(3, requires_grad=True), z)
    b = replace(a, strength=torch.full((3,), 3., requires_grad=True))
    entries = [([(a, 0), (b, 0)], [z, z]), ([(a, 0)], [z])]
    actual = assemble_many(entries, 2**24, assemble)
    gm = torch.autograd.grad((actual[0][2]*torch.tensor([.3, -.7, .9])).sum(), (a.strength, b.strength))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gm)
