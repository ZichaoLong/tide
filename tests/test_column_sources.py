"""0770593 architecture §2.2: literal formulas, root support and exact aliases."""
from types import SimpleNamespace
import pytest
import torch
from tidegraph.weighted.column_sources import source_columns, workspace_bytes
from tidegraph.weighted.column_aggregate import aggregate_columns_many
from tidegraph.weighted.message_columns import MessageColumns
from tidegraph.weighted.source_binding import SourceTransform
from tidegraph.weighted.packing_observation import observe
from tidegraph.compare import equivalent
from isolated_cases import vjp


def evaluate(requests, implementation, budget=2**25):
    if implementation == "python":
        return source_columns(requests, max_workspace_bytes=budget)
    from tidegraph.native_loader import load_native
    core = load_native()
    return core.weighted_source_columns(requests, budget)


def literal(request):
    z, parameters, alpha, epsilon = request
    if not alpha:
        return z
    a, b, e = parameters[:3]
    if len(parameters) == 7:
        a = a+parameters[3]@parameters[4]; b = b+parameters[5]@parameters[6]
    # Independent equation, not a production norm/source/linear helper.
    x = z.float() if z.dtype == torch.float16 else z
    u = (x/torch.sqrt(epsilon+x.square().mean(-1, keepdim=True))).to(z.dtype)
    h = u@a.t()+e
    return z+alpha*((h*torch.sigmoid(h))@b.t())


def fixture(dtype, adapted=False):
    generator = torch.Generator().manual_seed(93)
    def leaf(shape):
        return (torch.randn(shape, generator=generator, dtype=dtype)*.2).requires_grad_()
    rows = []
    for _ in range(4):
        p = [leaf((3, 3)), leaf((3, 3)), leaf((3,))]
        if adapted:
            p += [leaf((3, 2)), leaf((2, 3)), leaf((3, 2)), leaf((2, 3))]
        rows.append((leaf((4, 3)), tuple(p), .4, .1))
    # Exact base and low-rank aliases within and across mathematical slots.
    a = list(rows[0][1]); a[1] = a[0]
    if adapted:
        a[5] = a[3]; a[6] = a[4]
    rows[0] = rows[0][0], tuple(a), .4, .1
    b = list(rows[1][1]); b[0] = a[0]
    rows[1] = rows[1][0], tuple(b), .4, .1
    # Same values but independent roots must not enter the exact cache.
    rows[2] = rows[0][0].detach().clone().requires_grad_(), tuple(p.detach().clone().requires_grad_() for p in a), .4, .1
    roots = {}
    for i, (z, weights, *_rest) in enumerate(rows):
        for j, x in enumerate((z, *weights)):
            if all(x is not v for v in roots.values()):
                roots[f"{i}.{j}"] = x
    return rows, roots


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("adapted", [False, True])
def test_literal_formula_independent_roots_aliases_and_connected_zero(dtype, implementation, adapted):
    rows, roots = fixture(dtype, adapted)
    actual = evaluate(rows, implementation)
    expected = [literal(r) for r in rows]
    equivalent(actual, expected)
    bar = torch.linspace(-.7, .6, actual[0].numel(), dtype=dtype).reshape_as(actual[0])
    for zero in (False, True):
        loss = lambda y: (y[0]*bar).sum()+(y[1]*bar.flip(0)).sum()
        a, e = vjp(loss(actual), roots, zero=zero), vjp(loss(expected), roots, zero=zero)
        equivalent(a, e)
        assert a["2.0"] is None and a["3.0"] is None
    # An observed literal-zero root is connected; an unobserved output is not.
    a = vjp(actual[2].sum()*0, roots)
    assert a["2.0"] is not None and not a["2.0"].count_nonzero()
    assert a["3.0"] is None


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_exact_reuse_multiplicity_frozen_and_different_shapes(implementation):
    rows, roots = fixture(torch.float64)
    rows = [rows[0], rows[0], rows[1], rows[2], (rows[3][0][:2], rows[3][1], .7, .2)]
    frozen = (rows[0][0].detach(), tuple(p.detach() for p in rows[0][1]), .4, .1)
    actual = evaluate([*rows, frozen], implementation)
    assert actual[0].data_ptr() == actual[1].data_ptr()
    assert not actual[-1].requires_grad
    expected = [literal(r) for r in rows]
    loss = lambda y: sum((i+1)*x.square().sum() for i, x in enumerate(y[:5]))
    equivalent(actual[:5], expected)
    equivalent(vjp(loss(actual), roots), vjp(loss(expected), roots))


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("slot", [0, 1, 3, 4, 7])
def test_original_version_guards_and_first_order_only(implementation, slot):
    rows, roots = fixture(torch.float64, True)
    out = evaluate(rows, implementation)[0]
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(out.sum(), rows[0][0], create_graph=True, retain_graph=True)
    with torch.no_grad():
        (rows[0][0], *rows[0][1])[slot].add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(out.sum(), rows[0][0])


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("budget", [1, 25000, 2**25])
def test_bounded_batches_match_formula(implementation, budget):
    rows, roots = fixture(torch.float64, True)
    got = evaluate(rows, implementation, budget)
    expected = [literal(r) for r in rows]
    equivalent(got, expected)
    equivalent(vjp(sum(x.sum() for x in got), roots), vjp(sum(x.sum() for x in expected), roots))


def test_actual_grouping_and_exact_parameter_slot_identity():
    rows, _ = fixture(torch.float64)
    with observe() as counter:
        evaluate([rows[0], rows[0], rows[1], rows[2]], "python")
    m = counter.metrics()
    assert m["packing/bucket/source_columns/calls"] == 1
    assert m["packing/bucket/source_columns/rows"] == 3
    assert m["packing/bucket/source_exact_reuse/rows"] == 2
    budget = workspace_bytes(*rows[0][:2])
    with observe() as counter:
        evaluate(rows, "python", budget)
    assert counter.metrics()["packing/bucket/source_columns/calls"] == 4


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_half_payload_and_zero_alpha_binding(implementation):
    rows, roots = fixture(torch.float16, True)
    got = evaluate(rows, implementation)
    expected = [literal(r) for r in rows]
    equivalent(got, expected)
    equivalent(vjp(got[0].sum(), roots), vjp(expected[0].sum(), roots))
    z = rows[0][0]
    same = evaluate([(z, (), 0., .1), (z, (), 0., .1)], implementation)
    assert same[0].data_ptr() == z.data_ptr()
    equivalent(torch.autograd.grad(same[0].sum()+2*same[1].sum(), z)[0], torch.full_like(z, 3))


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_reused_source_adjoint_order_is_the_declared_use_order(implementation):
    z = torch.ones(1, 2, dtype=torch.float32, requires_grad=True)
    out = evaluate([(z, (), 0., .1)]*3, implementation)
    loss = out[2].sum()-1e8*out[1].sum()+1e8*out[0].sum()
    assert torch.equal(torch.autograd.grad(loss, z)[0], torch.ones_like(z))


def test_regional_aggregate_cache_keeps_distinct_atoms_and_empty_zero_roots():
    dtype = torch.float64
    rows, _ = fixture(dtype)
    transform = SourceTransform(*(torch.nn.Parameter(p) for p in rows[0][1]))
    program = SimpleNamespace(alpha=.4, epsilon=.1, transforms={"0": transform, "1": transform})
    z = rows[0][0]; m = torch.tensor([.2, .4, .7, .1], dtype=dtype, requires_grad=True)
    p = MessageColumns(torch.tensor([0, 1, 3, 4]), m, z)
    independent = MessageColumns(p.rows, m.detach().clone().requires_grad_(), z.detach().clone().requires_grad_())
    zero = MessageColumns(torch.tensor([0, 2, 4, 5]), torch.zeros(4, dtype=dtype, requires_grad=True),
                          z.detach().clone().requires_grad_())
    source_lists = [[(p, 0), (p, 1)], [(p, 0), (p, 1)], [(p, 0), (independent, 0)], [(zero, 0)], []]
    with observe() as counter:
        got = aggregate_columns_many([(program, sources) for sources in source_lists])
    assert got[-1] is None
    assert got[0][1].data_ptr() == got[1][1].data_ptr()
    assert got[0][1].data_ptr() != got[2][1].data_ptr()
    equivalent(got[0][1], 2*m)
    equivalent(got[0][2], literal((z, tuple(transform.parameters()), .4, .1)))
    assert counter.metrics()["packing/bucket/aggregate_exact_reuse/rows"] == 4
    zero_loss = got[3][1].sum()+got[3][2].sum()
    grads = torch.autograd.grad(zero_loss, [zero.strength, zero.content, *transform.parameters()], allow_unused=True)
    assert grads[0] is not None and all(g is None for g in grads[1:])
    # Reordered source slots with the same deduplicated Parameter list must not
    # collide; named slots, not Module.parameters(), define the cache identity.
    a = torch.nn.Parameter(torch.eye(3, dtype=dtype)*.2)
    b = torch.nn.Parameter(torch.arange(9, dtype=dtype).reshape(3, 3)*.03)
    e = transform.embedding
    first = SimpleNamespace(alpha=.4, epsilon=.1, transforms={"0": SourceTransform(a, b, e, (a, a, b, b))})
    other = SimpleNamespace(alpha=.4, epsilon=.1, transforms={"0": SourceTransform(a, b, e, (b, a, a, b))})
    assert list(map(id, first.transforms["0"].parameters())) == list(map(id, other.transforms["0"].parameters()))
    different = aggregate_columns_many([(first, [(p, 0)]), (other, [(p, 0)])])
    equivalent(different[0][2], literal((z, (a, b, e, a, a, b, b), .4, .1)))
    equivalent(different[1][2], literal((z, (a, b, e, b, a, a, b), .4, .1)))
    assert different[0][2].data_ptr() != different[1][2].data_ptr()
