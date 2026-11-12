"""Independent first-order roots for typed tensor blocks, without graph replay.

Packed arithmetic is checked against scalar dot formulas and explicit linear
mapping matrices. Numerical zero never decides whether a root is connected.
"""
import pytest
import torch

from tidegraph.autograd_census import Census
from tidegraph.compare import equivalent
from tidegraph.weighted.tensor_kernel import Role, execute, mapped_role


FIELDS = ("left", "right", "mixed", "identity", "constant")


class Formula:
    fields = FIELDS

    def __init__(self):
        self.calls = []

    def __call__(self, roles):
        self.calls.append((torch.is_grad_enabled(), any(t.requires_grad for t in roles.values())))
        x, y, p, q = (roles[name] for name in ("x", "y", "p", "q"))
        left = (x*p[:, None, None]).sum(-1)
        right = (y*q[:, None, None]).sum(-1)
        return dict(left=left, right=right, mixed=left+2*right, identity=x,
                    constant=x.new_full(x.shape[:-1], .7))


def material(dtype):
    def leaf(shape, offset):
        n = 1
        for s in shape:
            n *= s
        return (torch.arange(n, dtype=dtype).reshape(shape)*.03+offset).requires_grad_()
    shared = leaf((2,), .2)
    equal = shared.detach().clone().requires_grad_()
    roles = dict(x=Role(tuple(leaf((2, 2, 2), .1+i*.1) for i in range(3))),
                 y=Role(tuple(leaf((2, 2, 2), -.2+i*.1) for i in range(3))),
                 p=Role((shared, equal, leaf((2,), .4))),
                 q=Role((shared, shared, leaf((2,), -.3))),
                 unused=Role(tuple(leaf((2,), .6+i*.1) for i in range(3))))
    indices = (torch.tensor([3, 0, 2, 1]), torch.tensor([1, 3]), torch.tensor([0, 2]))
    return roles, indices


def roots(roles):
    result = {}
    for name, role in roles.items():
        for node, value in enumerate(role.values):
            if value.requires_grad and all(value is not old for old in result.values()):
                result[f"{name}.{node}"] = value
    return result


def literal(roles, indices):
    """Separate scalar formulas, with no candidate evaluator or packing helper."""
    result = []
    for node, rows in enumerate(indices):
        x, y, p, q = (roles[name].values[node] for name in ("x", "y", "p", "q"))
        left, right = [], []
        for row in rows.tolist():
            b, t = divmod(row, x.shape[1])
            left.append(sum(x[b, t, d]*p[d] for d in range(x.shape[-1])))
            right.append(sum(y[b, t, d]*q[d] for d in range(y.shape[-1])))
        left, right = torch.stack(left), torch.stack(right)
        identity = torch.stack([x[row//x.shape[1], row % x.shape[1]] for row in rows.tolist()])
        result.append(dict(left=left, right=right, mixed=left+2*right, identity=identity,
                           constant=x.new_full((len(rows),), .7)))
    return result


def objective(values, selected):
    result = 0
    for node, field, scale in selected:
        x = values[node][field]
        bar = torch.linspace(-.3, .7, x.numel(), dtype=x.dtype).reshape_as(x)
        result = result+(x*bar).sum()*scale
    return result


def vjp(loss, variables):
    return dict(zip(variables, torch.autograd.grad(loss, tuple(variables.values()),
                                                 allow_unused=True, retain_graph=True)))


@pytest.mark.parametrize("selected", [
    ((0, "left", 1.), (1, "right", 0.)),
    ((0, "mixed", .3), (1, "left", .7)),
    ((1, "right", 0.),),
    ((0, "identity", 1.), (2, "right", .2)),
])
def test_node_field_roots_aliases_equal_values_and_connected_zero(dtype, selected):
    roles, indices = material(dtype)
    formula = Formula()
    actual = execute(roles, indices, formula)
    expected = literal(roles, indices)
    equivalent(actual, expected)
    assert len(formula.calls) == 1
    assert roles["p"].values[0] is roles["q"].values[0] is roles["q"].values[1]
    assert roles["p"].values[1] is not roles["p"].values[0]
    assert torch.equal(roles["p"].values[1], roles["p"].values[0])
    variables = roots(roles)
    actual_grad = vjp(objective(actual, selected), variables)
    expected_grad = vjp(objective(expected, selected), variables)
    equivalent(actual_grad, expected_grad)
    assert all(actual_grad[f"unused.{i}"] is None for i in range(3))
    assert all(not row["constant"].requires_grad for row in actual)
    if selected == ((0, "left", 1.), (1, "right", 0.)):
        assert actual_grad["x.1"] is actual_grad["x.2"] is actual_grad["y.2"] is None
        assert actual_grad["p.1"] is None  # Equal values do not create aliases.
        assert actual_grad["y.1"] is not None and actual_grad["y.1"].count_nonzero() == 0
    if selected == ((1, "right", 0.),):
        assert actual_grad["p.0"] is not None and actual_grad["p.0"].count_nonzero() == 0


def test_frozen_node_fields_keep_their_own_gradient_identity(dtype):
    roles, indices = material(dtype)
    # Detach individual slots, leaving other nodes/roles differentiable.
    for name, frozen in (("x", {1, 2}), ("y", {0, 1}), ("p", {0, 1}), ("q", {0, 1, 2})):
        roles[name] = Role(tuple(x.detach() if i in frozen else x for i, x in enumerate(roles[name].values)))
    actual, expected = execute(roles, indices, Formula()), literal(roles, indices)
    for a, e in zip(actual, expected):
        for field in FIELDS:
            equivalent(a[field], e[field])
            assert a[field].requires_grad == e[field].requires_grad
    assert not any(x.requires_grad for x in actual[1].values())
    variables = roots(roles)
    selected = ((0, "left", 1.), (2, "left", .3), (2, "right", .7))
    equivalent(vjp(objective(actual, selected), variables), vjp(objective(expected, selected), variables))


class MappedFormula:
    fields = ("value",)

    def __call__(self, roles):
        x = roles["x"]
        return dict(value=x.square()+2*x)


def test_ragged_padding_duplicate_input_mapping_and_empty_root_transpose(dtype):
    values = tuple((torch.arange(n*2, dtype=dtype).reshape(n, 2)*.1+.2).requires_grad_()
                   for n in (3, 1, 0))
    mappings = (torch.tensor([[0, 2, 3], [1, 0, 3]]),
                torch.tensor([[0, 1, 1], [1, 0, 1]]), torch.zeros((2, 3), dtype=torch.int64))
    indices = (torch.tensor([5, 0, 3, 1, 4, 2]), torch.tensor([0, 4, 1]), torch.tensor([2, 5]))
    actual = execute({"x": mapped_role(values, mappings)}, indices, MappedFormula())
    expected, matrices = [], []
    for value, mapping, index in zip(values, mappings, indices):
        # An explicit linear map, including its zero-column empty domain,
        # independently defines both padding and repeated-row contributions.
        matrix = value.new_zeros((mapping.numel(), len(value)))
        for output, source in enumerate(mapping.flatten().tolist()):
            if source < len(value):
                matrix[output, source] = 1
        padded = matrix@value
        expected.append(dict(value=(padded.square()+2*padded)[index]))
        matrices.append(matrix)
    equivalent(actual, expected)
    variables = {str(i): x for i, x in enumerate(values)}
    selected = tuple((i, "value", i+.4) for i in range(3))
    ag = vjp(objective(actual, selected), variables)
    eg = vjp(objective(expected, selected), variables)
    equivalent(ag, eg)
    assert ag["2"] is not None and ag["2"].shape == (0, 2)
    # Direct M^T contraction verifies repeated input indices sum all uses.
    for i, (value, matrix, index) in enumerate(zip(values, matrices, indices)):
        output = expected[i]["value"]
        bar = torch.linspace(-.3, .7, output.numel(), dtype=dtype).reshape_as(output)*(i+.4)
        literal_bar = matrix[index].t()@(bar*(2*(matrix[index]@value.detach())+2))
        equivalent(ag[str(i)], literal_bar)


@pytest.mark.parametrize("role,node", [("x", 0), ("unused", 2)])
def test_original_versions_include_unobserved_roots(dtype, role, node):
    roles, indices = material(dtype)
    result = execute(roles, indices, Formula())
    with torch.no_grad():
        roles[role].values[node].add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        result[0]["left"].sum().backward()


def test_retained_graph_repeated_and_different_output_roots(dtype):
    roles, indices = material(dtype)
    actual, expected = execute(roles, indices, Formula()), literal(roles, indices)
    variables = roots(roles)
    selections = (((0, "mixed", 1.),), ((1, "identity", 0.),),
                  ((2, "right", -.4),), ((0, "mixed", 1.),))
    first = None
    for selected in selections:
        got = vjp(objective(actual, selected), variables)
        equivalent(got, vjp(objective(expected, selected), variables))
        if first is None:
            first = got
    equivalent(got, first, atol=0, rtol=0)


def test_higher_order_rejected(dtype):
    roles, indices = material(dtype)
    result = execute(roles, indices, Formula())
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(result[0]["left"].sum(), roles["x"].values[0], create_graph=True)


def test_no_grad_evaluates_once_without_private_autograd(dtype):
    roles, indices = material(dtype)
    formula = Formula()
    with torch.no_grad():
        actual = execute(roles, indices, formula)
        expected = literal(roles, indices)
    equivalent(actual, expected)
    assert formula.calls == [(False, False)]
    assert all(not x.requires_grad and x.grad_fn is None for row in actual for x in row.values())


class SharedFormula:
    fields = ("linear", "other", "scaled")

    def __init__(self):
        self.shared = set()
        self.shapes = None

    def __call__(self, roles):
        x, p, q, scale = (roles[name] for name in ("x", "p", "q", "scale"))
        self.shapes = tuple(tuple(t.shape) for t in (x, p, q, scale))
        linear, other = x@p, x@q
        return dict(linear=linear, other=other, scaled=(linear+2*other)*scale)


def shared_material(dtype, count=3, batch=2, length=2):
    x = tuple((torch.arange(batch*length*2, dtype=dtype).reshape(batch, length, 2)*.07+i*.1)
              .requires_grad_() for i in range(count))
    parameter = torch.tensor([[.2, -.3], [.4, .7]], dtype=dtype, requires_grad=True)
    scale = torch.tensor(.6, dtype=dtype, requires_grad=True)
    roles = dict(x=Role(x), p=Role((parameter,)*count, shared=True),
                 q=Role((parameter,)*count, shared=True), scale=Role((scale,)*count, shared=True))
    indices = tuple(torch.arange(batch*length-1, -1, -1) for _ in range(count))
    return roles, indices


def shared_literal(roles, indices):
    result = []
    for node, rows in enumerate(indices):
        x, p, q, scale = (roles[name].values[node] for name in ("x", "p", "q", "scale"))
        columns = []
        for matrix in (p, q):
            columns.append(torch.stack([torch.stack([
                sum(x[row//x.shape[1], row % x.shape[1], d]*matrix[d, out] for d in range(2))
                for out in range(2)]) for row in rows.tolist()]))
        left, right = columns
        result.append(dict(linear=left, other=right, scaled=(left+2*right)*scale))
    return result


@pytest.mark.parametrize("selected", [
    ((2, "scaled", 1.),),
    ((2, "scaled", 0.),),
    ((1, "linear", .3), (2, "other", .7), (2, "scaled", .2)),
])
def test_shared_roles_have_no_node_axis_and_transpose_alias_once(dtype, selected):
    roles, indices = shared_material(dtype)
    formula = SharedFormula()
    actual = execute(roles, indices, formula)
    expected = shared_literal(roles, indices)
    assert formula.shared == {"p", "q", "scale"}
    assert formula.shapes == ((3, 2, 2, 2), (2, 2), (2, 2), ())
    parameter = roles["p"].values[0]
    assert all(x is parameter for role in (roles["p"], roles["q"]) for x in role.values)
    equivalent(actual, expected)
    variables = roots(roles)
    got = vjp(objective(actual, selected), variables)
    equivalent(got, vjp(objective(expected, selected), variables))
    assert got["x.0"] is None
    if len(selected) == 1:
        assert got["x.1"] is None
        # Only node 2 contributes; G repeated argument slots must not multiply
        # the shared matrix VJP by G. The two aliased roles contribute 1 + 2.
        output = expected[2]["scaled"]
        bar = torch.linspace(-.3, .7, output.numel(), dtype=dtype).reshape_as(output)*selected[0][2]
        matrix_bar = torch.zeros_like(parameter)
        for row, index in enumerate(indices[2].tolist()):
            vector = roles["x"].values[2].detach()[index//2, index % 2]
            for d in range(2):
                for out in range(2):
                    matrix_bar[d, out] += 3*.6*vector[d]*bar[row, out]
        equivalent(got["p.0"], matrix_bar)
        if selected[0][2] == 0:
            assert all(got[name] is not None and got[name].count_nonzero() == 0
                       for name in ("x.2", "p.0", "scale.0"))


def test_private_formula_graph_is_constant_across_nodes_batch_and_length(dtype):
    observations = []
    for count, batch, length in ((2, 2, 2), (11, 2, 2), (2, 5, 7), (11, 5, 7)):
        roles, indices = shared_material(dtype, count, batch, length)
        with Census(executions=False, saved_tensors=False) as census:
            outputs = execute(roles, indices, SharedFormula())
            snapshot = census.snapshot()
        assert len(outputs) == count
        private = snapshot["groups"]["private/tensor_kernel"]
        assert private["registrations"] == 1
        assert private["unique_nodes"] > 0
        observations.append((private["unique_nodes"], private["connected_edges"], private["node_types"]))
    # This certifies private formula graph granularity, not total saved bytes,
    # runtime, or the number of distinct outer leaf roots and returned fields.
    assert all(observation == observations[0] for observation in observations[1:])
