"""Call-local regional parameter views and the actual sending operand bundle.

CPU dtype changes exercise real parameter copies. The fake selection transport
executes CPU copies through the same VJP primitive; it is not device evidence.
"""
from contextlib import nullcontext
import pytest
import torch

from tidegraph.compare import equivalent
from tidegraph.weighted import column_placement, placement_tensors
from tidegraph.weighted.parameter_tree import rebind
from tidegraph.weighted.regional_columns import _placed
from tidegraph.weighted.regional_groups import region_groups
from tidegraph.weighted.regional_packing import RegionalPacking
from test_regional_groups import material, literal, compare_roots


CPU = torch.device("cpu")


@pytest.mark.parametrize("path", ["grouped", "aligned", "inference"])
def test_parameter_conversion_is_shared_by_all_candidate_count_groups(monkeypatch, path):
    program, packing, fields, request = material(torch.float64, 3)
    program.float()
    program.query.out_weight = program.query.in_weight
    program.head.vector.requires_grad_(False)
    if path == "aligned":
        packing = RegionalPacking({v: torch.tensor([0, 2] if v < 2 else [1, 3]) for v in fields})
    assert len(packing.groups) == 2
    copies, mapper_calls = [], []
    original_copy, original_mapper = placement_tensors.copy_tensor, column_placement.mapper

    def copied(value, device, dtype, **options):
        copies.append(value.numel())
        return original_copy(value, device, dtype, **options)

    def mapped(values, device, dtype):
        values = list(values)
        mapper_calls.append(values)
        return original_mapper(values, device, dtype)

    monkeypatch.setattr(placement_tensors, "copy_tensor", copied)
    monkeypatch.setattr(column_placement, "mapper", mapped)
    with torch.no_grad() if path == "inference" else nullcontext():
        actual = region_groups(program, packing, fields, control=CPU, selection=CPU)
    assert len(mapper_calls) == 1
    assert copies == [sum(p.numel() for p in program.parameters())]
    expected = literal(rebind(program, lambda t: t.double()), packing, fields, request)
    equivalent(actual, expected)
    if path == "inference":
        assert all(not x.requires_grad for x in actual.values())
        return
    roots = [p for p in program.parameters() if p.requires_grad]
    roots += [t for row in fields.values() for t in row.values()]
    for node in (0, 2, 4):
        for factor in (1., 0.):
            compare_roots(actual[node].square().sum()*factor, expected[node].square().sum()*factor, roots)
    compare_roots(sum(x.square().sum() for x in actual.values()),
                  sum(x.square().sum() for x in expected.values()), roots)


def test_same_location_views_are_borrowed_and_calls_do_not_cache_parameter_versions(monkeypatch):
    program, _, _, _ = material(torch.float64, 3)
    def forbidden(*args):
        raise AssertionError("co-located region parameters were packed")
    monkeypatch.setattr(column_placement, "mapper", forbidden)
    views = column_placement.RegionParameterViews(program, CPU)
    assert views.parameters(torch.float64) is program
    assert views.parameters(torch.float64) is program
    monkeypatch.undo()
    first = column_placement.RegionParameterViews(program, CPU).parameters(torch.float32)
    with torch.no_grad():
        program.head.matrix.add_(.1)
    second = column_placement.RegionParameterViews(program, CPU).parameters(torch.float32)
    assert not torch.equal(first.head.matrix, second.head.matrix)
    torch.testing.assert_close(second.head.matrix, program.head.matrix.float())


@pytest.mark.parametrize("role", ["frozen_parameter", "unused_descriptor"])
def test_grouped_placed_view_keeps_original_versions_and_first_order(role):
    program, packing, fields, _ = material(torch.float64, 3)
    program.float()
    program.head.vector.requires_grad_(False)
    actual = region_groups(program, packing, fields, control=CPU, selection=CPU)
    with pytest.raises(ValueError, match="first-order"):
        torch.autograd.grad(actual[0].sum(), program.head.matrix, create_graph=True)
    changed = program.head.vector if role == "frozen_parameter" else fields[2]["mu"]
    with torch.no_grad():
        changed.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        torch.autograd.grad(actual[0].sum(), program.head.matrix)


def fake_selection_copy(monkeypatch):
    copied = []
    def copy(value, device, dtype, **options):
        copied.append((value.numel(), value.dtype))
        return value.to(dtype=dtype).clone()
    monkeypatch.setattr(placement_tensors, "copy_tensor", copy)
    return copied


@pytest.mark.parametrize("case", range(12))
def test_selection_moves_only_function_operands_and_retains_full_region_vjp(monkeypatch, case):
    program, packing, fields, request = material(torch.float64, case)
    copied = fake_selection_copy(monkeypatch)
    actual = region_groups(program, packing, fields, control=CPU, selection=torch.device("cuda:0"))
    expected_copies = []
    mode = program.config.selection_mode
    for group in packing.groups:
        r, n = len(group.rows), group.candidates
        if mode == "proportional":
            floats, ids = r*n, False
        elif mode.endswith("attention"):
            floats, ids = 3*r*n, False
        elif mode in {"regional_total", "regional_total_zero"}:
            floats, ids = 2*r*n+r, mode == "regional_total_zero"
        elif mode in {"hard_selection", "regional_threshold_hard"}:
            floats, ids = 2*r*n+r, True
        else:
            floats, ids = 2*r*n+3*r, False
        expected_copies.append((floats, torch.float64))
        if ids:
            expected_copies.append((r*n, torch.int64))
    assert copied == expected_copies
    expected = literal(program, packing, fields, request)
    equivalent(actual, expected)
    roots = [*program.parameters(), *(t for row in fields.values() for t in row.values())]
    for node in (0, 4):
        for factor in (1., 0.):
            compare_roots(actual[node].square().sum()*factor, expected[node].square().sum()*factor, roots)


def test_selection_total_alias_is_one_transport_root_and_equal_values_stay_distinct(monkeypatch):
    copied = fake_selection_copy(monkeypatch)
    a = torch.tensor([.2, .4], requires_grad=True)
    other = a.detach().clone().requires_grad_()
    first, second, independent, absent = _placed(torch.device("cuda:0"), a, a, other, None)
    assert first is second and first is not independent and absent is None
    assert copied == [(4, torch.float32)]
    ga, go = torch.autograd.grad((first+2*second).sum(), (a, other), retain_graph=True, allow_unused=True)
    torch.testing.assert_close(ga, torch.full_like(a, 3))
    assert go is None
    ga, go = torch.autograd.grad(first.sum()*0, (a, other), retain_graph=True, allow_unused=True)
    assert torch.equal(ga, torch.zeros_like(a)) and go is None
    with torch.no_grad():
        other.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        first.sum().backward()
