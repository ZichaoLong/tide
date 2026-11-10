"""Explicit device Describe checks; CPU radial witnesses are a separate scope.

Set TIDE_DESCRIBE_DEVICE before process startup, and use a supported --dtype.
This checks actual placed fields/independent roots, not a device-selection label.
"""
import os
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.runtime import resolve_device
from tidegraph.weighted.query_vjp import evaluate
from test_weighted_describe_capability import prepared, linear_vjp
from test_weighted_query_vjp import native_query
from test_weighted_read_batch import CASES

DEVICE, _ = resolve_device(os.environ.get("TIDE_DESCRIBE_DEVICE", "cpu"))


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_device_describe_complete_fields_and_independent_roots(dtype, family, read, native):
    model, scalar, selected, rows, roots = prepared(dtype, family, read, device=DEVICE)
    assert [id(p) for p in model.parameters()] == [id(p) for p in selected.parameters()]
    assert all(p.device == DEVICE for p in roots.values())
    assert all(p.is_leaf for p in roots.values())
    for row in rows:
        if row.state is not None:
            assert all(t.device == DEVICE for t in row.state.slots.values())
            assert all(t.dtype == torch.int64 for t in row.state.slots.values() if not t.is_floating_point())
    expected = [scalar.read_program.step(scalar, row) for row in rows]
    actual = (native_query if native else evaluate)(selected.read_program, selected, rows)
    equivalent(actual, expected)
    for i in (0, 1, 2, len(rows)-1):
        for field in expected[i]:
            assert actual[i][field].device == DEVICE
            equivalent(linear_vjp(actual[i][field], roots), linear_vjp(expected[i][field], roots),
                       path=f"row.{i}.{field}")
    def partial(values):
        return values[1]["score"]*3 + values[2]["evidence"][0] + values[-1]["w"] + 0*values[0]["mu"]
    equivalent(linear_vjp(partial(actual), roots), linear_vjp(partial(expected), roots))
    equivalent(linear_vjp(actual[2]["evidence"]*0, roots), linear_vjp(expected[2]["evidence"]*0, roots))
