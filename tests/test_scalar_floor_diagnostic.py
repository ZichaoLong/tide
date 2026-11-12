"""Independent input VJP/boundary evidence for the REQ-006 scalar-floor probe."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_scalar_floor import (boundary_material, bound_argument, bound_value,
                                   evaluate, independent_formula, represented_bound)


@pytest.mark.parametrize("name", ["zero", "tiny", "log-tiny"])
@pytest.mark.parametrize("variant", ["scalar", "tensor"])
def test_exact_threshold_adjacent_zero_tiny_value_and_input_vjp(dtype, name, variant):
    value = bound_value(name, dtype)
    x, bar = boundary_material(dtype, value)
    x.requires_grad_()
    bound = bound_argument(x, value, variant)
    y, gradient = evaluate(x, bar, bound, "backward")
    ey, eg = independent_formula(x.detach(), bar, value)
    assert torch.equal(y, ey)
    assert torch.equal(gradient, eg)
    # Literal branch roots: below, at, above. Equality sends the input cotangent.
    assert torch.equal(gradient[:3], x.new_tensor([0., -3., 5.]))
    assert torch.equal(y[:3], torch.stack((x[1], x[1], x[2])))
    assert not isinstance(bound, torch.Tensor) or (bound.ndim == 0 and not bound.requires_grad)


@pytest.mark.parametrize("variant", ["scalar", "tensor"])
def test_strided_rows_arbitrary_cotangents_and_forward_does_not_build_graph(dtype, variant):
    value = math.log(torch.finfo(dtype).tiny)
    x = torch.tensor([[value-1, value+1, 0.], [value+2, value-2, -0.]], dtype=dtype).T.requires_grad_()
    assert not x.is_contiguous()
    bar = torch.tensor([[2., -3.], [0., 7.], [-11., 13.]], dtype=dtype)
    bound = bound_argument(x, value, variant)
    y, gradient = evaluate(x, bar, bound, "backward")
    expected, expected_gradient = independent_formula(x.detach(), bar, value)
    assert torch.equal(y, expected)
    assert torch.equal(gradient, expected_gradient)
    no_grad, absent = evaluate(x, bar, bound, "forward")
    assert not no_grad.requires_grad and absent is None
    assert torch.equal(no_grad, expected)


def test_dtype_bound_is_explicit_constant_and_unknown_variant_is_rejected():
    x = torch.ones(2, dtype=torch.float32)
    value = math.log(torch.finfo(x.dtype).tiny)
    bound = bound_argument(x, value, "tensor")
    assert bound.dtype == x.dtype and bound.device == x.device and bound.shape == ()
    assert bound.item() == torch.tensor(value, dtype=x.dtype).item()
    with pytest.raises(ValueError, match="unknown"):
        bound_argument(x, value, "maximum")


def test_scalar_representation_is_explicit_and_preserves_the_original_request():
    value = bound_value("log-tiny", torch.float32)
    rounded = represented_bound(value, torch.float32, "dtype")
    assert rounded != value
    assert represented_bound(value, torch.float32, "requested") == value
    assert rounded == torch.tensor(value, dtype=torch.float32).item()
    x, bar = boundary_material(torch.float32, rounded)
    for variant in ("scalar", "tensor"):
        result = evaluate(x.requires_grad_(), bar, bound_argument(x, rounded, variant), "backward")
        expected = independent_formula(x.detach(), bar, rounded)
        assert torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])
    with pytest.raises(ValueError, match="representation"):
        represented_bound(value, torch.float32, "implicit")


@pytest.mark.parametrize("variant", ["scalar", "tensor"])
def test_nonfinite_classification_and_input_vjp_are_not_sanitized(dtype, variant):
    x = torch.tensor([-math.inf, math.inf, math.nan], dtype=dtype, requires_grad=True)
    bar = x.new_tensor([2., -3., 5.])
    value = bound_value("log-tiny", dtype)
    result = evaluate(x, bar, bound_argument(x, value, variant), "backward")
    expected = independent_formula(x.detach(), bar, value)
    torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0, equal_nan=True)
    assert torch.isposinf(result[0][1]) and torch.isnan(result[0][2])
    assert torch.equal(result[1], x.new_tensor([0., -3., 0.]))
    assert torch.equal(result[1], expected[1])
