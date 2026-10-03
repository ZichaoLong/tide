"""Floating tolerances must never certify different discrete graph tensors."""
from dataclasses import dataclass

import pytest
import torch

from tidegraph.compare import equivalent
from tidegraph.qualification_checks import compare_finite


@dataclass
class Observation:
    pending: dict


@pytest.mark.parametrize("compare", [equivalent, compare_finite])
@pytest.mark.parametrize("tolerances", [{}, {"atol": 100., "rtol": 100.}])
@pytest.mark.parametrize("storage_dtype,left,right", [
    (torch.int64, 2**53, 2**53 + 1),
    (torch.int64, 2**63 - 2, 2**63 - 1),
    (torch.int64, -(2**63), -(2**63) + 1),
    (torch.int32, 2**24, 2**24 + 1),
    (torch.int16, 120, 121),
    (torch.int8, -2, -1),
    (torch.uint8, 0, 1),
    (torch.bool, False, True),
])
def test_nested_discrete_mismatch_is_exact(compare, tolerances, storage_dtype, left, right):
    expected = Observation({"time_or_mask": [torch.tensor([left], dtype=storage_dtype)]})
    actual = Observation({"time_or_mask": [torch.tensor([right], dtype=storage_dtype)]})
    with pytest.raises(AssertionError, match="root.pending\\[time_or_mask\\]\\[0\\]"):
        compare(expected, actual, **tolerances)
    compare(expected, Observation({"time_or_mask": [torch.tensor([left], dtype=storage_dtype)]}), **tolerances)


@pytest.mark.parametrize("compare", [equivalent, compare_finite])
def test_payload_tolerance_remains_independent_of_discrete_values(compare):
    expected = {"payload": torch.tensor([1.]), "time": torch.tensor([2**63 - 1])}
    actual = {"payload": torch.tensor([1.1]), "time": torch.tensor([2**63 - 1])}
    with pytest.raises(AssertionError):
        compare(expected, actual)
    compare(expected, actual, atol=.2, rtol=0.)


@pytest.mark.parametrize("compare", [equivalent, compare_finite])
def test_discrete_shape_dtype_and_gradient_presence_still_fail(compare):
    for actual in [torch.tensor([True]), torch.tensor([1], dtype=torch.int32), torch.tensor(1)]:
        with pytest.raises(AssertionError):
            compare(torch.tensor([1], dtype=torch.int64), actual, atol=100., rtol=100.)
    with pytest.raises(AssertionError):
        compare({"grad": None}, {"grad": torch.tensor(0.)}, atol=100., rtol=100.)
