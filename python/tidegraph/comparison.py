"""Typed strict comparisons and finite, separately classified diagnostics.

Only finite floating value errors are numerical mismatches. Shape, dtype,
owner presence, int64 coordinates and routes cannot be accepted by numerical
exploration. This module executes no candidate, solver or communication call.
"""
from dataclasses import fields, is_dataclass
import math
import torch


class ComparisonMismatch(AssertionError):
    kind = "comparison"

    def __init__(self, path, message, **details):
        self.path, self.message, self.details = path, message, details
        super().__init__(f"{path}: {message}")

    def record(self):
        return dict(kind=self.kind, path=self.path, message=self.message, **self.details)


class NumericMismatch(ComparisonMismatch):
    kind = "numeric"


class DiscreteMismatch(ComparisonMismatch):
    kind = "discrete"


class StructureMismatch(ComparisonMismatch):
    kind = "structure"


class NonFiniteMismatch(ComparisonMismatch):
    kind = "nonfinite"


def finite(value, path="root"):
    """Reject nonfinite mathematical leaves, ignoring only dataclass stats."""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise NonFiniteMismatch(path, "nonfinite tensor")
    elif is_dataclass(value):
        for field in fields(value):
            if field.name != "stats":
                finite(getattr(value, field.name), path + "." + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}[{key}]")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            finite(item, f"{path}[{i}]")
    elif isinstance(value, (float, complex)):
        parts = (value.real, value.imag) if isinstance(value, complex) else (value,)
        if not all(math.isfinite(part) for part in parts):
            raise NonFiniteMismatch(path, "nonfinite scalar")


def _raise(error):
    raise error


def _diagnostic_number(value):
    value = float(value)
    return value if math.isfinite(value) else "overflow"


def _tensor(a, b, path, emit, atol, rtol, check_device, counts):
    if a.dtype != b.dtype or a.shape != b.shape or a.layout != b.layout:
        raise StructureMismatch(path, "tensor metadata mismatch",
                                actual_dtype=str(a.dtype), expected_dtype=str(b.dtype),
                                actual_shape=list(a.shape), expected_shape=list(b.shape))
    if check_device and a.device != b.device:
        raise StructureMismatch(path, "tensor device mismatch",
                                actual_device=str(a.device), expected_device=str(b.device))
    floating = a.is_floating_point() or a.is_complex()
    kind = "numeric" if floating else "discrete"
    counts[kind] += 1
    if floating:
        tol = ((1e-10, 1e-8) if a.dtype == torch.float64 else
               (1e-3, 2e-2) if a.dtype == torch.float16 else (1e-6, 1e-5))
        tensor_atol = tol[0] if atol is None else atol
        tensor_rtol = tol[1] if rtol is None else rtol
    else:
        tensor_atol = tensor_rtol = 0
    # Exact CPU recovery often compares multi-GB identical banks. Equality is
    # sufficient after metadata and finite validation; mismatches still use
    # the original strict assertion and diagnostics below. No tolerance path
    # or accelerator operator is changed by this positive-only shortcut.
    if (tensor_atol == 0 and tensor_rtol == 0 and a.device.type == b.device.type == "cpu"
            and a.layout == torch.strided and not a.is_quantized and torch.equal(a, b)):
        return
    try:
        torch.testing.assert_close(a, b, atol=tensor_atol, rtol=tensor_rtol,
                                   check_device=check_device)
    except AssertionError as error:
        if not floating:
            emit(DiscreteMismatch(path, "integer/bool tensor value mismatch"))
            return
        left, right = a.detach().cpu(), b.detach().cpu()
        emit(NumericMismatch(path, str(error), dtype=str(a.dtype), shape=list(a.shape),
                             atol=tensor_atol, rtol=tensor_rtol,
                             max_abs_error=_diagnostic_number((left-right).abs().max()),
                             reference_max_abs=_diagnostic_number(right.abs().max())))


def _walk(a, b, path, emit, atol, rtol, check_device, counts):
    # None is mathematical absence, including a disconnected gradient. It is
    # distinct from every present tensor, even a connected zero tensor.
    if a is None or b is None:
        counts["discrete"] += 1
        if a is not b:
            emit(DiscreteMismatch(path, "None/present value mismatch"))
    elif isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise StructureMismatch(path, "missing tensor or different value type")
        _tensor(a, b, path, emit, atol, rtol, check_device, counts)
    elif is_dataclass(a) or is_dataclass(b):
        if type(a) is not type(b):
            raise StructureMismatch(path, "different record types")
        for field in fields(a):
            if field.name != "stats":
                _walk(getattr(a, field.name), getattr(b, field.name), path + "." + field.name,
                      emit, atol, rtol, check_device, counts)
    elif isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise StructureMismatch(path, "different mapping types")
        if a.keys() != b.keys():
            actual_only, expected_only = a.keys()-b.keys(), b.keys()-a.keys()
            emit(DiscreteMismatch(path, "key/route/gradient-presence mismatch",
                                  actual_only=sorted(map(str, actual_only))[:32],
                                  expected_only=sorted(map(str, expected_only))[:32],
                                  actual_only_count=len(actual_only), expected_only_count=len(expected_only)))
        for key in a:
            if key in b:
                _walk(a[key], b[key], f"{path}[{key}]", emit, atol, rtol, check_device, counts)
    elif isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        if not isinstance(a, (tuple, list)) or not isinstance(b, (tuple, list)):
            raise StructureMismatch(path, "different sequence types")
        if len(a) != len(b):
            emit(DiscreteMismatch(path, f"record count mismatch {len(a)} != {len(b)}"))
        for i, (left, right) in enumerate(zip(a, b)):
            _walk(left, right, f"{path}[{i}]", emit, atol, rtol, check_device, counts)
    else:
        # Python scalars retain the historical exact comparison. Floating
        # tensor tolerances do not silently change metadata scalar equality.
        numeric = isinstance(a, (float, complex)) and isinstance(b, (float, complex))
        counts["numeric" if numeric else "discrete"] += 1
        if a != b:
            cls = NumericMismatch if numeric else DiscreteMismatch
            emit(cls(path, f"{a!r} != {b!r}"))


def equivalent(a, b, path="root", *, atol=None, rtol=None, check_device=True):
    """Strict historical tolerances; typed failures remain AssertionErrors."""
    finite(a, path + ".actual")
    finite(b, path + ".expected")
    _walk(a, b, path, _raise, atol, rtol, check_device, dict(numeric=0, discrete=0))


def collect_comparison(a, b, path="root", *, atol=None, rtol=None,
                       check_device=True, max_differences=32):
    """Inspect all comparable leaves, retaining bounded numerical diagnostics.

    Nonfinite values and malformed structures raise typed errors. Route,
    presence and exact metadata differences are returned separately and make
    numerical agreement ``not_comparable``. No exception from execution is
    caught here, and the report does not itself authorize exploration.
    """
    if type(max_differences) is not int or max_differences < 1:
        raise ValueError("max_differences must be a positive integer")
    finite(a, path + ".actual")
    finite(b, path + ".expected")
    differences, mismatches = [], dict(numeric=0, discrete=0)
    compared = dict(numeric=0, discrete=0)

    def collect(error):
        mismatches[error.kind] += 1
        if len(differences) < max_differences:
            differences.append(error.record())

    _walk(a, b, path, collect, atol, rtol, check_device, compared)
    discrete = "failed" if mismatches["discrete"] else "passed"
    numeric = ("not_comparable" if mismatches["discrete"] else
               "failed" if mismatches["numeric"] else "passed")
    return dict(schema="tide-comparison-v1", finite="passed", numeric=numeric, discrete=discrete,
                strict_passed=not any(mismatches.values()), compared_leaves=compared,
                mismatch_counts=mismatches, differences=differences,
                differences_truncated=sum(mismatches.values()) > len(differences))
