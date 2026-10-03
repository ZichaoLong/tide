"""Canonical comparisons fail on discrete differences before tensor tolerance."""
from dataclasses import fields, is_dataclass
import torch


def equivalent(a, b, path="root", *, atol=None, rtol=None, check_device=True):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            raise AssertionError(f"{path}: missing tensor")
        if a.is_floating_point() or a.is_complex():
            tol = (1e-10, 1e-8) if a.dtype == torch.float64 else (1e-3, 2e-2) if a.dtype == torch.float16 else (1e-6, 1e-5)
            tensor_atol, tensor_rtol = tol[0] if atol is None else atol, tol[1] if rtol is None else rtol
        else:
            # Integer time/count/identity and bool masks are exact, even when
            # the caller relaxes payload tolerances. Zero tolerances also avoid
            # converting large int64 values to floating point for comparison.
            tensor_atol = tensor_rtol = 0
        torch.testing.assert_close(a, b, atol=tensor_atol,
                                   rtol=tensor_rtol, check_device=check_device,
                                   msg=lambda msg: f"{path}: {msg}")
    elif is_dataclass(a):
        if type(a) is not type(b):
            raise AssertionError(f"{path}: different record types")
        for f in fields(a):
            if f.name != "stats":
                equivalent(getattr(a, f.name), getattr(b, f.name), path + "." + f.name, atol=atol, rtol=rtol, check_device=check_device)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: key/route/gradient-presence mismatch {a.keys()} != {b.keys()}")
        for k in a:
            equivalent(a[k], b[k], f"{path}[{k}]", atol=atol, rtol=rtol, check_device=check_device)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: record count mismatch {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, f"{path}[{i}]", atol=atol, rtol=rtol, check_device=check_device)
    elif a != b:
        raise AssertionError(f"{path}: {a} != {b}")


def objective(result, root="all"):
    def square(value):
        return (value.float() if value.dtype == torch.float16 else value).square().sum()
    terms = []
    if root in {"all", "output"}:
        terms += [square(x) * 0.7 for _, _, _, x in result.outputs]
    if root in {"all", "state"}:
        terms += [square(s.value) * 0.3 for s in result.continuation.states.values()]
        terms += [square(v) * 0.11 for s in result.continuation.states.values() for v in s.slots.values()]
    if root in {"all", "history"}:
        terms += [square(v) * 0.13 for h in result.continuation.history.values() for v in h.tensors.values()]
    if root in {"all", "pending"}:
        terms += [square(m.value) * 0.2 for m in result.continuation.pending]
    if not terms:
        raise ValueError("objective has no tensor roots")
    return sum(term.to(terms[0].device) for term in terms)
