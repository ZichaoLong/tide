"""Canonical comparisons fail on discrete differences before tensor tolerance."""
from dataclasses import fields, is_dataclass
import torch


def equivalent(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            raise AssertionError(f"{path}: missing tensor")
        tol = (1e-10, 1e-8) if a.dtype == torch.float64 else (1e-6, 1e-5)
        torch.testing.assert_close(a, b, atol=tol[0], rtol=tol[1], msg=lambda msg: f"{path}: {msg}")
    elif is_dataclass(a):
        if type(a) is not type(b):
            raise AssertionError(f"{path}: different record types")
        for f in fields(a):
            if f.name != "stats":
                equivalent(getattr(a, f.name), getattr(b, f.name), path + "." + f.name)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: key/route/gradient-presence mismatch {a.keys()} != {b.keys()}")
        for k in a:
            equivalent(a[k], b[k], f"{path}[{k}]")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: record count mismatch {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            equivalent(x, y, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"{path}: {a} != {b}")


def objective(result, root="all"):
    terms = []
    if root in {"all", "output"}:
        terms += [x.square().sum() * 0.7 for _, _, _, x in result.outputs]
    if root in {"all", "state"}:
        terms += [s.value.square().sum() * 0.3 for s in result.continuation.states.values()]
    if root in {"all", "pending"}:
        terms += [m.value.square().sum() * 0.2 for m in result.continuation.pending]
    if not terms:
        raise ValueError("objective has no tensor roots")
    return sum(terms)
