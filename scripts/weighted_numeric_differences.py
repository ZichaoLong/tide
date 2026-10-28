"""Explain strict comparison failures without changing the acceptance authority."""
import torch
from tidegraph.compare import equivalent


def tensors(left, right, *, limit=10):
    """Unique named leaves: None/zero, exact differences and original dtype tolerance.

    Ordering is by normalized tolerance excess. Nonfinite/metadata mismatches
    remain explicit failures, never NaN ratios silently filtered out of a report.
    ``equivalent`` still owns acceptance of the complete returned structure.
    """
    bad, nonidentical, examined = [], 0, 0
    for name in sorted(left.keys() | right.keys()):
        a, b = left.get(name), right.get(name)
        row = dict(name=name)
        if name not in left or name not in right:
            row["reason"] = "missing named root"
        elif a is None or b is None:
            if (a is None) == (b is None):
                continue
            row["reason"] = "None/connected gradient mismatch"
        elif (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device):
            row["reason"] = "tensor metadata mismatch"
        elif not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            row["reason"] = "nonfinite tensor"
        else:
            examined += 1
            if torch.equal(a, b):
                continue
            nonidentical += 1
            if not a.is_floating_point():
                row["reason"] = "exact integer/bool mismatch"
            else:
                atol, rtol = ((1e-10, 1e-8) if a.dtype == torch.float64 else
                              (1e-3, 2e-2) if a.dtype == torch.float16 else (1e-6, 1e-5))
                error = (a.detach().double()-b.detach().double()).abs()
                ratio = error/(atol+rtol*b.detach().double().abs())
                if not (torch.isfinite(error).all() and torch.isfinite(ratio).all()):
                    row["reason"] = "nonfinite difference or tolerance ratio"
                    bad.append(row)
                    continue
                if not ratio.numel() or float(ratio.max()) <= 1:
                    continue
                index = int(ratio.flatten().argmax())
                row.update(scaled_error=float(ratio.max()), maxabs=float(error.max()), flat_index=index,
                           left=float(a.detach().flatten()[index]), right=float(b.detach().flatten()[index]))
        bad.append(row)
    bad.sort(key=lambda r: r.get("scaled_error", float("inf")), reverse=True)
    return dict(examined_tensors=examined, nonidentical_tensors=nonidentical,
                failing_tensors=len(bad), worst=bad[:limit])


def first_trace_difference(left, right):
    """Earliest event with an exact record difference, not a causal diagnosis."""
    key = lambda e: (e["time"], e["batch"], e["node"])
    a, b = sorted(left.trace, key=key), sorted(right.trace, key=key)
    if len(a) != len(b):
        return dict(reason="event count mismatch", left=len(a), right=len(b))
    for x, y in zip(a, b):
        try:
            equivalent(x, y, atol=0, rtol=0)
        except AssertionError as error:
            return dict(left_event=list(key(x)), right_event=list(key(y)), error=str(error))
    return None
