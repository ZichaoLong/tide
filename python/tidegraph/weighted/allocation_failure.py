"""Actual failed allocation inputs/results, separate from successful returns.

Allocation §6/S.2. Float values use round-trippable JSON numbers, int64 metadata
uses integers. Nonfinite diagnostics use explicit strings, never JSON null.
"""
import math
import torch


class AllocationFailure(ArithmeticError):
    def __init__(self, message, row, tensors, details):
        super().__init__(message)
        self.row = int(row)
        self.tensors = {name: value.detach().clone() for name, value in tensors.items() if value is not None}
        self.details = dict(details)
        self.time, self.nodes, self.members = None, [], []

    def record(self):
        def number(value):
            return value if math.isfinite(value) else str(value)
        def tensor(value):
            return dict(dtype=str(value.dtype).removeprefix("torch."), device=str(value.device),
                        shape=list(value.shape), values=[number(v) for v in value.cpu().flatten().tolist()])
        return dict(schema="tide-allocation-failure-v1", error=str(self), row=self.row,
                    tensors={k: tensor(v) for k, v in self.tensors.items()}, details=self.details,
                    time=self.time, nodes=self.nodes, members=self.members)


def scalar_failure(message, scores, caps, lower, upper, temperature, *, regularizer="quad",
                   ent_alpha=None, branch="unknown", stage="numeric", score_shift=None, **actual):
    """Snapshot only values actually computed; malformed input stays ValueError."""
    values = dict(scores=scores, caps=caps, lower=lower, upper=upper, temperature=temperature,
                  shift=score_shift, **actual)
    values = {k: v if isinstance(v, torch.Tensor) or v is None else scores.new_tensor(v)
              for k, v in values.items()}
    return AllocationFailure(message, 0, values, dict(regularizer=regularizer, alpha=str(ent_alpha),
        branch=branch, solver="python-scalar", stage=stage))
