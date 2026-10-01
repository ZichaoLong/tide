"""Static tensor envelope for the packed consumer head (not total device HBM)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadBudget:
    rows: int
    fixed_bytes: int
    row_bytes: int
    reserved_bytes: int
    budget: int
    operator_allowance_bytes: int


def head_budget(capacity, width, vocab, payload, backward, budget, aggressive=False):
    if (any(type(x) is not int or not 0 < x < 2**63 for x in (capacity, width, vocab, payload, budget))
            or payload not in (2, 4)):
        raise ValueError("invalid head workspace geometry/budget")
    # Local CANN operator calibration observes a 16MiB floor even at one row.
    # Keep twice that allowance, plus the policy headroom and tensor envelope.
    operators = 32*1024**2
    fixed = operators + 4096 + 8*capacity + (4*capacity*width + 4*(3+(payload == 2))*vocab*width if backward else 0)
    row = (32 if backward else 16)*vocab + (payload+(12 if backward else 4))*width + 160
    usable = budget-budget//(10 if aggressive else 4)
    if fixed+row > usable:
        raise ValueError("one output head row exceeds head-workspace-bytes")
    rows = min(capacity, (usable-fixed)//row)
    return HeadBudget(rows, fixed, row, fixed+rows*row, budget, operators)
