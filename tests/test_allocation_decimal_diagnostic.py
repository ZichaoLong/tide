"""Independent allocation diagnostic: hand-solvable maps and original bits."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_allocation_failure import diagnose


def record(scores, caps, bound, tau=1., kind="quad", alpha=None):
    values = dict(scores=scores, caps=caps, lower=[bound], upper=[bound], temperature=[tau], controls=[0.]*len(scores))
    return dict(schema="tide-allocation-failure-v1", tensors={k: dict(values=v) for k, v in values.items()},
                details=dict(regularizer=kind, alpha=str(alpha)), row=1, time=2**60+7, nodes=[3, 5])


def test_decimal_preserves_actual_binary_bound_and_large_score_gap():
    from decimal import Decimal
    case = record([1e8, 0.], [1., 1.], 1.7000000476837158)
    result = diagnose(case)
    assert result["state"] == "solved" and result["time"] == 2**60+7
    exact = Decimal.from_float(case["tensors"]["lower"]["values"][0])-1
    assert abs(Decimal(result["controls"][1])-exact) < Decimal("1e-26")
    assert float(result["controls"][0]) == 1.
    assert float(result["residual"]) < 1e-100


@pytest.mark.parametrize("kind,alpha", [("quad", None), ("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
def test_decimal_equal_score_symmetry(kind, alpha):
    result = diagnose(record([3., 3.], [1., 1.], .5, kind=kind, alpha=alpha))
    assert [float(c) for c in result["controls"]] == [.25, .25]


def test_decimal_reports_exact_infeasibility_instead_of_clamping():
    result = diagnose(record([0., 0.], [.25, .25], .50000001))
    assert result["state"] == "infeasible_exact_inputs"
