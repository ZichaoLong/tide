"""Allocation §4.1/6: preserve original rejected rows without old run directories.

Independent Decimal solutions anchor current scalar/batch local forwards.
This does not certify the old full model, gradients or training trajectory.
"""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/"tests/fixtures/allocation"
sys.path.insert(0, str(ROOT/"scripts"))
from diagnose_allocation_failure import diagnose


def captured(name):
    entry = next(r for r in json.loads((FIXTURES/"provenance.json").read_text())["fixtures"]
                 if r["file"] == name+".json")
    data = (FIXTURES/entry["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return json.loads(data)


@pytest.mark.parametrize("name", ["wide2048", "matched2048"])
def test_captured_rejection_has_an_independent_feasible_solution(name):
    record = captured(name); result = diagnose(record)
    assert result["state"] == "solved"
    assert Decimal(result["residual"]) < Decimal("1e-100")
    tolerance = record["tensors"]["tolerance"]["values"][0]
    assert Decimal(result["candidate_bound_residual"]) > Decimal.from_float(tolerance)
    assert result["time"] == record["time"] and result["nodes"] == record["nodes"]


@pytest.mark.parametrize("name", ["wide2048", "matched2048"])
@pytest.mark.parametrize("implementation", ["python-scalar", "python-batch", "native-scalar", "native-batch"])
def test_current_local_quad_matches_decimal_for_retired_actual_row(name, implementation):
    if implementation.startswith("native") and not os.environ.get("TIDE_BUILD_DIR"):
        pytest.skip("matching native build required")
    record = captured(name)
    values = record["tensors"]
    tensor = lambda key: torch.tensor(values[key]["values"], dtype=torch.float32).reshape(values[key]["shape"])
    scores, caps, low, high, tau = map(tensor, ("scores", "caps", "lower", "upper", "temperature"))
    if implementation == "python-scalar":
        from tidegraph.weighted.allocation import allocate_quad
        controls = allocate_quad(scores, caps, low, high, tau).controls
    elif implementation == "python-batch":
        from tidegraph.weighted.allocation_batch import solve
        from tidegraph.weighted.regularizers import Regularizer
        controls = solve(scores[None], caps[None], low[None], high[None], tau[None], Regularizer("quad")).controls[0]
    else:
        import _tide_native as core
        if implementation == "native-scalar":
            controls = core.weighted_allocate_quad(scores, caps, low, high, tau)
        else:
            controls = core.weighted_allocation_batch([(scores, caps, low, high, tau, torch.zeros_like(low))], "quad", None)[0][0]
    oracle = torch.tensor([float(v) for v in diagnose(record)["controls"]], dtype=torch.float32)
    torch.testing.assert_close(controls, oracle, rtol=1e-5, atol=1e-6)
    assert float((controls.double().sum()-low.double()).abs()) <= values["tolerance"]["values"][0]
