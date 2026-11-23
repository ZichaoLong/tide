"""Bound the diagnostic exposure and retain update-scale failures separately."""
import torch

from pilotlab.risk_screen3 import budget_tokens, errors


def test_risk_screen_reserves_replay_and_precision_exposure():
    assert budget_tokens({"batch_size": 2, "sequence_length": 128, "screen_steps": 128,
                          "fp64_steps": [0, 127], "eval_tokens": 512}) == (33536, 34592)


def test_small_gradient_can_have_material_adam_update_discrepancy():
    actual = {"gradients": torch.tensor([1e-9]), "parameter_delta": torch.tensor([9e-5])}
    ref = {"gradients": torch.tensor([0.], dtype=torch.float64),
           "parameter_delta": torch.tensor([0.], dtype=torch.float64)}
    report = errors(actual, ref)
    assert not report["original_threshold_gate"]["within_tolerance"]
    assert report["original_threshold_gate"]["finite"]
    assert report["fields"]["gradients"]["failed_elements"] == 0
    assert report["fields"]["parameter_delta"]["failed_elements"] == 1


def test_discrete_failure_is_not_a_finite_precision_deviation():
    report = errors({"state": torch.tensor([1], dtype=torch.int64)},
                    {"state": torch.tensor([2], dtype=torch.int64)})
    assert not report["original_threshold_gate"]["integer_equal"]
