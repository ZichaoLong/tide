"""Installed validation distinguishes rounded values from broken contracts."""
from copy import deepcopy

import pytest
import torch

from scripts import check_installed_capacity as harness


@pytest.fixture
def complete_case(monkeypatch):
    monkeypatch.setattr(harness, "torch", torch, raising=False)
    exact=harness.compare({"values":torch.tensor([1.]), "times":torch.tensor([2])},
                          {"values":torch.tensor([1.]), "times":torch.tensor([2])})
    return dict(inference_grad_forward=exact,saved_checkpoint_data=deepcopy(exact),
                checkpoint_config_exact=True,projection_dtype_passed=True,
                grad_forward_connected=True,initial_vjp=dict(input_connected=True,nonfinite=[]),
                resume_exit_code=0,resume=dict(state="within_threshold"))


def test_mode_rounding_keeps_original_zero_threshold_as_finite_deviation(complete_case):
    complete_case["inference_grad_forward"]=harness.compare(
        {"values":torch.tensor([1.000001]),"times":torch.tensor([2])},
        {"values":torch.tensor([1.]),"times":torch.tensor([2])})
    result=complete_case["inference_grad_forward"]
    assert result["threshold_failures"]==["root.values"]
    assert result["atol"]==result["rtol"]==0
    assert not result["metadata_failures"] and result["finite"]
    assert harness.classify_case(complete_case)=="finite_deviation"


@pytest.mark.parametrize("failure", ["time", "nonfinite", "none", "saved_data", "input", "output"])
def test_structural_contracts_still_fail(complete_case,failure):
    if failure in {"time","nonfinite","none"}:
        actual={"value":torch.tensor([1.]),"times":torch.tensor([2]),"gradient":None}
        expected=deepcopy(actual)
        if failure=="time":actual["times"]+=1
        if failure=="nonfinite":actual["value"][0]=float("nan")
        if failure=="none":actual["gradient"]=torch.zeros(1)
        complete_case["inference_grad_forward"]=harness.compare(actual,expected)
    elif failure=="saved_data":
        complete_case["saved_checkpoint_data"]=harness.compare(torch.tensor([1.000001]),torch.tensor([1.]))
    elif failure=="input":complete_case["initial_vjp"]["input_connected"]=False
    else:complete_case["grad_forward_connected"]=False
    assert harness.classify_case(complete_case)=="structural_failed"


def test_support_change_is_distinct_and_parameter_weights_are_not_support(complete_case):
    complete_case["inference_grad_forward"]=harness.compare(
        {"strengths":torch.tensor([1e-8]),"parameters":{"weight":torch.tensor([1.])}},
        {"strengths":torch.tensor([0.]),"parameters":{"weight":torch.tensor([-1.])}})
    assert complete_case["inference_grad_forward"]["support_differences"]==[
        dict(name="root.strengths",different_entries=1,entries=1)]
    assert harness.classify_case(complete_case)=="support_deviation"


def test_resume_failure_cannot_be_hidden_by_exact_mode_forward(complete_case):
    complete_case["resume"]["state"]="structural_failed"
    assert harness.classify_case(complete_case)=="structural_failed"
