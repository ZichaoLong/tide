"""Observe actual fallback work without changing the solved-map derivatives."""
from contextlib import nullcontext
import importlib

import pytest
import torch

from benchmarks.detailed_stages import DetailedStageTimes, OperatorTimes, detailed_stage_hooks
from tide import ModelConfig, TideModel, TopologyConfig


@pytest.mark.parametrize("observer_type", [DetailedStageTimes, OperatorTimes])
def test_allocation_fallback_counts_requested_and_executed_rows(monkeypatch, observer_type):
    allocation = importlib.import_module("tide.allocation")
    original = allocation._quad_sorted

    def force_recovery(*args):
        value, valid = original(*args)
        return value, torch.zeros_like(valid)

    monkeypatch.setattr(allocation, "_quad_sorted", force_recovery)
    model = TideModel(ModelConfig(topology=TopologyConfig(n=2, layers=1), d=64))
    # The second row has an exactly representable sum of one, so only the
    # first row requests bisection while both rows enter the tensor kernel.
    source = torch.tensor([[1.3, .8, .2], [.125, .375, .5]])
    cotangent = torch.tensor([[.2, -.7, .9], [.5, -.4, .8]])

    def solve(x):
        return allocation.allocate(x, torch.full_like(x, .8), torch.ones_like(x),
                                   torch.zeros_like(x), torch.ones_like(x), 1.)

    x = source.clone().requires_grad_()
    reference = solve(x)
    expected = torch.autograd.grad((reference*cotangent).sum(), x)[0]
    x = source.clone().requires_grad_()
    observer = observer_type(torch.device("cpu"))
    with detailed_stage_hooks(model, observer), (observer if isinstance(observer, OperatorTimes) else nullcontext()):
        actual = solve(x)
        gradient = torch.autograd.grad((actual*cotangent).sum(), x)[0]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
    report = observer.report()
    rows = [row for row in report["allocation_work"] if row["kind"] == "bisect"]
    assert len(rows) == 1
    assert rows[0]["requested_rows"] == 1 and rows[0]["executed_rows"] == 2
    assert rows[0]["bisection_iterations"] == 40
    assert allocation._quad_sorted is force_recovery
    assert any("observation.metadata" in row["path"] for row in report["rows"])
