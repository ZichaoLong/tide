"""Explicit routes preserve independent zero/None roots and int64 metadata."""
import pytest
import torch
from tidegraph.communication import copy, policy
from tidegraph.transfer import copy_rows
from tidegraph.weighted.placement_tensors import move, mapper


@pytest.mark.parametrize("selected", ["ordinary", "blocking", "cpu_bridge"])
def test_private_outputs_none_zero_and_versions(selected, monkeypatch):
    monkeypatch.setenv("TIDE_COMMUNICATION_POLICY", selected)
    x, y = [torch.arange(6., dtype=torch.float64, requires_grad=True) for _ in range(2)]
    out = copy_rows([x, y], "cpu")
    first, unused = torch.autograd.grad(out[0].sum()*0, (x, y), allow_unused=True)
    assert torch.equal(first, torch.zeros_like(x)) and unused is None
    z = torch.arange(4., dtype=torch.float64, requires_grad=True)
    result = move(z, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        z.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        result.sum().backward()


def test_integer_raw_values_do_not_follow_scoring_cast(monkeypatch):
    monkeypatch.setenv("TIDE_COMMUNICATION_POLICY", "blocking")
    x = torch.tensor([2**53+1, 2**61+7], dtype=torch.int64)
    mapping = mapper([x, torch.ones(2)], torch.device("cpu"), torch.float32)
    assert mapping(x).dtype == torch.int64 and torch.equal(mapping(x), x)


def test_invalid_route_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("TIDE_COMMUNICATION_POLICY", "guessed-peer")
    with pytest.raises(ValueError, match="TIDE_COMMUNICATION_POLICY"):
        policy()
    with pytest.raises(ValueError, match="unknown explicit"):
        copy(torch.ones(1), "cpu", selected="bad")
