"""Diagnostic observations preserve actual values, gradients and hooks."""
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from weighted_allocation_observer import observe_allocations


def test_allocation_observer_preserves_returns_gradients_and_restores():
    from tidegraph.weighted import programs, region_batch
    original = programs.allocate, region_batch.allocate_batch
    s = torch.tensor([.2, .4], requires_grad=True)
    mu = torch.ones_like(s)
    with observe_allocations() as records:
        a = programs.allocate(s, mu, .5, .5)
        b = region_batch.allocate_batch([s], [mu], .5, .5)[0]
        torch.testing.assert_close(a.controls, b.controls)
        torch.testing.assert_close(torch.autograd.grad(a.controls[0], s)[0], torch.tensor([.5, -.5]))
        torch.testing.assert_close(torch.autograd.grad(b.controls[0], s)[0], torch.tensor([.5, -.5]))
    assert (programs.allocate, region_batch.allocate_batch) == original
    assert len(records["scalar"]) == len(records["batch"]) == 1
    assert records["scalar"][0]["tensors"]["scores"]["values"] == s.detach().tolist()
    assert records["batch"][0]["free"] == [True, True]
