"""Diagnostic observations preserve actual values, gradients and hooks."""
from pathlib import Path
import sys
import torch
import pytest
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


def test_native_observer_actual_results_gradient_budget_and_cleanup():
    import _tide_native as core
    s = torch.tensor([.2, .4], requires_grad=True)
    mu, bound, tau = torch.ones_like(s), s.new_tensor(.5), s.new_tensor(1.)
    with observe_allocations(native=core, max_rows=2) as records:
        with pytest.raises(ValueError, match="active observer"):
            core.begin_allocation_observation(2, 10000)
        a = core.weighted_allocate_quad(s, mu, bound, bound, tau)
        b = core.weighted_allocation_batch([(s, mu, bound, bound, tau, None)]*2, "quad", None)
        torch.testing.assert_close(torch.autograd.grad(a[0], s)[0], torch.tensor([.5, -.5]))
        torch.testing.assert_close(torch.autograd.grad(b[0][0][0], s)[0], torch.tensor([.5, -.5]))
    assert records["native_limits"]["dropped"] == 1
    assert len(records["native"]) == 2
    assert records["native"][0]["tensors"]["controls"]["values"] == a.detach().tolist()
    assert [r["details"]["solver"] for r in records["native"]] == ["native-scalar", "native-batch"]
    with pytest.raises(ValueError, match="no active"):
        core.end_allocation_observation()
    with observe_allocations(native=core, max_bytes=1) as limited:
        core.weighted_allocate_quad(s, mu, bound, bound, tau)
    assert limited["native"] == [] and limited["native_limits"]["dropped"] == 1


def test_native_full_graph_observation_does_not_change_actual_return():
    import _tide_native as core
    from tidegraph import Continuation
    from tidegraph.compare import equivalent
    from tidegraph.native import Native
    from test_weighted_execution import setup
    binding, _, xs, stop = setup(torch.float64, "hb")
    execute = Native(binding.graph, binding.model, algorithm="greedy", packed=True).run
    q = Continuation(binding.graph.identity, 1)
    with torch.no_grad():
        original = execute(q, xs, stop, sealed_until=stop)
        with observe_allocations(native=core) as records:
            actual = execute(q, xs, stop, sealed_until=stop)
    equivalent(original, actual, atol=0, rtol=0)
    assert records["native"] and records["native_limits"]["dropped"] == 0
