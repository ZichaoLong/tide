"""Allocation §S.2 quad slack closed form: true zero versus positive underflow.

The exact inputs are dyadic rationals; the positive second solution is tiny/2.
Checking the closed form avoids sharing the candidate's residual calculation.
"""
import pytest
import torch
from tidegraph.weighted.allocation import allocate_quad
from tidegraph.weighted.allocation_rows import allocate_batch


def solve(scores, caps, tau, native, batch, lower=1., upper=2.):
    if native:
        import _tide_native as core
        if batch:
            return core.weighted_allocation_batch([(scores, caps, scores.new_tensor(lower),
                scores.new_tensor(upper), scores.new_tensor(tau), None)], "quad", None)[0][0]
        return core.weighted_allocate(scores, caps, scores.new_tensor(lower), scores.new_tensor(upper),
                                      scores.new_tensor(tau), "quad", None)
    if batch:
        return allocate_batch([scores], [caps], lower, upper, tau)[0].controls
    return allocate_quad(scores, caps, lower, upper, tau).controls


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("batch", [False, True], ids=["scalar", "batch"])
def test_quad_mathematical_support_underflow_refused(dtype, native, batch):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
    scores = torch.tensor([4., tiny], dtype=dtype)
    # Exact slack solution: [1, tiny/2], and tiny/2 is a positive real number.
    assert tiny > 0 and (scores[1]/2).item() == 0
    with pytest.raises((ArithmeticError, RuntimeError), match="support underflowed"):
        solve(scores, torch.ones_like(scores), 2., native, batch)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("batch", [False, True], ids=["scalar", "batch"])
def test_quad_representable_support_and_mathematical_zeros(dtype, native, batch):
    tiny = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
    scores = torch.tensor([4., 2*tiny, 0., -tiny, tiny], dtype=dtype)
    caps = torch.tensor([1., 1., 1., 1., 0.], dtype=dtype)
    controls = solve(scores, caps, 2., native, batch)
    assert torch.equal(controls, scores.new_tensor([1., tiny, 0., 0., 0.]))
    controls = solve(scores, caps, 2., native, batch, lower=0., upper=0.)
    assert torch.equal(controls, torch.zeros_like(scores))
