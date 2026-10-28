"""Allocation §6/S.2: independently solved affine faces at large offsets.

These checks use the constrained quadratic closed form, not another allocator.
The requested dtype, original feasibility tolerance and positive support stay fixed.
"""
import pytest
import torch
from tidegraph.weighted.allocation import allocate_quad
from tidegraph.weighted.allocation_rows import allocate_batch


def evaluate(backend, s, mu, bound, tau, eta):
    if backend == "python":
        return allocate_quad(s, mu, bound, bound, tau, score_shift=eta).controls
    if backend == "python-batch":
        return allocate_batch([s], [mu], bound, bound, tau, score_shift=eta)[0].controls
    import _tide_native as core
    if backend == "native":
        return core.weighted_allocate_shifted(s, mu, bound, bound, tau, "quad", None, eta)
    return core.weighted_allocation_batch([(s, mu, bound, bound, tau, eta)], "quad", None)[0][0]


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
@pytest.mark.parametrize("offset", [10000., 1048576.])
def test_quad_large_offset_closed_form_and_independent_roots(dtype, backend, offset):
    # The two free coordinates differ by .5; the third is capped, fourth zero.
    # c=( (B-.25+.5)/2, (B-.25-.5)/2, .25, 0 ), independent of the common shift.
    s = torch.tensor([offset+.5, offset, offset+2, offset-3], dtype=dtype, requires_grad=True)
    mu = s.new_tensor([1., 1., .25, 1.], requires_grad=True)
    bound, tau, eta = [s.new_tensor(v, requires_grad=True) for v in (.95, 1., 64.)]
    c = evaluate(backend, s, mu, bound, tau, eta)
    expected = torch.stack(((bound.detach()+.25)/2, (bound.detach()-.75)/2,
                            s.new_tensor(.25), s.new_tensor(0.)))
    tolerance = 8*torch.finfo(dtype).eps
    torch.testing.assert_close(c, expected, atol=tolerance, rtol=0)
    assert torch.equal(c > 0, s.new_tensor([True, True, True, False], dtype=torch.bool))
    # For root (2,-1,4,7), mean over free coordinates is .5.
    actual = torch.autograd.grad(c, (s, mu, bound, tau, eta), s.new_tensor([2., -1., 4., 7.]))
    wanted = ([1.5, -1.5, 0., 0.], [0., 0., 3.5, 0.], .5, -.75, 0.)
    for a, w in zip(actual, wanted):
        torch.testing.assert_close(a, s.new_tensor(w), atol=tolerance, rtol=0)


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
def test_quad_offset_single_free_and_saved_input_versions(dtype, backend):
    s = torch.tensor([10000., 9999.], dtype=dtype, requires_grad=True)
    mu = torch.ones_like(s, requires_grad=True)
    bound, tau, eta = [s.new_tensor(v, requires_grad=True) for v in (.7, 1., 0.)]
    c = evaluate(backend, s, mu, bound, tau, eta)
    assert torch.equal(c, torch.stack((bound.detach(), s.new_tensor(0.))))
    actual = torch.autograd.grad(c, (s, mu, bound, tau, eta), s.new_tensor([2., -1.]), retain_graph=True)
    for a, w in zip(actual, ([0., 0.], [0., 0.], 2., 0., 0.)):
        assert torch.equal(a, s.new_tensor(w))
    # Saving the numeric response must not remove original root version checks.
    with torch.no_grad():
        s.add_(1.)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        c.sum().backward()
