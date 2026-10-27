"""0770593 allocation §6/S.2: device-source complete rows and exact-map VJP.

Independent anchors: exhaustive quadratic faces, Decimal entropy response,
literal derivatives, finite differences; CPU semantic adapters are not GPU proof.
"""
import pytest
import torch
from resident_weighted_adapter import build, Run
from test_weighted_allocation import enumerate_quad
from test_weighted_entropy_allocation import decimal_reference
from tidegraph.weighted.entropy_allocation import allocate


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("resident-weighted-source"))


RULES = [(0, 1., "quad"), (1, 1., "bern"), (2, 1., "ent"),
         (2, 1.000001, "ent"), (2, 1.5, "ent"), (2, 2., "ent")]


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind,alpha,name", RULES)
def test_resident_allocation_independent_optima_and_padding(library, kind, alpha, name):
    bounds = torch.tensor([[.2, 1., .8, 0.], [1.4, 1.7, .8, 0.], [.6, .6, .8, 0.],
                           [.2, .4, .8, 0.], [0., 3., .8, 0.]])
    scores = torch.tensor([[.4, -.7, 1.1, -.2, float("nan")]]).repeat(5, 1)
    caps = torch.tensor([[.7, .9, .25, 0., -42.]]).repeat(5, 1)
    run = Run(library, scores, caps, bounds, [4]*5, kind, alpha)
    assert run.error.item() == 0
    for row, (lo, hi, tau, _) in enumerate(bounds):
        s, mu = scores[row, :4], caps[row, :4]
        if kind == 0:
            expected = enumerate_quad(s, mu, lo, hi, tau)
        else:
            expected = s.new_tensor(decimal_reference(s, mu, lo, hi, tau, name, float(torch.tensor(alpha))))
        close(run.controls[row, :4], expected)
    assert run.controls[:, 4].eq(0).all() and run.slopes[:, 4].eq(0).all()
    assert (run.statistics[:, 1] < 1e-5).all()


@pytest.mark.parametrize("kind,alpha,name", RULES)
def test_resident_allocation_vjp_independent_roots_and_direction(library, kind, alpha, name):
    scores = torch.tensor([[.2, -.1, .4], [1.2, 1.1, -.6], [-2., -.1, -.4]])
    caps = torch.tensor([[.7, .8, .6], [.4, .8, .7], [.3, .6, .9]])
    bounds = torch.tensor([[.9, .9, .8, .2], [0., 3., 1.3, .1], [.3, 1., .7, 0.]])
    grad = torch.tensor([[.3, -.5, .7], [.6, -.8, -.2], [1.1, .9, -.7]])
    run = Run(library, scores, caps, bounds, [3]*3, kind, alpha)
    assert run.error.item() == 0
    dv, ds, connected = run.backward(grad, [True, False, True])
    assert run.error.item() == 0 and not connected[1].any()
    assert dv[1].eq(0).all() and ds[1].eq(0).all()
    for row in (0, 2):
        roots = [x.clone().requires_grad_() for x in (scores[row], caps[row], *bounds[row])]
        s, mu, lo, hi, tau, shift = roots
        expected = allocate(s, mu, lo, hi, tau, regularizer=name,
                            ent_alpha=float(torch.tensor(alpha)) if kind == 2 else None, score_shift=shift)
        actual = [dv[row, 0], dv[row, 1], *ds[row]]
        gradients = torch.autograd.grad((expected.controls*grad[row]).sum(), roots, allow_unused=True)
        for j, (a, e) in enumerate(zip(actual, gradients)):
            assert bool(connected[row, j]) == (e is not None)
            if e is not None:
                close(a, e)
        # FP32 directional difference on a smooth score branch; derivative is
        # from the actual mathematical response, not bisection iterations.
        direction = torch.tensor([.17, -.31, .23]);h = .002
        plus = scores.clone();minus = scores.clone();plus[row] += h*direction;minus[row] -= h*direction
        a = Run(library, plus, caps, bounds, [3]*3, kind, alpha)
        b = Run(library, minus, caps, bounds, [3]*3, kind, alpha)
        finite_difference = ((a.controls[row]-b.controls[row])*grad[row]).sum()/(2*h)
        torch.testing.assert_close((dv[row, 0]*direction).sum(), finite_difference, atol=5e-5, rtol=1e-3)


def test_resident_quad_literal_map_and_exact_connected_zero(library):
    scores = torch.tensor([[2., 1., 100.], [2., 1., 100.]])
    caps = torch.tensor([[1., 1., 0.], [1., 1., 0.]])
    bounds = torch.tensor([[1.4, 1.4, 2., 0.], [0., 3., 2., 0.]])
    run = Run(library, scores, caps, bounds, [3, 3])
    close(run.controls[0], torch.tensor([.95, .45, 0.]))
    grad = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    vectors, scalars, flags = run.backward(grad)
    assert run.error.item() == 0
    close(vectors[0, 0], torch.tensor([.25, -.25, 0.]))
    close(scalars[0], torch.tensor([.5, 0., -.125, 0.]))
    assert flags[0].tolist() == [True, True, True, False, True, True]
    assert flags[1].tolist() == [True, True, False, True, True, True]
    # A connected constant root has zero score/cap/temperature gradient under
    # an equality constraint. Numerically zero is not structural absence.
    v, s, con = run.backward(torch.ones_like(grad))
    assert v[0].eq(0).all() and s[0, 2:].eq(0).all() and con[0, 0]
    assert s[0, 0] == 1


@pytest.mark.parametrize("kind,alpha,name", RULES)
def test_resident_absent_zero_caps_singleton_and_saturated(library, kind, alpha, name):
    s = torch.tensor([[float("nan"), 1.], [2., -1.], [0., 0.], [0., float("nan")]])
    mu = torch.tensor([[float("nan"), 3.], [.2, .3], [.25, .5], [.4, 3.]])
    b = torch.tensor([[float("nan")]*4, [0., 0., 1., 0.], [.75, 1., 1., 0.], [.3, .3, 1., 0.]])
    run = Run(library, s, mu, b, [0, 2, 2, 1], kind, alpha)
    assert run.error.item() == 0 and run.branches.tolist() == [-1, 0, 1, 5]
    close(run.controls, torch.tensor([[0., 0.], [0., 0.], [.25, .5], [.3, 0.]]))
    v, values, flags = run.backward(torch.ones_like(s))
    assert run.error.item() == 0 and not flags[:2].any()
    assert flags[2].tolist() == [False, True, False, False, False, False]
    close(v[2, 1], torch.ones(2));assert values[3, 0] == 1 and v[3, 0].eq(0).all()


@pytest.mark.parametrize("failure", ["length", "caps", "nan", "bounds", "tau", "overflow"])
def test_resident_allocation_domain_and_sticky_error(library, failure):
    s = torch.tensor([[.3, .4]]);mu = torch.tensor([[.7, .8]])
    b = torch.tensor([[.5, 1., 1., 0.]]);length = 2
    if failure == "length":length = 3
    if failure == "caps":mu[0, 0] = 1.1
    if failure == "nan":s[0, 1] = float("nan")
    if failure == "bounds":b[0, 0] = 2
    if failure == "tau":b[0, 2] = 0
    if failure == "overflow":s[0, 0] = 3e38;b[0, 3] = -3e38
    run = Run(library, s, mu, b, [length])
    assert run.error.item() == (2 if failure == "length" else 20)
    prior = run.error.clone();run.errors.zero_();run.finish();assert torch.equal(prior, run.error)


@pytest.mark.parametrize("kind", [1, 2])
def test_resident_positive_support_underflow_is_refused(library, kind):
    run = Run(library, torch.tensor([[-10000., 0.]]), torch.ones(1, 2),
              torch.tensor([[0., 2., 1., 0.]]), [2], kind)
    assert run.error.item() == 21


def test_resident_rounded_bernoulli_one_retains_positive_slope(library):
    run = Run(library, torch.tensor([[20.]]), torch.ones(1, 1), torch.tensor([[0., 2., 1., 0.]]), [1], 1)
    v, _, flags = run.backward(torch.ones(1, 1), has_shift=False)
    assert run.error.item() == 0 and v[0, 0, 0] > 0 and not flags[0, 5]
    close(v[0, 0, 0], torch.exp(torch.tensor(-20.)))


def test_resident_nonfinite_cotangent_and_absent_root(library):
    run = Run(library, torch.tensor([[.2, .4]]), torch.ones(1, 2), torch.tensor([[.8, .8, 1., 0.]]), [2])
    run.backward(torch.full((1, 2), float("inf")), [False]);assert run.error.item() == 0
    run.backward(torch.full((1, 2), float("inf")), [True]);assert run.error.item() == 23
