"""0770593 allocation §6/S.2: column formulas, real branches and transpose roots."""
import pytest
import torch
from tidegraph.weighted import allocate_columns, region_columns
from tidegraph.compare import equivalent


def execute_allocation(native, scores, caps, low, high, tau, kind="quad", alpha=None, shift=None):
    if native:
        import _tide_native as core
        return core.weighted_allocation_columns(scores, caps, low, high, tau, kind, alpha, shift)[0]
    return allocate_columns(scores, caps, low, high, tau, regularizer=kind, ent_alpha=alpha, score_shift=shift).controls


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_column_literal_solution_and_shared_temperature(dtype, native):
    s = torch.tensor([[2., 1., 100.], [1., 1., -1.]], dtype=dtype, requires_grad=True)
    c = torch.tensor([[1., 1., 0.], [.8, .8, .9]], dtype=dtype, requires_grad=True)
    b = torch.tensor([1.4, .6], dtype=dtype, requires_grad=True)
    tau = torch.tensor(2., dtype=dtype, requires_grad=True)
    shift = torch.tensor(.25, dtype=dtype, requires_grad=True)
    actual = execute_allocation(native, s, c, b, b, tau, shift=shift)
    equivalent(actual, s.new_tensor([[.95, .45, 0.], [.3, .3, 0.]]))
    bar = s.new_tensor([[1., 3., 7.], [2., -4., 5.]])
    grads = torch.autograd.grad(actual, (s, c, b, tau, shift), bar)
    equivalent(grads[0], s.new_tensor([[-.5, .5, 0.], [1.5, -1.5, 0.]]))
    equivalent(grads[1], torch.zeros_like(c))
    equivalent(grads[2], b.new_tensor([2., -1.]))
    equivalent(grads[3], tau.new_tensor(.25))
    equivalent(grads[4], shift.new_zeros(()))


@pytest.mark.parametrize("kind,alpha", [("quad", None), ("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_column_directional_derivative(kind, alpha, native):
    # Smooth interior solution; finite differences independently differentiate
    # the returned exact solution, not the implementation's solver iteration.
    s = torch.tensor([[.12, -.07, .05], [.03, .09, -.02]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[.7, .8, .9], [.8, .7, .9]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.5, .6], dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(.9, dtype=torch.float64, requires_grad=True)
    shift = torch.tensor([.07, -.03], dtype=torch.float64, requires_grad=True)
    xs = s, c, b, tau, shift
    ds = tuple(torch.linspace(-.1, .17, x.numel(), dtype=x.dtype).reshape(x.shape) for x in xs)
    bar = s.new_tensor([[.3, -.8, .7], [-.1, .4, .9]])
    def objective(values):
        a, m, target, t, eta = values
        return (execute_allocation(native, a, m, target, target, t, kind, alpha, eta)*bar).sum()
    gradients = torch.autograd.grad(objective(xs), xs)
    analytic = sum((g*d).sum() for g, d in zip(gradients, ds))
    h = 2e-5
    with torch.no_grad():
        numeric = (objective(tuple(x+h*d for x, d in zip(xs, ds)))-objective(tuple(x-h*d for x, d in zip(xs, ds))))/(2*h)
    torch.testing.assert_close(analytic, numeric, atol=2e-8, rtol=2e-6)


def test_column_structural_support_is_independent_of_zero_bar():
    s = torch.tensor([[.1, .4], [.3, -.2]], dtype=torch.float64, requires_grad=True)
    caps = torch.tensor([[.5, .7], [.6, .8]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.2, .7], dtype=torch.float64, requires_grad=True)
    a = allocate_columns(s, caps, b, b)
    with torch.no_grad():
        cap_only = a.pullback(torch.zeros_like(s), torch.tensor([True, False]))
        normal_only = a.pullback(torch.zeros_like(s), torch.tensor([False, True]))
        unused = a.pullback(torch.ones_like(s), torch.tensor([False, False]))
    assert cap_only[0] is None and cap_only[1] is not None
    assert normal_only[0] is not None and normal_only[2] is not None and normal_only[3] is None
    assert all(x is None for x in unused)
    assert torch.count_nonzero(normal_only[0]) == 0
    with torch.no_grad():
        s.add_(.1)
        with pytest.raises(RuntimeError, match="changed"):
            a.pullback(torch.ones_like(s))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_column_constants_versions_and_first_order(native):
    s = torch.tensor([[.2, .3]], dtype=torch.float64, requires_grad=True)
    c = torch.ones_like(s, requires_grad=True)
    bound, tau = s.new_tensor(2.), s.new_tensor(1.)
    caps = execute_allocation(native, s, c, bound, bound, tau)
    ds, dc = torch.autograd.grad(caps.sum(), (s, c), allow_unused=True)
    assert ds is None
    equivalent(dc, torch.ones_like(c))
    zeros = execute_allocation(native, s, c, bound*0, bound*0, tau)
    assert not zeros.requires_grad
    actual = execute_allocation(native, s, c, bound*.2, bound*.2, tau)
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(actual.sum(), s, create_graph=True)
    actual = execute_allocation(native, s, c, bound*.2, bound*.2, tau)
    with torch.no_grad():
        s.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(actual.sum(), s)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("case", range(12))
def test_region_columns_complete_scoring_sending_and_parameter_roots(dtype, native, case):
    from test_weighted_region_batch import fixture
    from isolated_cases import vjp
    from tidegraph.region import RegionInput
    p, requests = fixture(dtype, case)
    requests = requests[:4]  # uniform real candidate count, no fake padding
    fields = {name: torch.stack([torch.stack([d[name] for _, d in r.candidates]) for r in requests]).detach().requires_grad_()
              for name in requests[0].candidates[0][1]}
    nodes = torch.tensor([[v for v, _ in r.candidates] for r in requests], dtype=torch.int64)
    scalar_rows = [RegionInput(r.history, r.time, tuple((v, {n: x[i, j] for n, x in fields.items()})
                   for j, (v, _) in enumerate(r.candidates)), r.layout, dtype, r.payload_device) for i, r in enumerate(requests)]
    expected = torch.stack([torch.stack(list(p.step(r).controls.values())) for r in scalar_rows])
    if native:
        import _tide_native as core
        from tidegraph.weighted.native import region_config
        w = core.RegionWeights(); w.extra = dict(p.named_parameters(remove_duplicate=False))
        core.configure_weighted_region(w, region_config(core, p.config))
        actual = core.weighted_region_columns(w, fields, nodes)
    else:
        actual = region_columns(p, fields, nodes).controls
    equivalent(actual, expected)
    roots = dict(p.named_parameters()) | fields
    equivalent(vjp(actual.square(), roots), vjp(expected.square(), roots))
    equivalent(vjp(actual, roots, zero=True), vjp(expected, roots, zero=True))
