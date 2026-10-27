"""0770593 allocation §4.2–4.4/§6.3: independent formulas and branch adjoints.

The CPU source adapter is neither an independent implementation nor GPU proof.
Anchors below use plain exp quotients, LP vertices and literal branch derivatives.
"""
import itertools
import math
import pytest
import torch
from resident_sending_adapter import Run, build
from tidegraph.weighted import attention_send, proportional, total_zero, hard_selection


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build(tmp_path_factory.mktemp("resident-sending"))


CONFIGS = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 2, 0), (2, 0, 0), (3, 0, 1), (4, 0, 0), (5, 0, 2)]


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def run(lib, s, mu, w=None, total=.3, tau=.7, kind=0, base=0, k=0, ids=None):
    s, mu = torch.tensor([s]), torch.tensor([mu])
    w = mu/2 if w is None else torch.tensor([w])
    return Run(lib, s, mu, w, torch.tensor([[total, tau]]),
               [list(range(s.shape[1])) if ids is None else ids], [s.shape[1]], kind, base, k)


@pytest.mark.parametrize("kind,base,k", CONFIGS)
def test_complete_rows_actual_vjp_and_absent_roots(library, kind, base, k):
    s = torch.tensor([[.2, -.3, .1, float("nan")]]).repeat(4, 1)
    mu = torch.tensor([[.15, .35, .2, -42.]]).repeat(4, 1);w = mu*.5
    params = torch.tensor([[.3, .7]]).repeat(4, 1)
    ids = [[2**60+7, 2**60+1, 2**60+3, -1]]*4
    r = Run(library, s, mu, w, params, ids, [3, 2, 0, 3], kind, base, k)
    g = torch.tensor([[.3, -.4, .7, float("nan")]]).repeat(4, 1)
    dv, ds, flags = r.backward(g, [True, True, True, False])
    assert r.error.item() == 0 and not flags[2:].any()
    assert dv[2:].eq(0).all() and ds[2:].eq(0).all()
    for row, n in ((0, 3), (1, 2)):
        roots = [x.detach().clone().requires_grad_() for x in (s[row, :n], mu[row, :n], w[row, :n], *params[row])]
        score, cap, recv, total, tau = roots
        if kind <= 2:
            mode = ["direct_attention", "unit_attention", "pooled_attention"][kind]
            b = ["mu", "received", "uniform"][base]
            expected = attention_send(score, cap, recv, mode=mode, base=b, temperature=tau)
        elif kind == 3:expected = proportional(cap, k)
        elif kind == 4:expected = total_zero(score, cap, total, ids[row][:n])
        else:expected = hard_selection(score, cap, k, ids[row][:n])
        close(r.controls[row, :n], expected)
        wanted = torch.autograd.grad(expected, roots, g[row, :n], allow_unused=True)
        for j, (actual, reference) in enumerate(zip([*dv[row, :, :n], *ds[row]], wanted)):
            assert bool(flags[row, j]) == (reference is not None)
            if reference is not None:close(actual, reference)
        assert r.controls[row, n:].eq(0).all() and dv[row, :, n:].eq(0).all()


@pytest.mark.parametrize("kind,base,k", CONFIGS[:5])
def test_attention_independent_quotient(library, kind, base, k):
    roots = [torch.tensor(x, requires_grad=True) for x in ([.2, -.3, .1], [.15, .35, .2], [.1, .2, .15], .7)]
    s, mu, w, tau = roots
    a = [mu, w, torch.ones_like(mu)][base]
    weighted = a*torch.exp(s/tau);p = weighted/weighted.sum()
    expected = p*mu if kind == 0 else p if kind == 1 else p*mu.sum()
    g = torch.tensor([.2, -.4, .7])
    reference = torch.autograd.grad(expected, roots, g, allow_unused=True)
    r = run(library, s.tolist(), mu.tolist(), w.tolist(), kind=kind, base=base)
    dv, ds, flags = r.backward(g[None]);assert r.error.item() == 0
    close(r.controls[0], expected)
    for actual, ref in zip([*dv[0], ds[0, 1]], reference):
        if ref is not None:close(actual, ref)
    direction = torch.tensor([.17, -.31, .23]);h = .002
    plus = run(library, (s+h*direction).tolist(), mu.tolist(), w.tolist(), kind=kind, base=base)
    minus = run(library, (s-h*direction).tolist(), mu.tolist(), w.tolist(), kind=kind, base=base)
    diff = ((plus.controls[0]-minus.controls[0])*g).sum()/(2*h)
    torch.testing.assert_close(diff, dv[0, 0]@direction, atol=2e-5, rtol=1e-3)


def test_zero_base_right_derivative_and_declared_zero_corner(library):
    r = run(library, [0., math.log(2), math.log(3)], [1., 0., 1.], kind=1, tau=1.)
    close(r.controls[0], torch.tensor([.25, 0., .75]))
    v, s, f = r.backward(torch.tensor([[0., 1., 0.]]))
    close(v[0, 1], torch.tensor([0., .5, 0.]));assert v[0, 0].eq(0).all()
    r = run(library, [0., 0.], [0., 0.]);v, s, f = r.backward(torch.ones(1, 2))
    assert r.error.item() == 0 and f[0].tolist() == [True, True, False, False, True]
    assert v.eq(0).all() and s.eq(0).all()
    uniform = run(library, [0., 0.], [0., 0.], kind=1, base=2)
    close(uniform.controls[0], torch.tensor([.5, .5]))


def test_pooled_can_exceed_cap_and_fractional_fill_is_not_topk(library):
    pooled = run(library, [math.log(9), 0.], [.1, .1], kind=2, tau=1.)
    close(pooled.controls[0], torch.tensor([.18, .02]))
    r = run(library, [3., 2., 1., 0.], [.3]*4, kind=4, total=1.)
    close(r.controls[0], torch.tensor([.3, .3, .3, .1]))
    v, s, f = r.backward(torch.tensor([[0., 0., 0., 1.]]))
    close(v[0, 1], torch.tensor([-1., -1., -1., 0.]));assert s[0, 0] == 1 and not f[0, 0]
    hard = run(library, [3., 2., 1., 0.], [.3]*4, kind=5, k=1)
    close(hard.controls[0], torch.tensor([.3, 0., 0., 0.]))


def test_sorted_fill_independent_lp_vertices_and_exact_cap_endpoint(library):
    scores, caps = [-.2, 1.3, .4], [.7, .6, .8];b = .95
    r = run(library, scores, caps, kind=4, total=b);value = float(r.controls[0]@torch.tensor(scores))
    for free in range(3):
        others = [i for i in range(3) if i != free]
        for mask in itertools.product((0, 1), repeat=2):
            vertex = [0.]*3
            for i, bit in zip(others, mask):vertex[i] = bit*caps[i]
            vertex[free] = b-sum(vertex)
            if 0 <= vertex[free] <= caps[free]:assert value+1e-6 >= sum(x*y for x, y in zip(scores, vertex))
    r = run(library, [1., 1., 1.], [.1, .3, .6], kind=4, total=1., ids=[2**60+9, 2**60+2, 2**60+4])
    close(r.controls[0], torch.tensor([.1, .3, .6]));v, s, f = r.backward(torch.ones(1, 3))
    assert v[0, 1].eq(1).all() and s.eq(0).all() and f[0, 3]


def test_stable_ids_hard_fallback_and_proportional_origin(library):
    r = run(library, [-1., -1., -1.], [.2, .4, .5], kind=5, k=2, ids=[2**60+9, 2**60+2, 2**60+4])
    close(r.controls[0], torch.tensor([0., .4, 0.]));v, s, f = r.backward(torch.ones(1, 3))
    close(v[0, 1], torch.tensor([0., 1., 0.]));assert f[0].tolist() == [False, True, False, False, False]
    for caps in ([0., 0.], [.4, .6], [.6, .9]):
        r = run(library, [float("nan")]*2, caps, kind=3, k=1)
        v, _, _ = r.backward(torch.tensor([[.3, -.7]]));assert r.error.item() == 0
        if sum(caps) <= 1:close(v[0, 1], torch.tensor([.3, -.7]))
        else:
            mu = torch.tensor(caps);g = torch.tensor([.3, -.7]);close(v[0, 1], (g-(g*mu).sum()/mu.sum())/mu.sum())


@pytest.mark.parametrize("case", ["underflow", "duplicate", "cap", "received", "total", "tau", "nonfinite", "overflow_vjp"])
def test_refusals_keep_error_identity(library, case):
    args = dict(s=[.2, -.3], mu=[.4, .6])
    if case == "underflow":args.update(s=[-10000., 0.])
    if case == "duplicate":args.update(kind=4, ids=[1, 1])
    if case == "cap":args.update(mu=[1.1, .6])
    if case == "received":args.update(w=[.5, .1])
    if case == "total":args.update(kind=4, total=2.)
    if case == "tau":args.update(tau=0.)
    if case == "nonfinite":args.update(s=[float("nan"), 0.])
    if case == "overflow_vjp":args.update(s=[0., 1000.], mu=[1., 0.], kind=1)
    r = run(library, **args)
    if case == "overflow_vjp":
        assert r.error.item() == 0;r.backward(torch.tensor([[0., 1.]]))
    assert r.error.item() == (21 if case == "underflow" else 23 if case == "overflow_vjp" else 20)
    prior = r.error.clone();r.errors.zero_();r.finish();assert torch.equal(prior, r.error)


def test_metadata_inactive_roots_and_unused_fields(library):
    s=torch.tensor([[.1, .2]]);mu=torch.tensor([[.4, .6]]);p=torch.tensor([[.3, .7]])
    r=Run(library,s,mu,mu/2,p,[[0,1]],[3]);assert r.error.item()==2
    r=Run(library,s,mu,mu/2,p,[[0,1]],[2])
    v,scalars,flags=r.backward(torch.full_like(s,float("nan")),[False])
    assert r.error.item()==0 and not flags.any() and v.eq(0).all() and scalars.eq(0).all()
    r.backward(torch.full_like(s,float("nan")));assert r.error.item()==23
    # Proportional has no mathematical score, received, B, tau or ID input.
    poison=torch.full_like(s,float("nan"))
    r=Run(library,poison,mu,poison,poison,[[-1,-1]],[2],3,0,1)
    assert r.error.item()==0;close(r.controls,mu)


@pytest.mark.parametrize("kind,base,k",[(0,2,0),(1,1,0),(2,2,0),(0,0,1),(3,0,0),(4,0,1),(5,0,0),(3,1,1)])
def test_inapplicable_configuration_refused(library,kind,base,k):
    r=run(library,[.1,.2],[.4,.6],kind=kind,base=base,k=k)
    assert r.error.item()==20
