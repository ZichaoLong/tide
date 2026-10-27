"""Allocation §4.1/6: complete region rows, true boundaries and residuals.

Existing independent KKT/finite-difference tests anchor the scalar definition;
these compare simultaneous numeric solves without replaying scalar allocation.
"""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.allocation_batch import solve, NAMES
from tidegraph.weighted.entropy_allocation import allocate
from tidegraph.weighted.regularizers import Regularizer


def native_rows(t, caps, low, high, tau, *, regularizer, ent_alpha, score_shift):
    import _tide_native as core
    from tidegraph.weighted.allocation import Allocation
    return [Allocation(*r) for r in core.weighted_allocation_batch(
        [(t[i], caps[i], low[i], high[i], tau, score_shift[i]) for i in range(len(t))], regularizer, ent_alpha)]


@pytest.mark.parametrize("kind,alpha", [("quad", None), ("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
@pytest.mark.parametrize("shifted", [False, True])
def test_allocation_complete_rows_and_branches(dtype, kind, alpha, shifted):
    generator = torch.Generator().manual_seed(1703)
    scores = torch.randn(9, 4, generator=generator, dtype=dtype)
    caps = .1+.6*torch.rand(9, 4, generator=generator, dtype=dtype)
    caps[0] = 0; caps[1, 1] = 0; caps[4, 2] = 0
    lower = caps.sum(1)*.31; upper = caps.sum(1)*.75
    lower[1] = upper[1] = caps[1].sum()
    lower[2] = upper[2] = 0
    upper[3] = lower[3]
    lower[5] = 0; upper[5] = 5
    scores[6] = -20; scores[7] = 20; scores[8] = 0
    tau = torch.linspace(.4, 1.3, 9, dtype=dtype)
    shift = torch.linspace(-.6, .3, 9, dtype=dtype) if shifted else None
    expected = [allocate(scores[i], caps[i], lower[i], upper[i], tau[i], regularizer=kind, ent_alpha=alpha,
                         score_shift=None if shift is None else shift[i]) for i in range(9)]
    actual = solve(scores, caps, lower, upper, tau, Regularizer(kind, alpha), shift)
    for i, row in enumerate(expected):
        equivalent(row.controls, actual.controls[i]); equivalent(row.multiplier, actual.multiplier[i])
        assert row.constraint == NAMES[int(actual.branch[i])]
        equivalent(row.free, actual.free[i])
        assert actual.residual[i] <= 64*torch.finfo(dtype).eps*4


def test_quad_batch_literal_and_tied_zero_caps(dtype):
    t = torch.tensor([[2., 1., 100.], [1., 1., -1.]], dtype=dtype)
    caps = torch.tensor([[1., 1., 0.], [.8, .8, .9]], dtype=dtype)
    total = torch.tensor([1.4, .6], dtype=dtype); tau = torch.full((2,), 2., dtype=dtype)
    result = solve(t, caps, total, total, tau, Regularizer("quad"))
    equivalent(result.controls, t.new_tensor([[.95, .45, 0.], [.3, .3, 0.]]))
    assert (result.branch == 5).all()


def test_batch_positive_support_underflow_is_rejected():
    scores = torch.tensor([[-1000., 0.], [1., 2.]], dtype=torch.float32)
    with pytest.raises(ArithmeticError, match="support underflowed"):
        solve(scores, torch.ones_like(scores), torch.zeros(2), torch.ones(2)*2, torch.ones(2), Regularizer("bern"))


@pytest.mark.parametrize("kind,alpha", [("quad", None), ("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_allocation_batch_independent_vjp_roots(dtype, kind, alpha, native):
    from tidegraph.weighted.allocation_rows import allocate_batch
    from isolated_cases import vjp
    t = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in ([2., 1., -.2], [-.3, .4], [0., 1., -1.], [.1, -.1], [0., 0., 0.])]
    caps = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in ([1., .8, .6], [.4, .7], [0., 0., 0.], [.4, .6], [.2, .5, .7])]
    low = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (.9, 0., 0., 1., .2)]
    high = [low[0], torch.tensor(1.2, dtype=dtype, requires_grad=True), low[2], low[3], torch.tensor(.8, dtype=dtype, requires_grad=True)]
    tau = torch.tensor(.8, dtype=dtype, requires_grad=True)
    shift = [torch.tensor(.1*i, dtype=dtype, requires_grad=True) for i in range(5)]
    roots = dict(zip([f"root.{i}" for i in range(21)], [*t, *caps, *low, *shift, tau]))
    roots["upper.1"], roots["upper.4"] = high[1], high[4]
    expected = [allocate(ti, ci, lo, hi, tau, regularizer=kind, ent_alpha=alpha, score_shift=s)
                for ti, ci, lo, hi, s in zip(t, caps, low, high, shift)]
    execute = native_rows if native else allocate_batch
    actual = execute(t, caps, low, high, tau, regularizer=kind, ent_alpha=alpha, score_shift=shift)
    equivalent(expected, actual)
    for x, y in zip(expected, actual):
        assert x.controls.requires_grad == y.controls.requires_grad
        if x.controls.requires_grad:
            equivalent(vjp(x.controls, roots), vjp(y.controls, roots))
            equivalent(vjp(x.controls, roots, zero=True), vjp(y.controls, roots, zero=True))
    def loss(rows):
        return sum(row.controls.square().sum() for row in rows)
    equivalent(vjp(loss(expected), roots), vjp(loss(actual), roots))
    assert allocate_batch([], [], [], [], regularizer=kind, ent_alpha=alpha) == []
