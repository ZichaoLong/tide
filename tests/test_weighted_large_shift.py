"""Allocation §4.1.1: constrained maps cancel a common shift before rounding."""
from pathlib import Path
import sys
import pytest
import torch
from tidegraph.weighted.entropy_allocation import allocate
from tidegraph.weighted.allocation_rows import allocate_batch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from diagnose_allocation_failure import diagnose


@pytest.mark.parametrize("kind,alpha", [("quad", None), ("bern", None), ("ent", 1.), ("ent", 1.5), ("ent", 2.)])
@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
def test_large_common_shift_keeps_independent_local_solution(dtype, kind, alpha, backend):
    s = torch.tensor([.2, .4], dtype=dtype, requires_grad=True)
    mu = torch.ones_like(s, requires_grad=True)
    b, tau, eta = [s.new_tensor(x, requires_grad=True) for x in (.7, 1., 1e30)]
    if backend == "python":
        c = allocate(s, mu, b, b, tau, regularizer=kind, ent_alpha=alpha, score_shift=eta).controls
    elif backend == "python-batch":
        c = allocate_batch([s], [mu], b, b, tau, regularizer=kind, ent_alpha=alpha, score_shift=eta)[0].controls
    else:
        import _tide_native as core
        c = (core.weighted_allocate_shifted(s, mu, b, b, tau, kind, alpha, eta) if backend == "native" else
             core.weighted_allocation_batch([(s, mu, b, b, tau, eta)], kind, alpha)[0][0])
    # The independent optimization uses the original small scores: the common
    # -eta*T term is constant, avoiding any shared rounded subtraction.
    values = dict(scores=s.detach().tolist(), caps=mu.detach().tolist(), lower=[float(b.detach())],
                  upper=[float(b.detach())], temperature=[1.], controls=c.detach().tolist())
    actual = diagnose(dict(schema="tide-allocation-failure-v1", row=0, time=None, nodes=[],
        tensors={k: dict(values=v) for k, v in values.items()}, details=dict(regularizer=kind, alpha=str(alpha))))
    expected = s.new_tensor([float(x) for x in actual["controls"]])
    torch.testing.assert_close(c, expected, atol=8*torch.finfo(dtype).eps, rtol=0)
    gradient = torch.autograd.grad(c, eta, s.new_tensor([2., -1.]))[0]
    assert gradient is not None and gradient == 0


@pytest.mark.parametrize("backend", ["python", "python-batch", "native", "native-batch"])
def test_collapsed_quadratic_transition_is_explicit_refusal(backend):
    from test_weighted_quad_offset import evaluate
    s = torch.zeros(2, dtype=torch.float32)
    tiny = torch.nextafter(s.new_tensor(0.), s.new_tensor(1.))
    mu, b, tau, eta = torch.stack((2*tiny, 2*tiny)), tiny, s.new_tensor(1e-38), s.new_tensor(0.)
    with pytest.raises((ArithmeticError, RuntimeError), match="transition interval underflowed"):
        evaluate(backend, s, mu, b, tau, eta)
