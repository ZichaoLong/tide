"""Read primitive roots independent of graph selection and shared parameters."""
import pytest
import torch
from tidegraph.read_vjp import _Read


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("scoring", [torch.float32, torch.float64])
@pytest.mark.parametrize("connected_zero", [False, True])
def test_read_independent_rows_and_precision(dtype, implementation, norm, scoring, connected_zero):
    rows = [torch.tensor(x, dtype=dtype, requires_grad=i != 2)
            for i, x in enumerate(([.25, -.5, 2.], [0., 0., 0.], [2., 1., -.25], [-.3, .7, 1.]))]
    weight = torch.tensor([.2, -.7, .4], dtype=dtype, requires_grad=True)
    if implementation == "native":
        import _tide_native
        actual = _tide_native.read_vjp_probe(rows, weight, norm, torch.empty((), dtype=scoring))
    else:
        actual = _Read.apply(norm, scoring, torch.device("cpu"), weight, *rows)
    expected = [torch.linalg.vector_norm(row.to(scoring)) if norm
                else (row.to(scoring)*weight.to(scoring)).sum() for row in rows]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
        assert a.requires_grad == e.requires_grad
    # Row 3 never participates; row 1 has a zero norm and an explicit cotangent.
    leaves = [rows[0], rows[1], rows[3], weight]
    roots = lambda values: values[0] * (0 if connected_zero else 1.3) + values[1]*.7
    a = torch.autograd.grad(roots(actual), leaves, allow_unused=True)
    e = torch.autograd.grad(roots(expected), leaves, allow_unused=True)
    for av, ev in zip(a, e):
        assert (av is None) == (ev is None)
        if av is not None:
            torch.testing.assert_close(av, ev, atol=2e-7, rtol=2e-6)
    assert a[2] is None


def test_python_read_no_scalar_replay_and_one_finite_check(monkeypatch):
    from types import SimpleNamespace
    from tidegraph.content import Content
    from tidegraph.readout import LinearRead, ReadInput, evaluate
    program = LinearRead()
    weights = SimpleNamespace(read_program=program, read=torch.ones(3, requires_grad=True))
    requests = [ReadInput(None, i, Content(torch.ones(3, requires_grad=True))) for i in range(9)]
    def forbidden(*args):
        raise AssertionError("scalar replay")
    monkeypatch.setattr(program, "step", forbidden)
    original, shapes = torch.isfinite, []
    def count(value):
        shapes.append(tuple(value.shape))
        return original(value)
    monkeypatch.setattr(torch, "isfinite", count)
    outputs = evaluate(weights, requests, packed=True)
    outputs[0].backward()
    assert shapes == [(9,)]
    assert requests[1].content.value.grad is None


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_read_saved_row_version_check(implementation):
    row = torch.ones(3, requires_grad=True)
    weight = torch.ones(3, requires_grad=True)
    if implementation == "native":
        import _tide_native
        score = _tide_native.read_vjp_probe([row], weight, False, row.detach())[0]
    else:
        score = _Read.apply(False, row.dtype, row.device, weight, row)[0]
    with torch.no_grad():
        row.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        score.backward()
