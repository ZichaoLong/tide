from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.token_window import token_inputs


def convert(implementation, outputs, q, layers=3, stop=3, body_cut=9, port=0):
    if implementation == "python":
        result = token_inputs(outputs, q, layers, stop, body_cut=body_cut, port=port)
    else:
        import _tide_native as core
        native = core.Continuation(); native.cut = q.cut; native.batch_size = q.batch_size; native.ledger = q.ledger
        result = core.token_inputs([core.Output(*row) for row in outputs], native, layers, stop, body_cut, port)
        assert native.cut == q.cut and native.ledger == q.ledger
    return [(x.batch, x.port, x.position, x.time, x.value) for x in result]


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_phase_identity_missing_samples_zeros_and_contiguous_positions(dtype, implementation):
    q = Continuation("readout", 3, cut=1, ledger={(0, 2): (0, 0), (1, 0): (0, 0)})
    before = q.fork()
    values = [torch.tensor([x], dtype=dtype, requires_grad=True) for x in (1., 0., 3., 5.)]
    # Port 4 is unrelated; token 1 has no row for sample 1, phase 1 is absent.
    body = [(0, 5, 0, values[0]), (0, 6, 0, values[1]), (1, 6, 0, values[2]),
            (0, 8, 0, values[3]), (2, 4, 4, values[0])]
    actual = convert(implementation, list(reversed(body)), q)
    assert [row[:4] for row in actual] == [(0, 2, 1, 1), (0, 0, 0, 2), (0, 2, 2, 2), (1, 0, 1, 2)]
    assert [row[4].item() for row in actual] == [1., 0., 5., 3.]
    assert actual[0][4].data_ptr() == values[0].data_ptr()
    grads = torch.autograd.grad(actual[0][4].sum(), values, allow_unused=True)
    equivalent(grads, (torch.ones_like(values[0]), None, None, None))
    equivalent(q, before)


@pytest.mark.parametrize("implementation", ["python", "native"])
@pytest.mark.parametrize("bad", ["layers", "seal", "overflow", "duplicate", "batch", "time", "empty-token", "ledger", "counter"])
def test_token_window_rejects_invalid_metadata(dtype, implementation, bad):
    q = Continuation("readout", 2, cut=1)
    x = torch.ones(2, dtype=dtype); body = [(0, 3, 0, x), (0, 6, 0, x)]
    kwargs = {}
    if bad == "layers": kwargs["layers"] = 0
    if bad == "seal": kwargs["body_cut"] = 8
    if bad == "overflow": kwargs.update(layers=2**62, stop=3)
    if bad == "duplicate": body += [body[0]]
    if bad == "batch": body[0] = (2, 3, 0, x)
    if bad == "time": body[0] = (0, 2, 0, x)
    if bad == "empty-token": body.pop()
    if bad == "ledger": q.ledger = {(0, 0): (0, 1)}
    if bad == "counter": q.ledger = {(0, 0): (2**63-1, 0)}
    with pytest.raises(ValueError): convert(implementation, body, q, **kwargs)


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_empty_cut_and_chunked_conversion(dtype, implementation):
    x = torch.ones(2, dtype=dtype)
    q = Continuation("readout", 1)
    assert not convert(implementation, [], q, stop=0, body_cut=0)
    body = [(0, 0, 0, x), (0, 5, 0, x), (0, 6, 0, x)]
    whole = convert(implementation, body, q)
    first = convert(implementation, body[:2], q, stop=2, body_cut=6)
    q = replace(q, cut=2, ledger={(b, p): (pos, t) for b, p, pos, t, _ in first})
    equivalent(whole, first+convert(implementation, body[2:], q))
