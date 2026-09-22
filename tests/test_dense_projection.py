"""Column partitioning must preserve the independent dense operator and VJP."""
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
import _tide_native as core
from tidegraph.compare import equivalent


@pytest.mark.parametrize("workers,columns", [(1, 11), (3, 11), (8, 3), (3, 0)])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_forward_shared_weight_vjp_and_strided_inputs(dtype, workers, columns, bias):
    torch.manual_seed(7)
    x = torch.randn(5, 14, dtype=dtype, requires_grad=True)
    w = torch.randn(columns, 14, dtype=dtype, requires_grad=True)
    b = torch.randn(columns, dtype=dtype, requires_grad=True) if bias else None
    head = core.DenseLinear(workers)
    leaves = (x, w, b) if bias else (x, w)
    # Two owners share weights; non-contiguous operands and uneven partitions.
    expected = torch.nn.functional.linear(x[:, ::2], w[:, ::2], b)
    expected = expected + torch.nn.functional.linear(2*x[:, ::2], w[:, ::2], b)
    actual = head.run(x[:, ::2], w[:, ::2], b) + head.run(2*x[:, ::2], w[:, ::2], b)
    equivalent(expected, actual)
    cotangent = torch.randn_like(expected)
    a = torch.autograd.grad(actual, leaves, cotangent)
    e = torch.autograd.grad(expected, leaves, cotangent)
    equivalent(e, a)


@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference_mode"])
def test_dense_pool_inherits_each_call_mode_and_serializes_callers(dtype, mode):
    head = core.DenseLinear(3)
    x = torch.ones(3, 4, dtype=dtype, requires_grad=True)
    w = torch.ones(7, 4, dtype=dtype, requires_grad=True)
    def call(_):
        context = torch.enable_grad if mode == "grad" else getattr(torch, mode)
        with context():
            return head.run(x, w)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for value in pool.map(call, range(2)):
            equivalent(value, torch.full((3, 7), 4., dtype=dtype))
            assert value.requires_grad == (mode == "grad")
            assert torch.is_inference(value) == (mode == "inference_mode")
    assert head.run(x, w).requires_grad


@pytest.mark.parametrize("workers", [0, -1, 257])
def test_dense_rejects_worker_budget(workers):
    with pytest.raises(ValueError, match="workers"):
        core.DenseLinear(workers)


@pytest.mark.parametrize("bad", ["rank", "width", "dtype", "bias"])
def test_dense_rejects_incompatible_matrices(dtype, bad):
    x = torch.ones(2, 4, dtype=dtype); w = torch.ones(7, 4, dtype=dtype); b = None
    if bad == "rank": x = x[0]
    if bad == "width": w = w[:, :3]
    if bad == "dtype": w = w.to(torch.int64)
    if bad == "bias": b = torch.ones(6, dtype=dtype)
    with pytest.raises(ValueError, match="dense projection"):
        core.DenseLinear(3).run(x, w, b)
