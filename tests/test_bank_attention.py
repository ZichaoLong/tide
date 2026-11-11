"""Independent literal weighted attention; stored sharing != adjoint sharing."""
import os
import pytest
import torch
from tidegraph.weighted.bank_attention import full
from tidegraph.runtime import resolve_device
from tidegraph.compare import equivalent


def test_matrix_bank_full_values_independent_snapshot_gradients_and_versions(dtype):
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    k = torch.linspace(-.4, .5, 48, dtype=dtype, device=device).reshape(12, 4)
    v = torch.cat((torch.zeros_like(k[:1]), torch.cos(k)))
    ranges = ((0, 3), (0, 4), (1, 5), (8, 11), (8, 12))
    keys = [k[a:b].detach().requires_grad_() for a, b in ranges]
    values = [v[a+1:b+1].detach().requires_grad_() for a, b in ranges]
    # Two different banks with nonzero offsets, ragged queries and sliding cache.
    other = k[8:].clone()
    keys[3:] = [other[:n].detach().requires_grad_() for n in (3, 4)]
    q = torch.linspace(-.5, .6, 20, dtype=dtype, device=device).reshape(5, 2, 2).requires_grad_()
    w = torch.tensor([[.2, 0, .4, 0], [.3, .8, 0, .7], [0, .2, .3, .5], [.1, .2, .5, 0], [.4, .3, .8, .9]],
                     dtype=dtype, device=device, requires_grad=True)
    actual = full(q, keys, values, w)
    assert actual is not None
    expected = []
    for i, (key, value) in enumerate(zip(keys, values)):
        positive = w[i, :len(key)] > 0
        logits = (q[i, :, None]*key[positive].reshape(-1, 2, 2).transpose(0, 1)).sum(-1)/2**.5
        logits = logits+w[i, :len(key)][positive].log()
        # Independent ordinary softmax, avoiding the candidate's custom VJP.
        expected.append((logits.softmax(-1)[..., None]*value[positive].reshape(-1, 2, 2).transpose(0, 1)).sum(-2))
    expected = torch.stack(expected)
    equivalent(actual, expected)
    roots = [q, w, *keys, *values]
    for objective in (lambda x: x.square().sum(), lambda x: x[1].sum(), lambda x: 0*x[0].sum()):
        a = torch.autograd.grad(objective(actual), roots, retain_graph=True, allow_unused=True)
        e = torch.autograd.grad(objective(expected), roots, retain_graph=True, allow_unused=True)
        equivalent(a, e)
    with torch.no_grad():
        keys[0].add_(1)
    with pytest.raises(RuntimeError, match="modified|version|inplace"):
        actual.sum().backward()


def test_matrix_bank_ignored_huge_values_and_unshared_fallback(dtype):
    base = torch.ones(4, 4, dtype=dtype)
    base[0] = torch.finfo(dtype).max
    rows = [base[:3].detach().requires_grad_(), base[:4].detach().requires_grad_()]
    q = torch.ones(2, 2, 2, dtype=dtype, requires_grad=True)
    weights = torch.tensor([[0., .5, .4, 0.], [0., .1, .2, .3]], dtype=dtype, requires_grad=True)
    actual = full(q, rows, rows, weights)
    assert torch.isfinite(actual).all()
    for g in torch.autograd.grad(actual.sum(), [q, weights, *rows]):
        assert torch.isfinite(g).all()
    assert full(q, [t.clone() for t in rows], rows, weights) is None
    assert full(q, rows, rows, torch.zeros_like(weights)) is None


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_default_entry_uses_actual_block_bank_contractions(dtype, native):
    from tidegraph import Continuation, ExecutionPlacement
    from tidegraph.weighted.execution import eager_runner
    from tidegraph.weighted.packing_observation import observe
    from test_layer_tensor_baseline import fixture, external, run, numerics
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    numerics(device)
    binding, spec = fixture("kv", width=4, depth=2, heads=2, dtype=dtype, device=device)
    execute = eager_runner(binding, "native" if native else "python",
        ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c",
                           scoring_dtype=str(dtype).split('.')[-1]),
        native_library=os.environ.get("TIDE_BUILD_DIR"), trace=False)
    x = torch.linspace(-.2, .4, 60, dtype=dtype, device=device).reshape(3, 5, 4)
    if native:
        import _tide_native as core
        core.reset_work(True)
    try:
        with observe() as observation, torch.no_grad():
            result = execute(Continuation(binding.graph.identity, 3), external(spec, x),
                             5*spec.stride, sealed_until=5*spec.stride)
        metrics = core.work_metrics() if native else observation.metrics()
        assert metrics.get("packing/projection/kv_block_full/rows", 0) > 0
    finally:
        if native:
            core.reset_work(False)
    expected, _ = run(binding, spec, x)
    actual = torch.stack([p[1:] for *_, p in result.outputs]).reshape(5, 3, 4).transpose(0, 1)
    equivalent(actual, expected)
