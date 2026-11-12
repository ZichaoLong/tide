"""KV §4 equations (7)–(8): scalar-query formula versus the B×L Full batch.

The oracle uses a weighted exponential quotient per query/head, independently
of the candidate's log-weight softmax, packing, and Attention implementation.
"""
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fc_tensor_baseline import full


def material(dtype, device):
    generator = torch.Generator().manual_seed(731)
    def value(shape, positive=False):
        x = torch.rand(shape, generator=generator, dtype=dtype)
        return ((x*.4+.2) if positive else (x-.5)*.4).to(device).requires_grad_()
    roots = dict(u=value((3, 5, 4)), keys=value((3, 7, 4)), values=value((3, 7, 4)),
                 weights=value((3, 7), True), query=value((4, 4)), out=value((4, 4)),
                 gate=value((8, 4)), up=value((8, 4)), down=value((4, 8)))
    return roots


def node(roots):
    return SimpleNamespace(kernel=SimpleNamespace(config=SimpleNamespace(
        epsilon=1e-5, heads=2, alpha_att=.7, alpha_ff=.6)),
        full_program=SimpleNamespace(content=SimpleNamespace(
            **{name: roots[name] for name in ("query", "out", "gate", "up", "down")})))


def literal(roots, controls, visible):
    def normalized(x):
        return x/torch.sqrt(torch.mean(x*x)+1e-5)
    rows = []
    for b in range(3):
        samples = []
        for t in range(5):
            if not bool(controls[b, t] > 0):
                samples.append(torch.zeros_like(roots["u"][b, t]))
                continue
            u = roots["u"][b, t]
            q = roots["query"]@normalized(u)
            weight = roots["weights"][b]*visible[t]
            heads = []
            for h in range(2):
                k = roots["keys"][b, :, h*2:(h+1)*2]
                v = roots["values"][b, :, h*2:(h+1)*2]
                score = (k@q[h*2:(h+1)*2])/math.sqrt(2)
                numerator = (score-score.max()).exp()*weight
                heads.append(((numerator/numerator.sum())@v)*weight.sum().clamp_max(1))
            x = u+.7*(roots["out"]@torch.cat(heads))
            n = normalized(x)
            gate = roots["gate"]@n
            samples.append(x+.6*(roots["down"]@((gate*gate.sigmoid())*(roots["up"]@n))))
        rows.append(torch.stack(samples))
    return torch.stack(rows)


@pytest.mark.parametrize("pattern", ["uneven", "one", "all", "empty"])
@pytest.mark.parametrize("attention", ["manual", "sdpa4d"])
def test_selected_batch_values_and_all_input_vjps(dtype, pattern, attention):
    from tidegraph.runtime import resolve_device
    from layer_tensor_baseline import numerics
    device, _ = resolve_device(os.environ.get("TIDE_FC_DEVICE", "cpu"))
    numerics(device)
    roots = material(dtype, device)
    oracle = {name: x.detach().clone().requires_grad_() for name, x in roots.items()}
    controls = torch.tensor([[1, 0, .3, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, .2]],
                            dtype=dtype, device=device)
    if pattern == "one":
        controls.zero_(); controls[2, 4] = .2
    elif pattern == "all":
        controls.fill_(.3)
    elif pattern == "empty":
        controls.zero_()
    controls.requires_grad_()
    visible = torch.arange(7, device=device)[None] <= torch.arange(5, device=device)[:, None]+2
    prepared = roots["u"], {name: roots[name] for name in ("keys", "values", "weights")}, visible
    actual, active = full(node(roots), prepared, controls, attention=attention)
    expected = literal(oracle, controls, visible)
    assert torch.equal(active, controls.detach() > 0)
    atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (1e-6, 1e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if pattern == "empty":
        assert not actual.requires_grad and not expected.requires_grad
        return
    coefficient = torch.linspace(-.7, 1.1, actual.numel(), dtype=dtype, device=device).reshape_as(actual)
    a = torch.autograd.grad((actual*coefficient).sum(), [*roots.values(), controls], allow_unused=True)
    e = torch.autograd.grad((expected*coefficient).sum(), [*oracle.values(), controls], allow_unused=True)
    for x, y in zip(a, e):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
