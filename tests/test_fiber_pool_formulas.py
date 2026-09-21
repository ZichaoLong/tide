from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.compare import equivalent
from tidegraph.native import Native
from tidegraph.reference import run
from fiber_pool_cases import POOLS, analytic
from read_cases import execute


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-frontier"])
def test_post_attention_pooling_analytic_values_source_and_coefficient_vjps(dtype, kind, reverse, implementation):
    g, m, w = analytic(dtype, kind, reverse)
    x = torch.tensor([[1.], [3.]], dtype=dtype, requires_grad=True)
    xs = [External(0, p, 0, 2, x[i]) for i, p in enumerate((0, 2))]
    result = execute(implementation, g, m, Continuation(g.identity, 1), xs, stop=3, mode="hard")
    slots = (2, 1) if reverse else (0, 2); missing = ({0, 1, 2}-set(slots)).pop()
    masses = x.new_tensor([2., 3., 5.]); selected = masses[list(slots)]
    coe = x.new_full((2,), .5) if kind == "mean" else selected if kind == "linear" else (
        selected/selected.sum() if kind == "active-softmax" else selected/10)
    s1, s3 = x.new_tensor(2).sigmoid(), x.new_tensor(6).sigmoid()
    rows = torch.stack((1+2*s1, 1+2*s3)); pooled = (coe*rows).sum()
    y = result.trace[0]["proposal"]; equivalent(y, (pooled+.7).reshape(1))
    roots = (x, w.extra["fiber_out_bias"], w.extra["fiber_decay"])
    if kind != "mean":
        roots += (w.extra["fiber_pool"],)
    grads = torch.autograd.grad(y.sum(), roots, allow_unused=True, retain_graph=True)
    da = coe[0]*(1-s1+2*s1*(1-s1))+coe[1]*(1-s3-6*s3*(1-s3))
    db = coe[0]*(s1+2*s1*(1-s1))+coe[1]*(s3+10*s3*(1-s3))
    equivalent(grads[:3], (torch.stack((da, db)).reshape(2, 1), x.new_ones(1), None))
    if kind != "mean":
        expected = x.new_zeros(3)
        if kind == "all-softmax":
            expected = -masses/10*pooled
        for i, slot in enumerate(slots):
            expected[slot] = rows[i] if kind == "linear" else coe[i]*(rows[i]-pooled) if kind == "active-softmax" else (
                expected[slot]+coe[i]*rows[i])
        equivalent(grads[3], expected)
        assert grads[3][missing] < 0 if kind == "all-softmax" else grads[3][missing] == 0
        cache = result.continuation.states[0, 0].slots
        assert torch.autograd.grad(cache["key"].sum()+cache["value"].sum(), roots[-1], allow_unused=True)[0] is None
    # Pool coefficients did not rescale Q/K/V inputs; local slots determine row order.
    equivalent(result.continuation.states[0, 0].slots["key"].flatten(), x.flatten().flip(0) if reverse else x.flatten())


@pytest.mark.parametrize("implementation", ["reference", "native-frontier"])
def test_zero_output_pool_weight_does_not_remove_its_key_or_query(dtype, implementation):
    g, m, w = analytic(dtype, "linear")
    with torch.no_grad():
        w.extra["fiber_pool"].copy_(w.bias.new_tensor([1., 0., 0.])); w.extra["fiber_out_bias"].zero_()
    x = torch.tensor([[1.], [3.]], dtype=dtype, requires_grad=True)
    result = execute(implementation, g, m, Continuation(g.identity, 1),
                     [External(0, p, 0, 0, x[i]) for i, p in enumerate((0, 2))], stop=1, mode="hard")
    y = result.trace[0]["proposal"]; equivalent(y, (1+2*x.new_tensor(2).sigmoid()).reshape(1))
    gx, gw = torch.autograd.grad(y.sum(), (x, w.extra["fiber_pool"]))
    assert gx[1] > 1 and gw[2] > 2.9
    assert len(result.continuation.states[0, 0].slots["key"]) == 2


@pytest.mark.parametrize("kind", POOLS)
@pytest.mark.parametrize("implementation", ["reference", "native-frontier"])
def test_present_zero_source_is_not_absent_pooling(dtype, kind, implementation):
    g, m, _ = analytic(dtype, kind); q = Continuation(g.identity, 1)
    xs = [External(0, p, 0, 0, torch.tensor([value], dtype=dtype)) for p, value in ((0, 1.), (2, 3.))]
    absent = execute(implementation, g, m, q, xs, stop=1)
    present = execute(implementation, g, m, q, xs+[External(0, 1, 0, 0, torch.zeros(1, dtype=dtype))], stop=1)
    assert len(present.continuation.states[0, 0].slots["key"]) == 3
    assert not torch.allclose(absent.trace[0]["proposal"], present.trace[0]["proposal"])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite", "domain", "profile"])
def test_pooling_parameter_and_shared_program_domain_validation(dtype, native, bad):
    g, m, w = analytic(dtype, "all-softmax")
    if bad == "shape":
        w.extra["fiber_pool"] = torch.nn.Parameter(torch.zeros(1, 3, dtype=dtype))
    elif bad == "dtype":
        w.extra["fiber_pool"] = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32 if dtype == torch.float64 else torch.float64))
    elif bad == "nonfinite":
        with torch.no_grad(): w.extra["fiber_pool"][1] = float("nan")
    elif bad == "domain":
        g = replace(g, inputs=(0, 0))
    else:
        g = replace(g, nodes=(replace(g.nodes[0], memory="lh-fiber-attention-linear-repeat-v1"),))
    with pytest.raises(ValueError, match="fiber"):
        q = Continuation(g.identity, 1)
        if native: Native(g, m).run(q, [], 0, sealed_until=0)
        else: run(g, m, q, [], 0, sealed_until=0)
