"""KV §1.2: independent literal maps/Jacobians, bounded packs and private roots."""
from types import SimpleNamespace
import math
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted.kv_independent_batch import prepare


def memory(dtype, epsilon=1.):
    def leaf(value):
        return torch.tensor(value, dtype=dtype, requires_grad=True)
    return SimpleNamespace(config=SimpleNamespace(epsilon=epsilon),
        base=leaf([[2.], [3.]]), adapter_u=leaf([[.4], [-.2]]),
        adapter_v=leaf([[.5]]), summary=leaf([[.7]]))


def test_distinct_inputs_bases_literal_jacobian_and_none(dtype):
    ms = [memory(dtype), memory(dtype, 2.), memory(dtype)]
    xs = [torch.tensor([.4, u], dtype=dtype, requires_grad=True) for u in (2., -3., 4.)]
    xs[-1] = torch.tensor([0., 4.], dtype=dtype, requires_grad=True)
    stats = {};rows = prepare(ms, xs, stats)
    for i, (u, eps) in enumerate(((2., 1.), (-3., 2.))):
        n = u/math.sqrt(eps+u*u)
        equivalent(rows[i]["values"], xs[i].new_tensor([2.2*n, 2.9*n]))
        equivalent(rows[i]["summaries"], xs[i].new_tensor([.7*n]))
    assert rows[-1] == {}
    roots = (xs[0], ms[0].base, ms[0].adapter_u, ms[0].adapter_v, ms[0].summary,
             xs[1], ms[1].base, xs[2])
    loss = rows[0]["values"].sum()+2*rows[0]["summaries"].sum()
    grads = torch.autograd.grad(loss, roots, allow_unused=True, retain_graph=True)
    n = 2/math.sqrt(5)
    for actual, expected in zip(grads[:5], ([0., 6.5/(5**1.5)], [[n], [n]],
                                         [[.5*n], [.5*n]], [[.2*n]], [[2*n]])):
        equivalent(actual, xs[0].new_tensor(expected))
    assert grads[5:] == (None, None, None)
    zero = torch.autograd.grad(loss*0, roots, allow_unused=True)
    assert zero[5:] == (None, None, None)
    assert all(torch.equal(z, torch.zeros_like(z)) for z in zero[:5])
    assert stats == dict(state_independent_projection_calls=4, state_independent_projection_rows=8,
                         max_state_independent_projection_batch=2)


def test_distinct_input_shared_base_all_contributions(dtype):
    ms = [memory(dtype), memory(dtype)]
    ms[1].base = ms[0].base
    xs = [torch.tensor([.4, u], dtype=dtype, requires_grad=True) for u in (2., -3.)]
    rows = prepare(ms, xs, {})
    grad, = torch.autograd.grad(rows[0]["values"].sum()+3*rows[1]["values"].sum(), (ms[0].base,))
    equivalent(grad, torch.full_like(grad, 2/math.sqrt(5)-9/math.sqrt(10)))


def test_bounded_packing_singleton_tail_and_parameter_versions(dtype):
    ms = [memory(dtype) for _ in range(5)]
    xs = [torch.tensor([.4, 2.], dtype=dtype, requires_grad=True) for _ in ms]
    row_bytes = 4*sum(p.numel()*p.element_size() for p in
                      (ms[0].base, ms[0].adapter_v, ms[0].adapter_u, ms[0].summary))
    stats = {};rows = prepare(ms, xs, stats, budget=2*row_bytes)
    assert all(rows[i] for i in range(4)) and rows[4] == {}
    assert stats["max_state_independent_projection_batch"] == 2
    assert prepare(ms, xs, {}, budget=row_bytes) == [{}]*5
    with torch.no_grad():
        ms[0].base.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        rows[0]["values"].sum().backward()


def test_heterogeneous_shapes_preserve_order(dtype):
    ms = [memory(dtype) for _ in range(3)]
    ms[1].adapter_u = torch.ones((2, 2), dtype=dtype, requires_grad=True)
    ms[1].adapter_v = torch.ones((2, 1), dtype=dtype, requires_grad=True)
    xs = [torch.tensor([.3, u], dtype=dtype, requires_grad=True) for u in (2., 1., -3.)]
    rows = prepare(ms, xs, {})
    assert rows[1] == {} and rows[0] and rows[2]
    assert rows[0]["values"][0] > 0 and rows[2]["values"][0] < 0
