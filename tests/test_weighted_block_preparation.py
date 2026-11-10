"""KV §1.2: B×sequence projections, literal dense formulas and independent roots."""
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.weighted import KVConfig, RegionalConfig, bind_layered, fully_connected
from tidegraph.weighted.kv_block_preparation import prepare


def fixture(dtype, rows=6):
    _, binding = bind_layered(fully_connected(2, 2), memory=KVConfig(width=3, summary_width=2),
                              regions=RegionalConfig(k=2), dtype=dtype)
    memories = [binding.model.nodes[i].kernel for i in (0, 1)]
    inputs = [torch.tensor([1., .2+i*.01, -.4, .7], dtype=dtype, requires_grad=True) for i in range(rows)]
    kernels = [m for x in inputs for m in memories]
    contents = [x for x in inputs for m in memories]
    return memories, inputs, kernels, contents


def literal(memory, content):
    x = content[1:]/torch.sqrt(memory.config.epsilon+content[1:].square().mean())
    return dict(values=(memory.base+memory.adapter_u@memory.adapter_v)@x, summaries=memory.summary@x)


@pytest.mark.parametrize("field", ["values", "summaries"])
def test_joint_preparation_literal_formula_vjp_and_absent_rows(dtype, field):
    memories, inputs, kernels, contents = fixture(dtype)
    stats = {}; actual = prepare(kernels, contents, stats)
    expected = [literal(m, x) for m, x in zip(kernels, contents)]
    equivalent(expected, actual)
    roots = list(dict.fromkeys([*inputs, *[p for m in memories for p in m.parameters()]]))
    for zero in (False, True):
        a, b = actual[3][field], expected[3][field]
        bar = torch.zeros_like(a) if zero else torch.arange(1, a.numel()+1, dtype=dtype)
        ga = torch.autograd.grad(a, roots, bar, allow_unused=True, retain_graph=True)
        gb = torch.autograd.grad(b, roots, bar, allow_unused=True, retain_graph=True)
        equivalent(ga, gb)
        assert all(ga[i] is None for i in (0, 2, 3, 4, 5))
    assert stats["state_block_base_projection_calls"] == 1
    assert stats["state_private_projection_calls"] == 6
    assert stats["max_state_private_projection_batch"] == 6


def test_joint_preparation_zero_write_and_equal_distinct_inputs(dtype):
    memories, inputs, kernels, contents = fixture(dtype, 3)
    inputs[0].data[0] = 0
    contents[3] = inputs[1].detach().clone().requires_grad_()
    stats = {}; actual = prepare(kernels, contents, stats)
    assert actual[0] == actual[1] == {}
    gradients = torch.autograd.grad(actual[2]["values"].sum(), [*inputs, contents[3]], allow_unused=True)
    assert gradients[0] is None and gradients[2] is None and gradients[3] is None
    assert gradients[1] is not None
    assert stats["state_preparation_reused_events"] == 1


def test_joint_preparation_shared_private_parameters_accumulate(dtype):
    memories, inputs, kernels, contents = fixture(dtype, 3)
    for name in ("adapter_v", "adapter_u", "summary"):
        setattr(memories[1], name, getattr(memories[0], name))
    actual = prepare(kernels, contents, {})
    expected = [literal(m, x) for m, x in zip(kernels, contents)]
    roots = [*inputs, *memories[0].parameters()]
    def loss(rows):
        return sum((i+1)*(r["values"].sum()+r["summaries"].sum()) for i, r in enumerate(rows))
    equivalent(torch.autograd.grad(loss(actual), roots, allow_unused=True),
               torch.autograd.grad(loss(expected), roots, allow_unused=True))


@pytest.mark.parametrize("mutate", [False, True])
def test_joint_preparation_versions_and_first_order(mutate):
    memories, inputs, kernels, contents = fixture(torch.float64, 2)
    output = prepare(kernels, contents, {})[0]["values"].sum()
    if mutate:
        with torch.no_grad(): memories[0].base.add_(1)
    with pytest.raises((ValueError, RuntimeError), match="inplace|first-order"):
        torch.autograd.grad(output, inputs[0], create_graph=not mutate)
