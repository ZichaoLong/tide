"""Literal one-node KV roots anchor storage precision independently of schedulers."""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.reference import run
from tidegraph.weighted import KVConfig, RegionalConfig, bind
from weighted_precision_reference import reference_model, Store, widen


def fixture(storage_dtype):
    graph = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    b = bind(graph, memories=KVConfig(width=2, summary_width=2, adapter_rank=1,
        ffn_width=2, epsilon=.3, alpha_att=0., alpha_ff=0.),
        regions=RegionalConfig(k=1, gamma_max=1.), dtype=storage_dtype)
    memory = b.model.nodes[0].kernel
    with torch.no_grad():
        memory.base.copy_(torch.tensor([[.2, .3], [-.4, .7], [.1, -.2], [.3, .5]]))
        memory.summary.copy_(torch.tensor([[.6, -.2], [.4, .3]]))
        memory.adapter_u.zero_(); memory.adapter_v.zero_()
    xs = [External(0, 0, 0, 1, torch.tensor([.5, .3, -.7], dtype=storage_dtype))]
    return b, xs


def literal(b, xs, name):
    """No autograd or production norm: hand RMS Jacobian and matrix outer product."""
    memory = b.model.nodes[0].kernel
    u = xs[0].value[1:].float()
    denominator = (u.square().mean()+.3).sqrt(); n = u/denominator
    if name == "weights":
        return torch.tensor(.5), torch.tensor(1.), torch.tensor([1., 0., 0.]), None, None
    matrix = memory.base[:2].detach().float() if name == "keys" else memory.summary.detach().float()
    bar = torch.tensor([.4, -.2]); v = matrix.t()@bar
    du = (v-n*(v*n).mean())/denominator
    gradient = torch.outer(bar, n)
    if name == "keys": gradient = torch.cat((gradient, torch.zeros_like(gradient)))
    return matrix@n, bar, torch.cat((torch.zeros(1), du)), gradient, "base" if name == "keys" else "summary"


@pytest.mark.parametrize("name", ["keys", "summaries", "weights"])
def test_literal_storage_oracle_vjp(name):
    b, xs = fixture(torch.float16)
    model = reference_model(b.model)
    inputs = [replace(xs[0], value=xs[0].value.float().requires_grad_())]
    result = run(b.graph, model, Continuation(b.graph.identity, 1), inputs, 3, sealed_until=3)
    expected, bar, input_bar, matrix_bar, parameter = literal(b, xs, name)
    value = result.continuation.states[0, 0].slots[name][0]
    torch.testing.assert_close(value, expected.half().float(), rtol=0, atol=0)
    memory = model.nodes[0].kernel
    leaves = (inputs[0].value,) if parameter is None else (inputs[0].value, getattr(memory, parameter))
    gradients = torch.autograd.grad(value, leaves, bar)
    torch.testing.assert_close(gradients[0], input_bar, rtol=1e-5, atol=1e-6)
    if parameter is not None: torch.testing.assert_close(gradients[1], matrix_bar, rtol=1e-5, atol=1e-6)
    assert result.continuation.states[0, 0].slots["times"].dtype == torch.int64


def test_store_vjp_is_declared_identity_at_underflow_and_rounding():
    x = torch.tensor([2**-25, .3333], requires_grad=True)
    value = Store.apply(x)
    assert value[0] == 0 and value[1] != x[1]
    bar = torch.tensor([1., .123456])
    torch.testing.assert_close(torch.autograd.grad(value, x, bar)[0], bar, rtol=0, atol=0)
    with pytest.raises(ValueError, match="overflow"): Store.apply(torch.tensor([1e5]))


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
def test_independent_storage_reference_feedback_and_cut(family):
    from test_weighted_execution import setup
    from test_weighted_recurrent_execution import model as recurrent_model
    from tidegraph.compare import equivalent
    b, _, xs, stop = setup(torch.float16, "pdg") if family == "kv" else recurrent_model(torch.float16, family)
    model = reference_model(b.model); xs = widen(xs)
    aliases = lambda m: sorted(sorted(n for n, x in m.named_parameters(remove_duplicate=False) if x is p)
                               for p in m.parameters())
    assert aliases(b.model) == aliases(model)
    q = Continuation(b.graph.identity, 1)
    full = run(b.graph, model, q, xs, stop, sealed_until=stop)
    head = run(b.graph, model, q, [x for x in xs if x.time < 3], 3, sealed_until=3)
    tail = run(b.graph, model, head.continuation, [x for x in xs if x.time >= 3], stop, sealed_until=stop)
    equivalent(full.continuation, tail.continuation, atol=0, rtol=0)
    for state in full.continuation.states.values():
        for x in (state.value, *state.slots.values()):
            if x.is_floating_point(): torch.testing.assert_close(x, x.half().float(), atol=0, rtol=0)
