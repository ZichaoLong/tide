"""Original logical-time half arithmetic: int64 cuts, zero rates, long KV phases."""
import math
import pytest
import torch
from tidegraph import Graph, Node, Region, External, Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.qualification_checks import finite, vjp
from tidegraph.reference import run as reference
from tidegraph.placement import ExecutionPlacement, place_model
from tidegraph.weighted import KVConfig, RecurrentConfig, RegionalConfig, bind
from tidegraph.weighted.numeric_time import passive_decay, time_values
from tidegraph.weighted.tensors import rotate_relative
from test_weighted_precision import run, half_close


def test_half_passive_decay_zero_rate_after_large_int64_subtraction():
    old = torch.tensor(2**54+7, dtype=torch.int64)
    age = old.new_tensor(2**54+100007)-old
    like = torch.zeros(2, dtype=torch.float16)
    result = passive_decay(age, (0., 1e-6), like)
    assert age.item() == 100000 and time_values(age, like).dtype == torch.float32
    torch.testing.assert_close(result, like.new_tensor([1., math.exp(-.1)]), atol=0, rtol=0)


def test_half_rotation_uses_large_logical_displacement():
    keys = torch.tensor([[.25, -.5]], dtype=torch.float16, requires_grad=True)
    delta = time_values(torch.tensor([-100000], dtype=torch.int64), keys)/3
    actual = rotate_relative(keys, delta, (.25,))
    phase = float(delta[0]*.25)
    expected = keys.new_tensor([[.25*math.cos(phase)+.5*math.sin(phase),
                                 .25*math.sin(phase)-.5*math.cos(phase)]])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient, = torch.autograd.grad(actual.sum(), keys)
    torch.testing.assert_close(gradient, keys.new_tensor([[math.cos(phase)+math.sin(phase),
                                                          math.cos(phase)-math.sin(phase)]]), atol=1e-3, rtol=2e-3)


@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("native", [False, True])
def test_half_graph_preserves_sparse_large_timestamps_and_full_vjp(family, native):
    graph = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    if family == "kv":
        config = KVConfig(width=2, summary_width=2, ffn_width=3, epsilon=.3, memory_read="qry",
                          beta_short=1e-6, read_frequencies=(.25,), full_frequencies=((.25,),),
                          distance_slopes=(.3,), position_scale=3.)
    else:
        extra = {} if family in {"affine", "delta"} else dict(conv_size=3, conv_stride=3)
        config = RecurrentConfig(memory_family=family, width=2, summary_width=2, ffn_width=3,
                                 epsilon=.3, lambda_short=1e-6, memory_read="state", **extra)
    a = bind(graph, memories=config, regions=RegionalConfig(k=1), dtype=torch.float16)
    b = bind(graph, memories=config, regions=RegionalConfig(k=1), dtype=torch.float16)
    start = 2**54+7
    def inputs():
        return [External(0, 0, i, start+t, torch.tensor([.5, .3, -.4], dtype=torch.float16, requires_grad=True))
                for i, t in enumerate((0, 100000))]
    xa, xb = inputs(), inputs()
    q = Continuation(a.graph.identity, 1, cut=start)
    view = place_model(a.graph, a.model, ExecutionPlacement(scoring_dtype="float32"))
    expected = reference(a.graph, view, q, xa, start+100001, sealed_until=start+100001)
    actual = run(b, native)(q, xb, start+100001, sealed_until=start+100001)
    finite(expected); finite(actual); half_close(expected, actual)
    state = actual.continuation.states[0, 0]
    clock = state.slots["times"] if family == "kv" else state.slots["time"]
    assert clock.dtype == torch.int64 and int(clock.flatten()[-1]) == start+100000
    roots = lambda owner, xs: {n: p for n, p in owner.model.named_parameters() if p.requires_grad} | {
        f"input{i}": x.value for i, x in enumerate(xs)}
    half_close(vjp(objective(expected), roots(a, xa)), vjp(objective(actual), roots(b, xb)))
