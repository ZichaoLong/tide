from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.compare import equivalent
from tidegraph.fiber_attention import decode_bias
from tidegraph.ops import Model
from tidegraph.ports import PortLayout
from fiber_cases import PROFILE, native_decode, scalar_model
from read_cases import execute


@pytest.mark.parametrize("implementation", ["reference", "python-frontier", "native-serial", "native-packed", "native-frontier"])
@pytest.mark.parametrize("reverse_slots", [False, True])
def test_same_fiber_all_queries_see_all_current_keys(dtype, implementation, reverse_slots):
    g, m, w = scalar_model(dtype)
    if reverse_slots:
        g = replace(g, layout=PortLayout((), (), (1, 0), (0,)))
    x = torch.tensor([[1.], [3.]], dtype=dtype, requires_grad=True)
    xs = [External(0, p, 0, 2, x[p]) for p in range(2)]
    result = execute(implementation, g, m, Continuation(g.identity, 1), xs, stop=3, mode="hard")
    s2, s6 = x.new_tensor(2).sigmoid(), x.new_tensor(6).sigmoid()
    proposal = result.trace[0]["proposal"]
    equivalent(proposal, (2+2*(s2+s6)).reshape(1))
    # Neither one aggregated token (=4) nor a within-fiber triangular mask.
    assert abs(proposal.item()-4) > 1
    assert abs(proposal.item()-(2+2*s6).item()) > 1
    expected_dx = torch.stack((2-s2-s6+2*(s2*(1-s2)-3*s6*(1-s6)),
                              s2+s6+2*(s2*(1-s2)+5*s6*(1-s6)))).reshape(2, 1)
    grads = torch.autograd.grad(proposal.sum(), (x, w.extra["fiber_out_bias"], w.extra["fiber_decay"], w.decay),
                                allow_unused=True)
    equivalent(grads, (expected_dx, x.new_ones(1), None, None))
    state = result.continuation.states[0, 0]
    equivalent(state.slots["key"].flatten(), x.detach().flatten().flip(0) if reverse_slots else x.detach().flatten())
    assert (state.observations, len(state.slots["key"]), state.last_time) == (1, 2, 2)


@pytest.mark.parametrize("implementation", ["reference", "native-frontier"])
def test_zero_key_is_present_and_tick_zero_decay_has_analytic_vjp(dtype, implementation):
    g, m, w = scalar_model(dtype)
    with torch.no_grad():
        w.extra["fiber_qkv"][:, :2].zero_(); w.extra["fiber_decay"].zero_()
    old = State(torch.zeros(1, dtype=dtype, requires_grad=True), slots={
        "key": torch.zeros(1, 1, 1, dtype=dtype, requires_grad=True),
        "value": torch.full((1, 1, 1), 5., dtype=dtype, requires_grad=True),
        "log_bias": torch.zeros(1, dtype=dtype, requires_grad=True)})
    x = torch.tensor([[1.], [3.]], dtype=dtype, requires_grad=True)
    result = execute(implementation, g, m, Continuation(g.identity, 1, states={(0, 0): old}),
                     [External(0, p, 0, 0, x[p]) for p in range(2)], stop=1)
    y = result.trace[0]["proposal"]; equivalent(y, x.new_tensor([6.]))
    roots = (x, old.slots["key"], old.slots["value"], old.slots["log_bias"], w.extra["fiber_decay"], old.value)
    equivalent(torch.autograd.grad(y.sum(), roots, allow_unused=True),
               (torch.full_like(x, 2/3), torch.zeros_like(roots[1]), torch.full_like(roots[2], 2/3),
                x.new_tensor([4/3]), x.new_tensor(-4/3), None))


@pytest.mark.parametrize("native", [False, True])
def test_fiber_physical_bias_decoder_preserves_repeat_and_structural_absence(dtype, native):
    _, _, w = scalar_model(dtype); fn = native_decode if native else decode_bias
    bias = torch.tensor([0., .3], dtype=dtype, requires_grad=True)
    state = State(torch.zeros(1, dtype=dtype), slots={"key": bias.new_zeros((2, 1, 1)),
                  "value": bias.new_zeros((2, 1, 1)), "log_bias": bias})
    eager = bias
    for cut in range(18):
        assert torch.equal(fn(w, state, cut), eager)
        eager = eager-w.extra["fiber_decay"]
    grad = torch.autograd.grad(fn(w, state, 7).sum(), w.extra["fiber_decay"])[0]
    equivalent(grad, bias.new_tensor(-14))
    empty = w.initial(); empty.slots["log_bias"].requires_grad_()
    assert torch.autograd.grad(fn(w, empty, 17).sum(), w.extra["fiber_decay"], allow_unused=True)[0] is None
    assert state.last_time == -1 and state.observations == 0


@pytest.mark.parametrize("native", [False, True])
def test_fiber_decoder_rejects_bad_state_and_weights(dtype, native):
    _, _, w = scalar_model(dtype); fn = native_decode if native else decode_bias
    initial = w.initial()
    bad_states = [replace(initial, value=torch.zeros(2, dtype=dtype)), replace(initial, last_time=2),
                  replace(initial, last_time=-2), replace(initial, observations=-1),
                  replace(initial, slots={}), replace(initial, slots=initial.slots | {"log_bias": torch.zeros(1, dtype=dtype)}),
                  replace(initial, slots=initial.slots | {"value": torch.zeros(1, 1, 1, dtype=dtype)})]
    for state in bad_states:
        with pytest.raises((ValueError, RuntimeError)):
            fn(w, state, 2)
    for value in (torch.zeros(1, dtype=dtype), torch.tensor(float("nan"), dtype=dtype),
                  torch.zeros((), dtype=torch.float32 if dtype == torch.float64 else torch.float64)):
        w.extra["fiber_decay"] = torch.nn.Parameter(value)
        with pytest.raises((ValueError, RuntimeError), match="parameter"):
            fn(w, initial, 2)


@pytest.mark.parametrize("policy", [dict(query_heads=2, kv_heads=1), dict(window=3), dict(aggregation="mean")])
def test_fiber_profile_rejects_different_semantics(dtype, policy):
    g = Graph((Node(0, memory=PROFILE, **policy),), (), (Region(1),), (0,), ())
    with pytest.raises(ValueError, match="equal heads, no eviction and sum"):
        Model(g, width=4, dtype=dtype)
