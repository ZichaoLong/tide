from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent
from tidegraph.full import FullProgram, FullResult, ProjectionEmit
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.reference import run


class ClockFull(FullProgram):
    profile = "clock-full-v1"

    def step(self, weights, request, slots, mode, zeta):
        assert request.comparison.last_time == request.time and slots == 1
        return FullResult(None, {0: request.comparison.value + request.time + request.comparison.observations})


def test_python_custom_full_receives_complete_preclear_state_and_falls_back(dtype):
    g = Graph((Node(0, clear=True, emission="clock-full-v1"),), (), (Region(1),), (0,), (0,))
    m = Model(g, width=2, dtype=dtype, full_programs={0: ClockFull()})
    with torch.no_grad():
        m.input_scale[0].fill_(1); m.output_scale[0].fill_(1)
    x = torch.ones(2, dtype=dtype, requires_grad=True)
    xs = [External(0, 0, p, time, x) for p, time in enumerate((0, 2))]
    q = Continuation(g.identity, 1)
    expected = run(g, m, q, xs, 3, sealed_until=3)
    actual = frontier(g, m, q, xs, 3, sealed_until=3)
    equivalent(expected, actual)
    equivalent([v for _, _, _, v in actual.outputs], [x+1, x+4])
    assert all(e["full"] is None for e in actual.trace)
    assert actual.stats["full_scalar_fallback_steps"] == 2
    for result in (expected, actual):
        gradient, = torch.autograd.grad(sum(v.sum() for _, _, _, v in result.outputs), x)
        equivalent(gradient, torch.full_like(x, 2))
    with pytest.raises(ValueError, match="no native implementation"):
        Native(g, m)


@pytest.mark.parametrize("fault", ["presence", "slot", "shape", "count"])
def test_python_invalid_full_batch_contract_is_rejected(dtype, fault):
    class Broken(ClockFull):
        def batch(self, weights, requests, slots, mode, zeta):
            result = super().batch(weights, requests, slots, mode, zeta)
            if fault == "presence":
                result[0].emitted = {}
            elif fault == "slot":
                result[0].emitted = {1: requests[0].content.value}
            elif fault == "shape":
                result[0].emitted = {0: requests[0].content.value[:1]}
            else:
                result.pop()
            return result
    g = Graph((Node(0, emission="clock-full-v1"),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype, full_programs={0: Broken()})
    with pytest.raises(ValueError, match="Full"):
        frontier(g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))], 1, sealed_until=1)


def test_emission_policy_checkpoint_and_shared_program_guards(dtype, tmp_path):
    g = Graph((Node(0, emission="slot_affine", emit_period=2, emit_phases=(0,)),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); path = tmp_path / "emission.pt"
    save(path, g, m, Continuation(g.identity, 1))
    changed = replace(g, nodes=(replace(g.nodes[0], emit_phases=(1,)),))
    restored = Model(changed, dtype=dtype, seed=29)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, restored)
    equivalent(before, restored.state_dict())
    with pytest.raises(ValueError, match="policy"):
        Native(changed, m)
    with pytest.raises(ValueError, match="policy"):
        run(changed, m, Continuation(changed.identity, 1), [], 0, sealed_until=0)
    changed_backbone = replace(g, nodes=(replace(g.nodes[0], full="swiglu"),))
    with pytest.raises(ValueError, match="backbone"):
        Native(changed_backbone, m)


@pytest.mark.parametrize("policy", [{"emit_period": 0}, {"emit_phases": (-3,)}, {"emit_phases": (1,)},
                                    {"emit_phases": (0, 0)}, {"emit_phases": (True,)}])
def test_invalid_emission_policy_fails_at_graph_construction(policy):
    with pytest.raises(ValueError, match="phase"):
        Graph((Node(0, **policy),), (), (Region(1),), (0,), (0,))


def test_python_custom_full_parameters_are_registered_and_trainable(dtype):
    class Learned(ClockFull):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.3, dtype=dtype))

        def step(self, weights, request, slots, mode, zeta):
            result = super().step(weights, request, slots, mode, zeta)
            result.emitted[0] = result.emitted[0] * self.gain
            return result
    program = Learned()
    g = Graph((Node(0, emission="clock-full-v1"),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype, full_programs={0: program})
    assert dict(m.named_parameters())["nodes.0.full_program.gain"] is program.gain
    result = frontier(g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))], 1, sealed_until=1)
    grad, = torch.autograd.grad(result.outputs[0][-1].sum(), program.gain)
    assert grad.abs().sum() > 0


def test_python_projection_override_is_not_silently_replaced_by_native(dtype):
    class Altered(ProjectionEmit):
        def step(self, weights, request, slots, mode, zeta):
            result = super().step(weights, request, slots, mode, zeta)
            result.emitted = {slot: value * 2 for slot, value in result.emitted.items()}
            return result
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype, full_programs={0: Altered(g.nodes[0])})
    base = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    xs = [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))]
    actual = run(g, m, q, xs, 1, sealed_until=1)
    expected = run(g, base, q, xs, 1, sealed_until=1)
    equivalent(actual.outputs[0][-1], expected.outputs[0][-1] * 2)
    with pytest.raises(ValueError, match="no native implementation"):
        Native(g, m)
