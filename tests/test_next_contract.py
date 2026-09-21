from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.next import NextProgram
from tidegraph.ops import Model
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from isolated_cases import vjp


class ClockNext(NextProgram):
    profile = "clock-next-v1"

    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2., dtype=dtype))

    def step(self, w, r):
        value = r.old.value + self.gain * (r.comparison.value + r.time*r.content.value +
                                         (r.control if r.active else -r.control))
        source = sum((s.slot+1)*(s.atom.position+1)*s.atom.value*s.scale for s in r.content.sources)
        return State(value, r.time, r.old.observations+1, {"memory": r.old.slots["memory"]+self.gain*source})


def custom_fixture(dtype, clear=False):
    g = Graph((Node(0, clear=clear, memory="ssm", next_state=ClockNext.profile),
               Node(0, memory="ssm", next_state=ClockNext.profile)), (),
              (Region(1, count_priority=False, read_mode="content"),), (0, 1), (0, 1))
    transition = ClockNext(dtype)
    m = Model(g, width=1, dtype=dtype, next_programs={0: transition, 1: transition})
    with torch.no_grad():
        for w in m.nodes:
            w.read.zero_()
            for p in w.extra.values():
                p.zero_()
        for p in [*m.input_scale, *m.output_scale]:
            p.fill_(1)
    x = torch.tensor([[1., 3.], [2., 0.]], dtype=dtype, requires_grad=True)
    other = torch.ones_like(x, requires_grad=True)
    xs = [External(b, v, i, time, (x if b == 0 else other)[i, v:v+1]) for b in range(2) for v in range(2)
          for i, time in enumerate((1, 4))]
    return g, m, Continuation(g.identity, 2), xs, dict(m.named_parameters()) | {"x": x, "other": other}


@pytest.mark.parametrize("clear", [False, True])
def test_custom_next_uses_full_request_and_survives_optimizer_checkpoint(dtype, clear, tmp_path):
    g, m, q, xs, leaves = custom_fixture(dtype, clear)
    expected = run(g, m, q, xs, 5, sealed_until=5, mode="hst")
    actual = frontier(g, m, q, xs, 5, sealed_until=5, mode="hst")
    equivalent(expected, actual); equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    assert actual.stats.get("state_blocks", 0) == 0 and actual.stats["full_blocks"] > 0
    if not clear:
        loss = sum(actual.continuation.states[0, v].value.sum() + actual.continuation.states[0, v].slots["memory"].sum()
                   for v in range(2))
        equivalent(loss, loss.new_tensor(40.))
        gain = m.nodes[0].next_program.gain
        dx, dg, unused = torch.autograd.grad(loss, [leaves["x"], gain, leaves["other"]], allow_unused=True, retain_graph=True)
        equivalent(dx, torch.tensor([[4., 4.], [12., 12.]], dtype=dtype))
        equivalent(dg, gain.new_tensor(20.)); assert unused is None
    with torch.no_grad():
        equivalent(actual, frontier(g, m, q, xs, 5, sealed_until=5, mode="hst"))
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    objective(actual).backward(); optimizer.step()
    gain = m.nodes[0].next_program.gain; assert gain in optimizer.state
    path = tmp_path / "custom-next.pt"; save(path, g, m, actual.continuation, optimizer)
    _, restored, _, _, _ = custom_fixture(dtype, clear); ro = torch.optim.AdamW(restored.parameters(), lr=.01)
    equivalent(actual.continuation.detach(), load(path, g, restored, ro))
    equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())
    with pytest.raises(ValueError, match="custom Next.*no native"):
        Native(g, m)


def test_custom_source_aware_next_embeds_into_settle_graph(dtype):
    g, m, q, _, _ = custom_fixture(dtype)
    spec = SettleGraph(g, (1,)); x = torch.ones((2, 3, 1), dtype=dtype, requires_grad=True)
    expected = settle(spec, m, q, x, mode="hst"); eg, em = spec.embed(m)
    actual = spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True),
                                   9, sealed_until=9, mode="hst"))
    equivalent(expected, actual)
    leaves = dict(m.named_parameters()) | {"input": x}
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    equivalent(vjp(expected.continuation.states[0, 0].slots["memory"], leaves),
               vjp(actual.continuation.states[0, 0].slots["memory"], leaves))


@pytest.mark.parametrize("fault", ["clock", "count", "shape", "dtype", "nonfinite", "slot"])
def test_invalid_custom_next_result_is_rejected(dtype, fault):
    class Broken(ClockNext):
        def step(self, w, r):
            state = super().step(w, r)
            if fault == "clock":
                return replace(state, last_time=r.time+1)
            if fault == "count":
                return replace(state, observations=-1)
            if fault == "shape":
                return replace(state, value=state.value.reshape(()))
            if fault == "dtype":
                return replace(state, value=state.value.to(torch.float32 if dtype == torch.float64 else torch.float64))
            if fault == "nonfinite":
                return replace(state, value=state.value*float("nan"))
            return replace(state, slots={})
    g, m, q, xs, _ = custom_fixture(dtype)
    m.nodes[0].next_program = Broken(dtype)
    for execute in (run, frontier):
        with pytest.raises(ValueError, match="Next|SSM"):
            execute(g, m, q, xs, 5, sealed_until=5)


@pytest.mark.parametrize("native", [False, True])
def test_blend_rejects_incompatible_growing_attention_slots(dtype, native):
    g = Graph((Node(0, memory="attention", next_state="control-blend-v1"),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype); q = Continuation(g.identity, 1)
    xs = [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))]
    with pytest.raises(ValueError, match="matching slot shapes"):
        if native:
            Native(g, m).run(q, xs, 1, sealed_until=1)
        else:
            run(g, m, q, xs, 1, sealed_until=1)
