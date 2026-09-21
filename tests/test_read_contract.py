from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region, State
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.readout import ReadProgram, LinearRead
from tidegraph.reference import run
from isolated_cases import vjp
from read_cases import fixture


class SlotRead(ReadProgram):
    profile = "slot-read-v1"

    def __init__(self, dtype, mode):
        super().__init__()
        self.mode = mode
        self.gain = torch.nn.Parameter(torch.tensor(0.7, dtype=dtype))

    def step(self, w, r):
        assert bool(r.content.sources) and r.content.contributions
        value = r.content.contributions[0].sum() + r.time
        if self.mode == "content":
            assert r.state is None
        else:
            assert r.state.last_time < r.time if self.mode == "old" else r.state.last_time == r.time
            value = value + r.state.slots["memory"].sum() + r.state.last_time + r.state.observations
        return self.gain * value


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
def test_custom_read_slots_clocks_registration_sharing_and_checkpoint(dtype, read_mode, tmp_path):
    g = Graph((Node(0, memory="ssm", readout=SlotRead.profile), Node(0, memory="ssm", readout=SlotRead.profile)),
              (), (Region(1, read_mode=read_mode),), (0, 1), (0, 1))
    program = SlotRead(dtype, read_mode)
    m = Model(g, dtype=dtype, read_programs={0: program, 1: program})
    assert m.nodes[0].read_program is m.nodes[1].read_program
    assert dict(m.named_parameters())["nodes.0.read_program.gain"] is program.gain
    q = Continuation(g.identity, 2)
    x = torch.ones(3, dtype=dtype, requires_grad=True)
    other = torch.full_like(x, 2., requires_grad=True)
    xs = [External(b, v, i, time, x if b == 0 else other) for b in range(2) for v in range(2)
          for i, time in enumerate((1, 4))]
    leaves = dict(m.named_parameters()) | {"input": x, "other": other}
    expected = run(g, m, q, xs, 5, sealed_until=5, mode="hst")
    actual = frontier(g, m, q, xs, 5, sealed_until=5, mode="hst")
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    equivalent(vjp(expected.outputs[0][-1], leaves), vjp(actual.outputs[0][-1], leaves))
    assert actual.stats["read_scalar_batch_steps"] == 8
    with torch.no_grad():
        equivalent(actual, frontier(g, m, q, xs, 5, sealed_until=5, mode="hst"))
    optimizer = torch.optim.AdamW(m.parameters(), lr=.01)
    objective(actual).backward(); optimizer.step()
    assert program.gain in optimizer.state
    path = tmp_path / "read.pt"; save(path, g, m, actual.continuation, optimizer)
    reader = SlotRead(dtype, read_mode)
    restored = Model(g, dtype=dtype, read_programs={0: reader, 1: reader})
    ro = torch.optim.AdamW(restored.parameters(), lr=.01)
    equivalent(actual.continuation.detach(), load(path, g, restored, ro))
    equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())
    with pytest.raises(ValueError, match="custom Read.*no native"):
        Native(g, m)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite", "count"])
def test_invalid_read_batch_is_rejected(dtype, fault):
    class Broken(LinearRead):
        def batch(self, w, requests):
            result = super().batch(w, requests)
            if fault == "shape":
                result[0] = result[0].reshape(1)
            elif fault == "dtype":
                result[0] = result[0].to(torch.float32 if dtype == torch.float64 else torch.float64)
            elif fault == "nonfinite":
                result[0] = result[0] * float("nan")
            else:
                result.pop()
            return result
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype, read_programs={0: Broken()})
    with pytest.raises(ValueError, match="Read"):
        frontier(g, m, Continuation(g.identity, 1), [External(0, 0, 0, 0, torch.ones(3, dtype=dtype))], 1, sealed_until=1)
    with pytest.raises(ValueError, match="custom Read.*no native"):
        Native(g, m)


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
def test_read_modes_cursor_cuts_detach_and_graph_identity(dtype, read_mode, tmp_path):
    g, m, q, xs, leaves = fixture(dtype, read_mode)
    engine = Native(g, m, workers=3, packed=True, mode="hst")
    expected = engine.run(q, xs, 6, sealed_until=6)
    cursor = engine.cursor(q)
    cursor.advance([x for x in xs if x.time < 4], 4, sealed_until=4)
    prefix = cursor.snapshot()
    tail = cursor.advance([x for x in xs if x.time >= 4], 6, sealed_until=6)
    equivalent(expected.continuation, cursor.snapshot())
    equivalent(expected.trace[6:], tail.trace)
    equivalent(vjp(expected.outputs[-1][-1], leaves), vjp(tail.outputs[-1][-1], leaves))
    detached = engine.run(prefix.detach(), [x for x in xs if x.time >= 4], 6, sealed_until=6)
    for name, grad in vjp(objective(detached), leaves).items():
        if name.startswith("initial.") or name.startswith("input.") and name.endswith(".0"):
            assert grad is None
    path = tmp_path / "modes.pt"; save(path, g, m, prefix)
    changed = replace(g, regions=(replace(g.regions[0], read_mode="old" if read_mode != "old" else "content"),))
    restored = Model(changed, width=1, dtype=dtype, seed=101)
    before = {name: p.clone() for name, p in restored.state_dict().items()}
    with pytest.raises(ValueError, match="graph"):
        load(path, changed, restored)
    equivalent(before, restored.state_dict())
    resumed = engine.run(load(path, g, m), [x for x in xs if x.time >= 4], 6, sealed_until=6)
    equivalent(detached, resumed)


def test_invalid_read_mode_and_shared_profile_are_rejected(dtype):
    with pytest.raises(ValueError, match="Read mode"):
        Graph((Node(0),), (), (Region(1, read_mode="unknown"),), (0,), (0,))
    g = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    m = Model(g, dtype=dtype)
    m.nodes[0].read_program = SlotRead(dtype, "old")
    with pytest.raises(ValueError, match="Read program.*profile"):
        run(g, m, Continuation(g.identity, 1), [], 0, sealed_until=0)
    core = __import__("_tide_native")
    ng = Native(g, Model(g, dtype=dtype)).compiled
    ng.regions = [core.Region(1, True, True, "unknown")]
    with pytest.raises(ValueError, match="Read mode"):
        ng.compile()
