from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation, External, Graph, History, Node, Region
from tidegraph.compare import equivalent, objective
from tidegraph.full import FullProgram, FullResult
from tidegraph.history import increment
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.region import RegionProgram, Selection
from tidegraph.reference import run
from tidegraph.frontier import run as frontier
from isolated_cases import vjp
from region_cases import fixture, linear_vjp


class ClockSelector(RegionProgram):
    profile = "clock-selector-v1"

    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2., dtype=dtype))

    def initial(self, layout, reference):
        return History(scalars={"calls": 0, "signed": -2}, node_maps={"seen": {}}, tensors={"memory": reference.new_ones(())})

    def step(self, r):
        h = r.history.fork(); h.last_time = r.time
        h.scalars["calls"] = increment(h.scalars["calls"])
        for v, _ in r.candidates:
            h.node_maps["seen"][v] = increment(h.node_maps["seen"].get(v, 0))
        h.tensors["memory"] = self.gain*r.history.tensors["memory"] + sum(d for _, d in r.candidates) + r.time
        controls = {v: torch.stack([r.history.tensors["memory"], d]) for v, d in r.candidates}
        active = {r.candidates[0][0]} if r.history.scalars["calls"] % 2 else set()
        return Selection(active, controls, h)


class VectorFull(FullProgram):
    profile = "vector-control-v1"

    def step(self, weights, request, slots, mode, zeta):
        value = request.content.value + request.control.sum()
        return FullResult(value, {slot: value for slot in range(slots)})


def test_python_custom_selector_vector_controls_history_and_empty_selection(dtype):
    g = Graph((Node(0, emission="vector-control-v1"),), (),
              (Region(1, selector="clock-selector-v1", read_mode="content"),), (0,), (0,))
    p = ClockSelector(dtype)
    m = Model(g, width=1, dtype=dtype, region_programs={0: p}, full_programs={0: VectorFull()})
    with torch.no_grad():
        m.nodes[0].read.fill_(1); m.input_scale[0].fill_(1); m.output_scale[0].fill_(1)
    h = torch.tensor(1., dtype=dtype, requires_grad=True)
    history = p.initial(g.region_layouts[0], h); history.tensors["memory"] = h
    q = Continuation(g.identity, 1, history={(0, 0): history})
    x = torch.tensor([3.], dtype=dtype, requires_grad=True)
    y = torch.tensor([2.], dtype=dtype, requires_grad=True)
    xs = [External(0, 0, 0, 1, x), External(0, 0, 1, 4, y)]
    expected = run(g, m, q, xs, 6, sealed_until=6)
    actual = frontier(g, m, q, xs, 6, sealed_until=6)
    equivalent(expected, actual)
    assert len(actual.outputs) == 1 and not actual.trace[0]["active"] and actual.trace[1]["active"]
    result = actual.continuation.history[0, 0]
    equivalent(result.tensors["memory"], torch.tensor(18., dtype=dtype))
    assert result.scalars == {"calls": 2, "signed": -2} and result.node_maps == {"seen": {0: 2}}
    leaves = dict(m.named_parameters()) | {"history": h, "x": x, "y": y}
    gradients = linear_vjp(result.tensors["memory"], leaves)
    for name, value in {"history": 4., "x": 2., "y": 1., "regions.0.gain": 8.}.items():
        equivalent(gradients[name], torch.full_like(leaves[name], value))
    equivalent(vjp(objective(expected), leaves), vjp(objective(actual), leaves))
    # Adapter must not replace a Python override with the graph's built-in profile.
    plain = Graph((Node(0),), (), (Region(1),), (0,), (0,))
    p.profile = "count-v1"
    pm = Model(plain, dtype=dtype, region_programs={0: p})
    with pytest.raises(ValueError, match="custom region"):
        Native(plain, pm)


@pytest.mark.parametrize("malformed", ["subset", "capacity", "missing", "extra", "nan", "dtype", "clock",
                                      "int64", "node", "slot", "names"])
def test_selector_rejects_malformed_outputs_without_mutating_input(dtype, malformed):
    class Broken(RegionProgram):
        profile = "broken-v1"

        def step(self, r):
            controls = {v: d for v, d in r.candidates}; h = r.history.fork(); active = {0}
            if malformed == "subset": active = {9}
            if malformed == "capacity": active = {0, 1}
            if malformed == "missing": controls.pop(0)
            if malformed == "extra": controls[9] = r.candidates[0][1]
            if malformed == "nan": controls[0] = controls[0]*float("nan")
            if malformed == "dtype": controls[0] = controls[0].to(torch.int64)
            if malformed == "clock": h.last_time = r.time+1
            if malformed == "int64": h.scalars["bad"] = 2**63
            if malformed == "node": h.node_maps["bad"] = {9: 0}
            if malformed == "slot": h.tensors["bad"] = torch.tensor(float("inf"), dtype=dtype)
            if malformed == "names": h.scalars[""] = 0
            return Selection(active, controls, h)
    g = Graph((Node(0), Node(0)), (), (Region(1, selector="broken-v1"),), (0, 1), ())
    m = Model(g, dtype=dtype, region_programs={0: Broken()}); q = Continuation(g.identity, 1)
    xs = [External(0, v, 0, 0, torch.ones(3, dtype=dtype)) for v in range(2)]
    for executor in (run, frontier):
        with pytest.raises(ValueError, match="region"):
            executor(g, m, q, xs, 1, sealed_until=1)
    assert not q.history and not q.states and not q.ledger


@pytest.mark.parametrize("implementation", ["reference", "native"])
@pytest.mark.parametrize("malformed", ["layout", "clock", "count", "node", "nonfinite"])
def test_imported_history_validation(dtype, implementation, malformed):
    g, m, q, xs, _ = fixture(dtype)
    h = q.history[0, 0]
    if malformed == "layout": h.tensors["memory"] = torch.zeros(2, dtype=dtype)
    if malformed == "clock": h.last_time = 0
    if malformed == "count": h.node_maps["selected"][0] = -1
    if malformed == "node": h.node_maps["selected"][3] = 0
    if malformed == "nonfinite": h.tensors["memory"] = torch.tensor(float("nan"), dtype=dtype)
    with pytest.raises(ValueError, match="history|membership"):
        if implementation == "native":
            Native(g, m).run(q, xs, 6, sealed_until=6)
        else:
            run(g, m, q, xs, 6, sealed_until=6)


def test_shared_region_weights_reject_wrong_membership_size(dtype):
    g, m, q, xs, _ = fixture(dtype); m.regions[1] = m.regions[0]
    for call in (lambda: Native(g, m), lambda: run(g, m, q, xs, 6, sealed_until=6)):
        with pytest.raises(ValueError, match="membership"):
            call()


@pytest.mark.parametrize("consumer", ["Full", "Next"])
def test_scalar_profiles_reject_vector_controls(dtype, consumer):
    g = Graph((Node(0, next_state="control-blend-v1" if consumer == "Next" else "adopt-v1"),), (),
              (Region(1, selector="clock-selector-v1"),), (0,), (0,))
    m = Model(g, dtype=dtype, region_programs={0: ClockSelector(dtype)})
    xs = [External(0, 0, i, i, torch.ones(3, dtype=dtype)) for i in range(2)]
    for executor in (run, frontier):
        with pytest.raises(ValueError, match="scalar control"):
            executor(g, m, Continuation(g.identity, 1), xs, 2, sealed_until=2)
