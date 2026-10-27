"""Core-5 finite products / weighted architecture §2–3 representation anchors.

These custom formulas test state/Read/Full visibility and representation, not
the KV/allocation formulas. Field-local roots expose accidental zero gradients.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Atom, Continuation, Edge, External, Graph, Node, Region, State
from tidegraph.checkpoint import load, save
from tidegraph.compare import equivalent, objective
from tidegraph.frontier import run as frontier
from tidegraph.greedy import run as greedy
from tidegraph.history import History
from tidegraph.ops import Model
from tidegraph.readout import ReadProgram
from tidegraph.reference import run as reference
from tidegraph.region import RegionProgram, Selection
from tidegraph.source_class import SourceClasses
from tidegraph.state_program import StateProgram
from tidegraph.tensor_fields import PackedFields, pack_fields
from isolated_cases import vjp


class TimestampState(StateProgram):
    profile = "typed-clock-test-v1"
    sequence_contract = True
    integer_slots = frozenset({"times"})

    def initial(self, w):
        return State(torch.zeros_like(w.bias), slots={"times": torch.empty(0, device=w.bias.device, dtype=torch.int64)})

    def step(self, w, old, content, time):
        stamp = torch.tensor([time], dtype=torch.int64, device=content.value.device)
        return State(.5*old.value+content.value, time, old.observations+1,
                     {"times": torch.cat((old.slots["times"], stamp))})

    def validate(self, w, state):
        if (state.slots.keys() != {"times"} or state.slots["times"].ndim != 1
                or len(state.slots["times"]) != state.observations):
            raise ValueError("invalid timestamp sequence")


class NamedRead(ReadProgram):
    profile = "named-read-test-v1"
    joint_batch = True
    field_layout = {"score": (), "memory": (2,), "evidence": (2,)}

    def __init__(self, dtype):
        super().__init__()
        self.bound_initial = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=dtype))
        self.evidence_gain = torch.nn.Parameter(torch.tensor(.3, dtype=dtype))

    def step(self, w, request):
        memory = self.bound_initial if request.state is None else request.state.value
        return dict(score=(memory*w.read).sum(), memory=memory,
                    evidence=self.evidence_gain*request.content.value)

    def batch(self, w, requests):
        memory = torch.stack([self.bound_initial if r.state is None else r.state.value for r in requests])
        scores = (memory*w.read).sum(-1)
        evidence = self.evidence_gain*torch.stack([r.content.value for r in requests])
        return [dict(score=s, memory=m, evidence=e) for s, m, e in zip(scores, memory, evidence)]


class NamedSelector(RegionProgram):
    profile = "named-selector-test-v1"

    def step(self, r):
        return Selection({v for v, _ in r.candidates},
                         {v: (d["score"]+d["evidence"].sum()).sigmoid() for v, d in r.candidates},
                         History(last_time=r.time))


def setup(dtype, read_mode, family):
    g = Graph(tuple(Node(i, memory=TimestampState.profile, readout=NamedRead.profile) for i in range(2)),
              (Edge(0, 1, 1), Edge(0, 1, 3)) + ((Edge(1, 0, 2),) if family == "pdg" else ()),
              tuple(Region(1, read_mode=read_mode, selector=NamedSelector.profile) for _ in range(2)), (0, 1), (0, 1))
    shared = NamedRead(dtype)
    m = Model(g, width=2, dtype=dtype, state_programs={i: TimestampState() for i in range(2)},
              read_programs={i: shared for i in range(2)}, region_programs={i: NamedSelector() for i in range(2)})
    clock = 2**54+17
    q = Continuation(g.identity, 1, cut=clock)
    variables = dict(m.named_parameters())
    for i in range(2):
        state = m.nodes[i].initial()
        state.value = torch.tensor([.7+.1*i, .9], dtype=dtype, requires_grad=True)
        variables[f"initial.{i}"] = state.value; q.states[0, i] = state
    xs = []
    for port, times in enumerate(((clock, clock+2), (clock+1,))):
        for occurrence, time in enumerate(times):
            x = torch.tensor([.6-.1*port, -.1], dtype=dtype, requires_grad=True)
            variables[f"input.{port}.{occurrence}"] = x
            xs.append(External(0, port, occurrence, time, x))
    return g, m, q, xs, variables


@pytest.mark.parametrize("read_mode", ["content", "old", "proposal"])
@pytest.mark.parametrize("family", ["pdg", "dag"])
def test_named_describe_integer_state_and_snapshot_vjp(dtype, read_mode, family, tmp_path):
    g, m, q, xs, variables = setup(dtype, read_mode, family)
    stop = q.cut+6
    expected = reference(g, m, q, xs, stop, sealed_until=stop, mode="softp")
    candidate = greedy if family == "pdg" else frontier
    actual = candidate(g, m, q, xs, stop, sealed_until=stop, mode="softp")
    equivalent(expected, actual)
    first = actual.trace[0]
    initial = q.states[0, 0].value
    proposal = .5*initial+xs[0].value*m.input_scale[0]
    read_state = m.nodes[0].read_program.bound_initial if read_mode == "content" else initial if read_mode == "old" else proposal
    equivalent(first["descriptor"]["memory"], read_state)
    equivalent(first["comparison"], proposal)  # Full always reads proposal in this binding.
    for key in NamedRead.field_layout:
        equivalent(vjp(first["descriptor"][key], variables), vjp(expected.trace[0]["descriptor"][key], variables))
    gradients = vjp(first["descriptor"]["score"], variables)
    assert gradients["nodes.0.read_program.evidence_gain"] is None
    assert (gradients["nodes.0.read_program.bound_initial"] is None) == (read_mode != "content")
    assert (gradients["initial.0"] is None) == (read_mode == "content")
    stamps = actual.continuation.states[0, 0].slots["times"]
    assert stamps.dtype == torch.int64 and stamps[0].item() == q.cut and not stamps.requires_grad
    path = tmp_path/"typed.pt"
    save(path, g, m, actual.continuation)
    rg, rm, _, _, _ = setup(dtype, read_mode, family)
    restored = load(path, rg, rm)
    equivalent(actual.continuation.detach(), restored)
    # Corrupting metadata into a float must fail before execution/ownership changes.
    invalid = restored.fork(); state = invalid.states[0, 0]
    invalid.states[0, 0] = replace(state, slots={"times": stamps.to(dtype)})
    with pytest.raises(ValueError, match="slot dtype"):
        candidate(g, m, invalid, [], stop, sealed_until=stop)
    assert torch.isfinite(objective(actual))  # Integer timestamps are not loss roots.


def test_typed_packing_preserves_values_int64_and_independent_field_roots(dtype):
    a = torch.tensor([.2, -.5], dtype=dtype, requires_grad=True)
    b = torch.tensor(.7, dtype=dtype, requires_grad=True)
    fields = {"score": b.square(), "vector": 3*a, "timestamps": torch.tensor([2**60+3], dtype=torch.int64)}
    packed = pack_fields(fields)
    assert len(packed.buffers) == 2
    decoded = packed.decode(); equivalent(fields, decoded)
    equivalent(vjp(decoded["vector"], {"a": a, "b": b}), {"a": 12.6*a, "b": None})
    detached = PackedFields(packed.layout, packed.buffers).decode()
    equivalent(fields["timestamps"], detached["timestamps"])
    assert not detached["vector"].requires_grad
    corrupted = list(packed.buffers)
    bank = next(i for i, t in enumerate(corrupted) if t.dtype == torch.int64)
    corrupted[bank] = corrupted[bank]+1
    with pytest.raises(ValueError, match="integer metadata"):
        PackedFields(packed.layout, tuple(corrupted), packed.roots).decode()


def test_source_class_is_not_source_domain(dtype):
    g = Graph((Node(0), Node(1)), (Edge(0, 1, 1), Edge(0, 1, 1)),
              (Region(1), Region(1)), (1,), (1,))
    classes = SourceClasses((0, 0), (0,)); classes.validate(g)
    x = torch.tensor([2.], dtype=dtype, requires_grad=True)
    atoms = [Atom(0, 1, 1, 1, edge, 0, x) for edge in range(2)] + [Atom(0, 1, 1, 0, 0, 0, x)]
    coefficients = torch.tensor([3.], dtype=dtype, requires_grad=True)
    total = sum(a.value*coefficients[classes.classify(a)] for a in atoms)
    equivalent(total, 9*x)
    equivalent(torch.autograd.grad(total.sum(), (x, coefficients)), (torch.tensor([9.], dtype=dtype), 3*x))
    assert len({a.key() for a in atoms}) == 3 and {classes.classify(a) for a in atoms} == {0}


@pytest.mark.parametrize("fault", ["field", "shape", "dtype", "nonfinite", "count"])
def test_structured_batch_rejects_changed_domain(dtype, fault):
    class Broken(NamedRead):
        def batch(self, w, requests):
            rows = super().batch(w, requests)
            if fault == "field": rows[0].pop("evidence")
            if fault == "shape": rows[0]["evidence"] = rows[0]["evidence"][:1]
            if fault == "dtype": rows[0]["score"] = rows[0]["score"].to(torch.int64)
            if fault == "nonfinite": rows[0]["score"] = rows[0]["score"]*float("nan")
            if fault == "count": rows.pop()
            return rows
    g, m, q, xs, _ = setup(dtype, "proposal", "dag")
    m.nodes[0].read_program = Broken(dtype)
    with pytest.raises(ValueError, match="Read|selector"):
        frontier(g, m, q, xs, q.cut+5, sealed_until=q.cut+5)
