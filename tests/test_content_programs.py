import pytest
import torch
from tidegraph import Continuation, External, Graph, Node, Region
from tidegraph.checkpoint import save, load
from tidegraph.compare import equivalent, objective
from tidegraph.full import FullInput, FullProgram, FullResult
from tidegraph.frontier import run as frontier
from tidegraph.native import Native
from tidegraph.ops import Model
from tidegraph.records import State
from tidegraph.reference import run
from tidegraph.settle import SettleGraph, run as settle
from tidegraph.state_program import StateProgram
from isolated_cases import vjp


class SourceMemory(StateProgram):
    profile = "source-memory-v1"
    sequence_contract = True  # Exact scalar sequence is an explicit fallback.

    def __init__(self, dtype):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(2., dtype=dtype))

    def initial(self, w):
        return State(torch.zeros_like(w.bias), slots={"sum": torch.zeros_like(w.bias)})

    def step(self, w, old, content, time):
        value = sum((s.slot+1)*(s.atom.kind+1)*(s.atom.position+1)*s.atom.value*s.scale for s in content.sources)
        total = sum((slot+1)*value for slot, value in content.contributions.items())
        return State(old.value+self.gain*value, time, old.observations+1, {"sum": old.slots["sum"]+total})

    def read(self, w, old, proposal, content, time):
        return proposal.value.sum()+content.contributions[0].sum()

    def validate(self, w, state):
        if set(state.slots) != {"sum"} or state.slots["sum"].shape != w.bias.shape:
            raise ValueError("source memory slot mismatch")


class SourceFull(FullProgram):
    profile = "source-full-v1"

    def step(self, w, request, slots, mode, zeta):
        c = request.content
        source = sum((s.atom.kind+s.atom.position)*s.atom.value for s in c.sources)
        value = request.comparison.value+source+c.contributions[1]
        return FullResult(None, {0: value})


def fixture(dtype, clear=False):
    g = Graph((Node(0, clear=clear, memory="source-memory-v1", emission="source-full-v1"),),
              (), (Region(1),), (0, 0), (0,))
    m = Model(g, width=2, dtype=dtype, state_programs={0: SourceMemory(dtype)}, full_programs={0: SourceFull()})
    with torch.no_grad():
        for p in [*m.input_scale, *m.output_scale]:
            p.fill_(1)
    x = torch.ones(2, dtype=dtype, requires_grad=True); y = torch.full_like(x, 3., requires_grad=True)
    unused = torch.full_like(x, 0.4, requires_grad=True)
    xs = [External(b, p, i, time, value) for b in range(2) for p in range(2)
          for i, (time, value) in enumerate(((1, x if b == 0 else unused), (4, y if b == 0 else unused)))]
    return g, m, Continuation(g.identity, 2), xs, {"x": x, "y": y, "other": unused, "gain": m.nodes[0].kernel.gain}


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_complete_content_custom_state_read_and_full_hand_vjp(dtype, clear, prefill):
    g, m, q, xs, variables = fixture(dtype, clear)
    expected = run(g, m, q, xs, 5, sealed_until=5)
    g, m, q, xs, actual_variables = fixture(dtype, clear)
    actual = frontier(g, m, q, xs, 5, sealed_until=5, prefill=prefill)
    equivalent(expected, actual)
    for result, leaves in ((expected, variables), (actual, actual_variables)):
        loss = sum(v.sum() for b, _, _, v in result.outputs if b == 0)
        equivalent(loss, loss.new_tensor(104 if clear else 116))
        dx, dy, unused, gain = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True, retain_graph=True)
        equivalent(dx, torch.full_like(dx, 7 if clear else 13)); equivalent(dy, torch.full_like(dy, 15))
        assert unused is None
        equivalent(gain, gain.new_tensor(42 if clear else 48))
    equivalent(vjp(objective(expected), variables), vjp(objective(actual), actual_variables))
    for field in ("descriptor", "proposal"):
        equivalent(vjp(expected.trace[0][field], variables), vjp(actual.trace[0][field], actual_variables))
    if prefill and not clear:
        assert actual.stats["state_scalar_sequence_steps"] == 4


def test_complete_content_custom_state_settle_embedding(dtype):
    g, m, _, _, _ = fixture(dtype); spec = SettleGraph(g, (1,))
    x = torch.ones((2, 2, 2), dtype=dtype, requires_grad=True)
    variables = dict(m.named_parameters()) | {"input": x}
    expected = settle(spec, m, Continuation(g.identity, 2), x)
    eg, em = spec.embed(m)
    actual = spec.project(frontier(eg, em, Continuation(eg.identity, 2), spec.external(x, encoded=True), 6, sealed_until=6))
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), variables), vjp(objective(actual), variables))
    equivalent(vjp(expected.outputs[0][-1], variables), vjp(actual.outputs[0][-1], variables))


def test_registered_state_parameters_checkpoint_optimizer_and_native_guard(dtype, tmp_path):
    g, m, q, xs, _ = fixture(dtype)
    gain = m.nodes[0].kernel.gain
    assert dict(m.named_parameters())["nodes.0.kernel.gain"] is gain
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)
    result = frontier(g, m, q, xs, 5, sealed_until=5)
    objective(result).backward(); optimizer.step()
    assert gain in optimizer.state
    path = tmp_path / "content.pt"; save(path, g, m, result.continuation, optimizer)
    _, restored, _, _, _ = fixture(dtype); ro = torch.optim.AdamW(restored.parameters(), lr=0.01)
    loaded = load(path, g, restored, ro)
    equivalent(result.continuation.detach(), loaded)
    equivalent(m.state_dict(), restored.state_dict()); equivalent(optimizer.state_dict(), ro.state_dict())
    # A custom state with a built-in Full must also fail at the native boundary.
    basic = Graph((Node(0, memory=SourceMemory.profile),), (), (Region(1),), (0, 0), (0,))
    model = Model(basic, dtype=dtype, state_programs={0: SourceMemory(dtype)})
    with pytest.raises(ValueError, match="custom state.*no native"):
        Native(basic, model)


@pytest.mark.parametrize("kind", ["matrix", "attention"])
def test_shared_state_program_requires_matching_policy(dtype, kind):
    nodes = ((Node(0, memory="linear"), Node(0, memory="delta")) if kind == "matrix" else
             (Node(0, memory="attention", window=1), Node(0, memory="attention", window=2)))
    g = Graph(nodes, (), (Region(2),), (0, 1), (0, 1))
    m = Model(g, dtype=dtype)
    m.nodes[1] = m.nodes[0]
    with pytest.raises(ValueError, match="shared state program does not match"):
        run(g, m, Continuation(g.identity, 1), [], 0, sealed_until=0)
    with pytest.raises(ValueError, match="shared state program does not match"):
        Native(g, m)


def test_builtin_profile_python_override_is_not_silently_replaced(dtype):
    class Override(SourceMemory):
        profile = "ema"

    g = Graph((Node(0),), (), (Region(1),), (0, 0), (0,))
    m = Model(g, dtype=dtype, state_programs={0: Override(dtype)})
    run(g, m, Continuation(g.identity, 1), [], 0, sealed_until=0)
    with pytest.raises(ValueError, match="custom state.*no native"):
        Native(g, m)
