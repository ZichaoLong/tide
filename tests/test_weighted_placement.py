"""Architecture §3 / allocation §2: temporary placement and exact public roots.

CPU tests exercise the adapter and explicit precision. Actual accelerator
transfers, ownership and timing require separate target-device evidence.
"""
from dataclasses import replace
import pytest
import torch
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run as greedy
from tidegraph.reference import run as reference
from tidegraph.native import Native
from tidegraph.placement import ExecutionPlacement, place_model
from isolated_cases import vjp
from test_weighted_execution import setup, variables
from test_weighted_recurrent_execution import model
from test_weighted_sending import MODES
from test_weighted_sending_execution import configured as sending
from test_weighted_scoring_execution import configured as scoring


def runner(b, native, placement=None):
    placement = placement or ExecutionPlacement(read="cpu", control="cpu", selection="cpu")
    options = dict(packed=True, aggregate_autograd="batched", full_autograd="batched")
    if native:
        return Native(b.graph, b.model, algorithm="greedy", placement=placement, **options).run
    view = place_model(b.graph, b.model, placement)
    assert view.state_dict().keys() == b.model.state_dict().keys()
    assert all(p is dict(b.model.named_parameters())[n] for n, p in view.named_parameters())
    return lambda *a, **kw: greedy(b.graph, view, *a, **options, **kw)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("family", ["kv", "affine", "delta", "gated_delta", "ssd"])
@pytest.mark.parametrize("mode", ["0", "-", "+"])
def test_placed_weighted_full_snapshots_roots_and_retained_cut(dtype, native, family, mode):
    b, _, xs, stop = setup(dtype, "pdg", mode) if family == "kv" else model(dtype, family, mode)
    q = Continuation(b.graph.identity, 1)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    execute = runner(b, native)
    actual = execute(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    roots = variables(b, xs)
    equivalent(vjp(objective(expected), roots), vjp(objective(actual), roots))
    for name in expected.trace[0]["descriptor"]:
        equivalent(vjp(expected.trace[0]["descriptor"][name], roots), vjp(actual.trace[0]["descriptor"][name], roots))
    first = execute(q, [x for x in xs if x.time < 3], 3, sealed_until=3)
    rest = execute(first.continuation, [x for x in xs if x.time >= 3], stop, sealed_until=stop)
    equivalent(expected.continuation, rest.continuation)
    joined = replace(rest, outputs=first.outputs+rest.outputs)
    equivalent(vjp(objective(expected), roots), vjp(objective(joined), roots))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("mode", MODES)
def test_placed_sending_all_formal_candidates(dtype, native, mode):
    b, xs, stop = sending(dtype, mode, "dag")
    q = Continuation(b.graph.identity, 1)
    expected = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    actual = runner(b, native)(q, xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    equivalent(vjp(objective(expected), variables(b, xs)), vjp(objective(actual), variables(b, xs)))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_placed_heterogeneous_attention_training_checkpoint(dtype, native, tmp_path):
    from tidegraph.checkpoint import save, load
    a, xa, stop = scoring(dtype, 3, "hb", tie=True)
    b, xb, _ = scoring(dtype, 3, "hb", tie=True)
    oa = torch.optim.AdamW(a.model.parameters(), lr=.001)
    ob = torch.optim.AdamW(b.model.parameters(), lr=.001)
    run = runner(b, native)
    for _ in range(3):
        oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
        ra = reference(a.graph, a.model, Continuation(a.graph.identity, 1), xa, stop, sealed_until=stop)
        rb = run(Continuation(b.graph.identity, 1), xb, stop, sealed_until=stop)
        equivalent(ra, rb); objective(ra).backward(); objective(rb).backward()
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            equivalent(p.grad, q.grad)
        oa.step(); ob.step(); equivalent(a.model.state_dict(), b.model.state_dict())
    first = run(Continuation(b.graph.identity, 1), [x for x in xb if x.time < 3], 3, sealed_until=3)
    save(tmp_path/"placed.pt", b.graph, b.model, first.continuation, ob)
    c, _, _ = scoring(dtype, 3, "hb", tie=True)
    oc = torch.optim.AdamW(c.model.parameters(), lr=.001)
    restored = load(tmp_path/"placed.pt", c.graph, c.model, oc)
    equivalent(first.continuation.detach(), restored); equivalent(ob.state_dict(), oc.state_dict())
    tail = [x for x in xb if x.time >= 3]
    for item, state, opt, execute in ((b, first.continuation.detach(), ob, run), (c, restored, oc, runner(c, native))):
        opt.zero_grad(set_to_none=True)
        result = execute(state, tail, stop, sealed_until=stop)
        objective(result).backward(); opt.step()
    equivalent(b.model.state_dict(), c.model.state_dict()); equivalent(ob.state_dict(), oc.state_dict())


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_explicit_fp64_scoring_from_fp32_payload_matches_independent_schedule(native):
    b, xs, stop = scoring(torch.float32, 3, "dag", tie=True)
    placement = ExecutionPlacement(read="cpu", control="cpu", selection="cpu", scoring_dtype="float64")
    expected = reference(b.graph, place_model(b.graph, b.model, placement), Continuation(b.graph.identity, 1), xs, stop, sealed_until=stop)
    actual = runner(b, native, placement)(Continuation(b.graph.identity, 1), xs, stop, sealed_until=stop)
    equivalent(expected, actual)
    assert all(t.dtype == torch.float64 for event in actual.trace for t in event["descriptor"].values())
    assert all(event["control"].dtype == torch.float32 for event in actual.trace)
    equivalent(vjp(objective(expected), variables(b, xs)), vjp(objective(actual), variables(b, xs)))


def test_scoring_cast_literal_formula_initial_snapshot_and_version_guard():
    from tidegraph.content import Content
    from tidegraph.readout import ReadInput
    from tidegraph.weighted import KVConfig, RegionalConfig
    from test_weighted_kv_formulas import fixture
    b = fixture(torch.float32, memories=KVConfig(width=1, summary_width=1, epsilon=1.), regions=RegionalConfig(read_mode="0"))
    view = place_model(b.graph, b.model, ExecutionPlacement(scoring_dtype="float64"))
    w, read = view.nodes[0], b.model.nodes[0].read_program
    x = torch.tensor([.4, 2.], requires_grad=True)
    r = ReadInput(None, 2**54+17, Content(x))
    actual = w.read_program.batch_grad(w, [r])[0]
    v = x.double(); zero = v.new_zeros(())
    normal = v[1]/torch.sqrt(1+v[1].square())
    current = (b.model.nodes[0].kernel.summary.double()@normal.reshape(1))[0]
    features = torch.stack((normal, v[0].log1p(), zero, zero, zero, zero, current*zero, current*zero))
    h = read.score.matrix.double()@features+read.score.hidden_bias.double()
    expected = read.score.bias.double()+(h*h.sigmoid()*read.score.vector.double()).sum()
    torch.testing.assert_close(actual["score"], expected)
    roots = {n: p for n, p in b.model.named_parameters() if p.requires_grad} | dict(x=x)
    equivalent(vjp(actual["score"], roots), vjp(expected, roots))
    with torch.no_grad():
        read.score.matrix.add_(.1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        actual["score"].backward()


def test_control_cast_rejects_loss_of_positive_support():
    from types import SimpleNamespace
    from tidegraph.history import History
    from tidegraph.region import Selection
    from tidegraph.weighted.placement_tensors import controls
    selection = Selection({0}, {0: torch.tensor(1e-100, dtype=torch.float64)}, History())
    with pytest.raises(ValueError, match="positive support"):
        controls(selection, SimpleNamespace(payload_device=torch.device("cpu"), payload_dtype=torch.float32))


def test_packed_placement_copies_independent_roots_aliases_int64_and_capacity(monkeypatch):
    from tidegraph.weighted import placement_tensors as transfer
    monkeypatch.setattr(transfer, "PACK_BYTES", 32)  # Two width-2 FP64 destination rows per group.
    xs = [torch.tensor([.25, .5], requires_grad=True) for _ in range(5)]
    clock = torch.tensor([2**54+3, -1], dtype=torch.int64)
    f = transfer.mapper([*xs, xs[0], clock], torch.device("cpu"), torch.float64)
    assert f(xs[0]) is f(xs[0])
    assert f(clock).dtype == torch.int64 and torch.equal(f(clock), clock)
    loss = f(xs[0]).sum()+3*f(xs[0]).sum()+0*f(xs[1]).sum()+2*f(xs[4]).sum()
    grads = torch.autograd.grad(loss, xs, retain_graph=True, allow_unused=True)
    for actual, coefficient in zip(grads, (4, 0, None, None, 2)):
        if coefficient is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, torch.full_like(xs[0], coefficient))
    with torch.no_grad():
        xs[0].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        loss.backward()
