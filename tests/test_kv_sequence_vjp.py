"""Untraced KV sequence VJP: all exposed roots, causal None, exact raw cuts."""
import os
import pytest
import torch
from tidegraph import Continuation, External
from tidegraph.compare import equivalent, objective
from tidegraph.greedy import run
from tidegraph.reference import run as reference
from tidegraph.runtime import resolve_device
from test_weighted_execution import setup, variables
from isolated_cases import vjp


@pytest.mark.parametrize("read_mode", ["0", "-", "+"])
@pytest.mark.parametrize("topology", ["fc", "hb", "pdg"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_untraced_sequence_all_exposed_roots(dtype, read_mode, topology, native, monkeypatch):
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    b, spec, _, _ = setup(dtype, topology, read_mode=read_mode)
    b.model.to(device)
    stride = spec.stride if spec else 1
    xs = [External(s, port, i, 1+t*stride, torch.tensor([mass, .3+.03*i, -.2],
                    dtype=dtype, device=device, requires_grad=True))
          for s in range(2) for port in range(len(b.graph.inputs))
          for i, (t, mass) in enumerate(((0, .4), (1, 1.2), (3, 0.), (7, .6))[:4-s])]
    stop = 9*stride
    q = Continuation(b.graph.identity, 2)
    # Actual optimization must be used on the layered cases, without a step VJP.
    if topology != "pdg":
        import tidegraph.state_vjp as sv
        monkeypatch.setattr(sv, "bind_sequence", lambda *a: pytest.fail("unexpected step VJP"))
    a = reference(b.graph, b.model, q, xs, stop, sealed_until=stop)
    from tidegraph import ExecutionPlacement
    from tidegraph.weighted.execution import eager_runner
    execute = eager_runner(b, "native" if native else "python",
        ExecutionPlacement(preset="cpu" if device.type == "cpu" else "mixed-c",
                           scoring_dtype=str(dtype).split(".")[-1]),
        packing="packed", trace=False)
    c = execute(q, xs, stop, sealed_until=stop)
    if topology != "pdg":
        assert c.stats["joint_state_vjp_events"] == c.stats["candidate_events"]
    equivalent(a.outputs, c.outputs)
    equivalent(a.continuation, c.continuation)
    roots = variables(b, xs)
    pairs = [(a.outputs[0][-1], c.outputs[0][-1]), (a.outputs[-1][-1], c.outputs[-1][-1])]
    for owner, state in c.continuation.states.items():
        for name, value in state.slots.items():
            if value.is_floating_point():
                pairs.append((a.continuation.states[owner].slots[name], value))
    for x, y in pairs:
        for zero in (False, True):
            equivalent(vjp(x*0 if zero else x, roots), vjp(y*0 if zero else y, roots))


def test_prefix_adjoint_keeps_absent_and_connected_expired_roots(dtype):
    from tidegraph.weighted.kv_sequence_vjp import reduce_prefixes
    ref = torch.empty(3, 2, dtype=dtype)
    bars = [torch.tensor([[1., 2.]], dtype=dtype), None, torch.tensor([[3., 4.]], dtype=dtype)]
    value, last = reduce_prefixes(bars, [(0, 1), (0, 2), (2, 3)], 4, ref)
    equivalent(value, torch.tensor([[1., 2.], [0., 0.], [3., 4.], [0., 0.]], dtype=dtype))
    assert last == 3
    assert reduce_prefixes([None]*3, [(0, 1)]*3, 4, ref) == (None, -1)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_untraced_warm_state_versions_and_exact_training_resume(dtype, native, tmp_path):
    from dataclasses import replace
    from tidegraph.checkpoint import save, load
    from tidegraph.native import Native
    b, spec, xs, stop = setup(dtype, "hb")
    device, _ = resolve_device(os.environ.get("TIDE_WINDOW_DEVICE", "cpu"))
    b.model.to(device)
    xs = [replace(x, value=x.value.detach().to(device).requires_grad_()) for x in xs]
    def execute(binding, q, inputs, stop):
        if native:
            return Native(binding.graph, binding.model, algorithm="window", packed=True, trace=False,
                          full_autograd="batched", aggregate_autograd="batched").run(q, inputs, stop, sealed_until=stop)
        return run(binding.graph, binding.model, q, inputs, stop, sealed_until=stop, trace=False,
                   component_window=True, full_autograd="batched", aggregate_autograd="batched")
    cut = spec.stride+1
    head = execute(b, Continuation(b.graph.identity, 1), [x for x in xs if x.time < cut], cut)
    q = head.continuation.detach()
    roots = {}
    for owner, state in q.states.items():
        for name, t in state.slots.items():
            if t.is_floating_point():
                roots[str((owner, name))] = t.requires_grad_()
    tail = [x for x in xs if x.time >= cut]
    expected = reference(b.graph, b.model, q, tail, stop, sealed_until=stop)
    actual = execute(b, q, tail, stop)
    def loss(r):
        return sum(t.square().sum() for *_, t in r.outputs)+sum(
            t.square().sum()*.01 for s in r.continuation.states.values() for t in s.slots.values() if t.is_floating_point())
    equivalent(vjp(loss(expected), roots), vjp(loss(actual), roots))
    optimizer = torch.optim.SGD(b.model.parameters(), lr=1e-5, momentum=.3)
    optimizer.zero_grad(set_to_none=True); loss(actual).backward(); optimizer.step()
    save(tmp_path/"kv.pt", b.graph, b.model, actual.continuation.detach(), optimizer)
    c, _, _, _ = setup(dtype, "hb"); c.model.to(device)
    other = torch.optim.SGD(c.model.parameters(), lr=1e-5, momentum=.3)
    restored = load(tmp_path/"kv.pt", c.graph, c.model, other)
    for step in range(2):
        records = []
        for binding, initial, opt in ((b, actual.continuation.detach(), optimizer), (c, restored, other)):
            # New independent window after a checkpoint, same original-time code.
            shifted = [External(x.batch, x.port, x.position+2+2*step, x.time+stop*(step+1), x.value.detach()) for x in xs]
            opt.zero_grad(set_to_none=True)
            r = execute(binding, initial, shifted, stop*(step+2))
            loss(r).backward(); records.append(r); opt.step()
        equivalent(records[0], records[1], atol=0, rtol=0)
        equivalent(b.model.state_dict(), c.model.state_dict(), atol=0, rtol=0)
        equivalent(optimizer.state_dict(), other.state_dict(), atol=0, rtol=0)
        actual, restored = records[0], records[1].continuation.detach()
    final = execute(b, Continuation(b.graph.identity, 1), xs, stop)
    with torch.no_grad():
        b.model.nodes[0].kernel.summary.add_(.01)
    with pytest.raises((RuntimeError, ValueError), match="modified|version|inplace"):
        loss(final).backward()
