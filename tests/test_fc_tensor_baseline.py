"""Literal dense FC schedule, scalar oracle and finite-interval allocation."""
import os
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fc_tensor_baseline import run, quadratic
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.settle import SettleGraph
from tidegraph.weighted import layered_inputs, readout


def fixture(dtype, device="cpu"):
    from tidegraph.planning.recipes import prepare
    from tidegraph.planning.materialize import materialize
    plan, _ = prepare(dict(topology="fc", family="kv", width=4, depth=2, region_width=4,
                           sparse_denominator=2, memory_overrides=dict(memory_read="qry"),
                           region_overrides=dict(allocation_temperature=.25)), samples=2, positions=3, seed=7)
    bound, _, _ = materialize(plan, max_bytes=2**25, dtype=dtype)
    bound.model.to(device)
    return bound, SettleGraph(bound.graph, tuple(plan["graph"]["ranks"]), binding="ports", time_stride=plan["stride"])


def test_finite_interval_solution_and_implicit_direction(dtype):
    scores = torch.tensor([[.1, .4, -.2, .6], [4., 3., .5, .4]], dtype=dtype, requires_grad=True)
    caps = torch.tensor([[.8, .9, .5, .9], [.7, .7, .7, .7]], dtype=dtype, requires_grad=True)
    target = torch.tensor([1.5, 1.6], dtype=dtype, requires_grad=True)
    result = quadratic(scores, caps, target, .7)
    # Independent bisection only in this test, not the performance candidate.
    left, right = (scores-.7*caps).min(-1).values, scores.max(-1).values
    for _ in range(80):
        mid = (left+right)/2
        total = torch.minimum(((scores-mid[:, None])/.7).clamp_min(0), caps).sum(-1)
        left, right = torch.where(total > target, mid, left), torch.where(total > target, right, mid)
    expected = torch.minimum(((scores-((left+right)/2)[:, None])/.7).clamp_min(0), caps)
    equivalent(result, expected)
    if dtype == torch.float64:
        assert torch.autograd.gradcheck(lambda *x: quadratic(*x, .7), (scores, caps, target))


def test_fc_formula_outputs_controls_terminal_state_and_numeric_gradients(dtype):
    from tidegraph.runtime import resolve_device
    from layer_tensor_baseline import numerics
    device, _ = resolve_device(os.environ.get("TIDE_FC_DEVICE", "cpu"))
    numerics(device)
    b, spec = fixture(dtype, device)
    oracle, ospec = fixture(dtype)
    original = torch.linspace(-.2, .4, 24, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    x = original.detach().to(device).requires_grad_()
    actual = run(b, spec, x)
    stop = 3*spec.stride
    expected = reference(oracle.graph, oracle.model, Continuation(oracle.graph.identity, 2),
                         layered_inputs(ospec, original), stop, sealed_until=stop)
    rows = []
    for sample in range(2):
        rows.append(torch.stack([readout(expected.outputs,
            [(p*spec.stride+spec.rank(n), port) for port, n in enumerate(b.graph.outputs)],
            original[sample, p], batch=sample) for p in range(3)]))
    pairs = [(actual['hidden'], torch.stack(rows))]
    for event in expected.trace:
        v, sample = event['node'], event['batch']
        position = (event['time']-spec.rank(v))//spec.stride
        record = actual['regions'][b.graph.nodes[v].region]
        assert bool(record['active'][sample, position, v % 4]) == event['active']
        pairs.append((record['controls'][sample, position, v % 4], event['control']))
    for v, raw in actual['states'].items():
        for sample in range(2):
            old = expected.continuation.states[sample, v]
            for name, value in raw.items():
                pairs.append((value if name == 'times' else value[sample], old.slots[name]))
    for a, e in pairs:
        equivalent(a.detach().cpu(), e)
    # Output/state numeric projection: do not claim None-versus-connected-zero equivalence.
    left = actual['hidden'].square().sum()+sum(s['keys'].square().sum()*.01 for s in actual['states'].values())
    right = pairs[0][1].square().sum()+sum(s.slots['keys'].square().sum()*.01 for s in expected.continuation.states.values())
    roots = [x, *(p for p in b.model.parameters() if p.requires_grad)]
    oroots = [original, *(p for p in oracle.model.parameters() if p.requires_grad)]
    ga = torch.autograd.grad(left, roots, allow_unused=True)
    ge = torch.autograd.grad(right, oroots, allow_unused=True)
    for a, e, root in zip(ga, ge, oroots):
        equivalent(torch.zeros_like(root) if a is None else a.detach().cpu(), torch.zeros_like(root) if e is None else e)


def test_fc_formula_rejects_unsupported_domain():
    b, spec = fixture(torch.float32)
    with pytest.raises(ValueError, match="L<=256"):
        run(b, spec, torch.ones(1, 257, 4))
