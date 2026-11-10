"""Allocation §3: general Describe columns, independent formulas and row roots.

Finite pairwise cases cover each head, key and gain; this is not a Cartesian
qualification. Scalar state queries stay independent of the column candidate.
"""
from dataclasses import replace
import math
import os
from types import SimpleNamespace

import pytest
import torch

from tidegraph.compare import equivalent
from tidegraph.runtime import resolve_device
from tidegraph.weighted import KVConfig, RegionalConfig
from tidegraph.weighted import dense_describe
from tidegraph.weighted.programs import Describe
from tidegraph.weighted.query_execution import execution_view
from tidegraph.weighted.query_vjp import evaluate
from test_weighted_describe_capability import linear_vjp
from test_weighted_query_vjp import native_query
from test_weighted_read_batch import fixture
from test_weighted_scoring import scalar_encoder


DEVICE, _ = resolve_device(os.environ.get("TIDE_DESCRIBE_DEVICE", "cpu"))
CASES = [
    dict(local_score="match", region_score="att", region_evidence="grouped", region_key="content"),
    dict(selection_mode="regional_threshold", region_score="att", region_key="content_stats"),
    dict(selection_mode="regional_total_zero", local_score="match", region_control_mode="fixed", total_fraction=.4),
    dict(selection_mode="regional_threshold_hard", region_score="att", region_control_mode="fixed", region_key="content"),
    dict(selection_mode="direct_attention", local_score="match", region_score="att", send_gain="context"),
    dict(selection_mode="proportional", fixed_gain=1.3),
    dict(selection_mode="constrained", local_score="match", region_score="att", region_key="content", gain_logit=-.2),
    dict(selection_mode="hard_selection", send_gain="context"),
]


def regional(case):
    return RegionalConfig(hidden_width=3, gamma_max=1.7, **CASES[case])


def independent_head(program, memory, mass, u, values):
    """Literal scalar sums; no candidate head, encoder or projection calls."""
    c, eps = program.config, memory.config.epsilon
    def normalize(x):
        return x/torch.sqrt(eps+sum(v*v for v in x)/x.numel())
    def mv(matrix, vector):
        return torch.stack([sum(row[j]*vector[j] for j in range(vector.numel())) for row in matrix])
    def silu(x):
        return x*x.sigmoid()
    def scalar(module, value):
        h = silu(mv(module.matrix, value)+module.hidden_bias)
        return module.bias+sum(h[j]*module.vector[j] for j in range(h.numel()))
    def encoders(module, groups):
        return tuple(scalar_encoder(module.encoders[name], groups[name]) for name in ("cur", "mem", "stat"))
    short, long, ws, wl = values
    cur = normalize(u)
    current = mv(memory.summary, cur)
    stat = torch.stack((mass.log1p(), ws.log1p(), wl.log1p()))
    groups = dict(cur=cur, mem=torch.cat((short, long)), stat=stat)
    f = torch.cat((cur, stat[:1], short, long, stat[1:], current*short, current*long))
    if c.local_score == "cat":
        score = scalar(program.score, f)
    else:
        module = program.score
        a, b, z = encoders(module, groups)
        joined = torch.cat((a, b, z))
        residual = silu(mv(module.gate, joined)+module.gate_bias)*(mv(module.up, joined)+module.up_bias)
        score = module.bias+c.alpha_match*sum(a*b)/math.sqrt(c.selection_width)+c.alpha_res*sum(residual*module.vector)
    received = mass.clamp(max=1)
    if c.regional:
        mu = (c.gamma_max*received).clamp(max=1)
    else:
        gamma = (1+(c.gamma_max-1)*scalar(program.gain, f).sigmoid() if c.send_gain == "context" else
                 1+(c.gamma_max-1)*program.gain_logit.sigmoid() if c.gain_logit is not None else c.fixed_gain)
        mu = (gamma*mass).clamp(max=1)
    result = dict(w=received, mu=mu, score=score)
    if c.regional:
        if c.region_control_mode == "fixed":
            result["evidence"] = f.new_zeros(c.region_control_width)
        elif c.region_evidence == "cat":
            result["evidence"] = normalize(mv(program.evidence_weight, f)+program.evidence_bias)
        else:
            module = program.evidence
            a, b, z = encoders(module, groups)
            result["evidence"] = normalize(mv(module.cur, a)+mv(module.mem, b)+mv(module.stat, z)
                                           +mv(module.interaction, a*b)+module.bias)
    if c.region_score == "att":
        result["x"] = mv(program.current_weight, cur)+program.current_bias
        result["key"] = (scalar_encoder(program.key, groups["mem"]) if c.region_key == "content" else
                         mv(program.key_weight, torch.cat((groups["mem"], stat[1:])))+program.key_bias)
    return result


@pytest.mark.parametrize("case", range(len(CASES)))
def test_columns_batch_time_axes_independent_formula_and_roots(dtype, case):
    c, r = KVConfig(width=2, summary_width=2, epsilon=.2), regional(case)
    g = torch.Generator().manual_seed(83+case)
    p = Describe("columns-test", c, r, g, dtype)
    def leaf(shape):
        return torch.randn(shape, generator=g, dtype=dtype).requires_grad_()
    mass = torch.tensor([[0., .2, 1.], [1.3, .4, .7]], dtype=dtype, requires_grad=True)
    u, short, long = leaf((2, 3, 2)), leaf((2, 3, 2)), leaf((2, 3, 2))
    ws, wl, summary = leaf((2, 3)).square(), leaf((2, 3)).square(), leaf((2, 2))
    memory = SimpleNamespace(config=c, summary=summary)
    actual = dense_describe.columns(p, memory, mass, u, (short, long, ws, wl))
    roots = dict(p.named_parameters()) | dict(mass=mass, u=u, short=short, long=long, ws=ws, wl=wl, summary=summary)
    assert {k: tuple(v.shape[2:]) for k, v in actual.items()} == p.field_layout
    for a, b in ((0, 0), (0, 2), (1, 0), (1, 2)):
        expected = independent_head(p, memory, mass[a, b], u[a, b], tuple(x[a, b] for x in (short, long, ws, wl)))
        for name, value in expected.items():
            result = actual[name][a, b]
            equivalent(result, value)
            assert result.requires_grad == value.requires_grad
            if value.requires_grad:
                equivalent(linear_vjp(result, roots), linear_vjp(value, roots))


def prepared(case, dtype):
    family, read = (("kv", "sum"), ("ssd", "state"), ("delta", "state"), ("gated_delta", "sum"))[case % 4]
    w, requests = fixture(dtype, family, read, False)
    w.read_program = Describe("columns-test", w.kernel.config, regional(case), torch.Generator().manual_seed(29+case), dtype)
    if case == 0:
        # Same parameter owners, different functions and all usage contributions.
        w.read_program.evidence.encoders["mem"] = w.read_program.score.encoders["mem"]
        w.read_program.key = w.read_program.score.encoders["mem"]
    w.read_program.score.bias.requires_grad_(False)
    w.to(DEVICE)
    roots = {n: p for n, p in w.named_parameters() if p.requires_grad}
    rows = []
    for i, r in enumerate([requests[0], requests[1], requests[2], requests[-1]]):
        def leaf(name, tensor):
            t = tensor.detach().to(DEVICE).clone().requires_grad_(tensor.is_floating_point())
            if t.requires_grad:
                roots[f"row.{i}.{name}"] = t
            return t
        state = r.state
        if state is not None:
            state = replace(state, value=leaf("value", state.value), slots={k: leaf(k, t) for k, t in state.slots.items()})
        rows.append(replace(r, state=state, content=replace(r.content, value=leaf("content", r.content.value))))
    model = torch.nn.Module(); model.nodes = torch.nn.ModuleList([w])
    selected = execution_view(model, "auto").nodes[0]
    return w, selected, rows, roots


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_general_heads_independent_snapshot_fields_and_partial_vjp(dtype, case, native, monkeypatch):
    scalar, selected, rows, roots = prepared(case, dtype)
    expected = [scalar.read_program.step(scalar, r) for r in rows]
    def forbidden(*a, **kw):
        raise AssertionError("batched Describe invoked scalar head/Describe")
    monkeypatch.setattr(selected.read_program, "describe", forbidden)
    monkeypatch.setattr(selected.read_program, "step", forbidden)
    monkeypatch.setattr(selected.read_program.score, "forward", forbidden)
    actual = (native_query if native else evaluate)(selected.read_program, selected, rows)
    equivalent(actual, expected)
    for a, b in zip(actual, expected):
        for name in b:
            assert a[name].requires_grad == b[name].requires_grad
            if b[name].requires_grad:
                equivalent(linear_vjp(a[name], roots), linear_vjp(b[name], roots))
    def partial(values):
        return values[1]["score"]*3+0*values[0]["mu"]+values[2]["w"]+values[-1]["score"]
    equivalent(linear_vjp(partial(actual), roots), linear_vjp(partial(expected), roots))
    equivalent(linear_vjp(actual[2]["score"]*0, roots), linear_vjp(expected[2]["score"]*0, roots))


def test_general_heads_submit_shared_projections_once_per_batch(dtype, monkeypatch):
    from tidegraph.weighted.read_batch import evaluate as numeric
    from torch.utils._python_dispatch import TorchDispatchMode
    _, selected, rows, _ = prepared(0, dtype)
    calls = []
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            # CPU decomposes matmul to mm before this dispatch boundary;
            # TorchNPU 2.12 exposes the original two-dimensional matmul.
            # Observe the actual matrix operands in either case, not wrappers.
            if func in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default):
                assert args[0].ndim == args[1].ndim == 2
                calls.append(tuple(args[1].shape))
            return func(*args, **(kwargs or {}))
    def forbidden(*a, **kw):
        raise AssertionError("batch called scalar Describe or memory function")
    monkeypatch.setattr(selected.read_program, "describe", forbidden)
    monkeypatch.setattr(selected.kernel, "current", forbidden)
    monkeypatch.setattr(selected.kernel, "mem", forbidden)
    with torch.no_grad(), Observe():
        result = numeric(selected.read_program, selected, rows)
    # current 1 + match 8 + grouped 10 + x 1 + content key 2.
    assert len(calls) == 22
    assert all(shape[-1] == len(rows) for shape in calls)
    assert all(set(row) == set(selected.read_program.field_layout) for row in result)


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_zero_scales_keep_declared_fields_and_connected_roots(dtype, native):
    scalar, selected, rows, roots = prepared(0, dtype)
    config = replace(scalar.read_program.config, alpha_match=0., alpha_res=0., alpha_loc=0., alpha_sel=0.)
    scalar.read_program.config = selected.read_program.config = config
    scalar.read_program.score.config = config
    expected = [scalar.read_program.step(scalar, r) for r in rows]
    actual = (native_query if native else evaluate)(selected.read_program, selected, rows)
    equivalent(actual, expected)
    root = scalar.read_program.score.encoders["mem"].in_weight
    score_gradient = torch.autograd.grad(actual[1]["score"], root, retain_graph=True)[0]
    assert torch.equal(score_gradient, torch.zeros_like(root))
    assert torch.autograd.grad(actual[1]["key"].sum(), root, retain_graph=True)[0].count_nonzero() > 0
    assert set(actual[1]) == {"w", "mu", "score", "evidence", "x", "key"}
    equivalent(linear_vjp(actual[1]["score"], roots), linear_vjp(expected[1]["score"], roots))


@pytest.mark.parametrize("kind", ["content", "content_stats"])
def test_columns_key_direct_dependencies_and_empty_memory_bias(dtype, kind):
    c = KVConfig(width=2, summary_width=2, epsilon=.2)
    p = Describe("key-test", c, RegionalConfig(region_score="att", region_key=kind),
                 torch.Generator().manual_seed(43), dtype)
    mass = torch.tensor([.2, .6], dtype=dtype, requires_grad=True)
    u = torch.tensor([[.3, -.2], [.1, .4]], dtype=dtype, requires_grad=True)
    short = torch.zeros((2, 2), dtype=dtype, requires_grad=True)
    long = torch.zeros((2, 2), dtype=dtype, requires_grad=True)
    ws = torch.tensor([.5, .9], dtype=dtype, requires_grad=True)
    wl = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
    memory = SimpleNamespace(config=c, summary=torch.ones((2, 2), dtype=dtype))
    key = dense_describe.columns(p, memory, mass, u, (short, long, ws, wl))["key"]
    assert key.count_nonzero() > 0
    grads = torch.autograd.grad(key.sum(), (mass, u, short, long, ws, wl), allow_unused=True)
    assert grads[1] is None
    if kind == "content":
        assert grads[0] is grads[4] is grads[5] is None
    else:
        # stat is a single stacked tensor: its unused received coordinate is
        # a connected zero, matching the scalar content_stats formula.
        assert torch.equal(grads[0], torch.zeros_like(mass))
        assert grads[4].count_nonzero() > 0 and grads[5].count_nonzero() > 0
