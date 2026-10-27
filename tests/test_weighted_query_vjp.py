"""Architecture §3/4: independent structured roots and exact query VJP branches.

Local scalar graphs anchor connectivity; independent formula tests anchor the
functions. Empty/expired/zero-weight caches must not inherit another row's roots.
"""
from dataclasses import asdict, replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.full import FullInput
from tidegraph.weighted.query_vjp import evaluate
from isolated_cases import vjp
from test_weighted_read_batch import CASES, fixture


def native_query(program, w, requests, *, kind="read", slots=0):
    import _tide_native as core
    from tidegraph.full import FullResult
    from tidegraph.weighted.config import KVConfig
    from tidegraph.weighted.native import region_config
    weight = core.NodeWeights(w.decay, w.weight, w.bias, w.read)
    weight.extra = dict(w.named_parameters(remove_duplicate=False))
    is_kv = isinstance(w.kernel.config, KVConfig)
    config = core.WeightedKVConfig() if is_kv else core.WeightedRecurrentConfig()
    for name, value in asdict(w.kernel.config).items():
        setattr(config, name, value)
    config.validate()
    configure = core.configure_weighted_node if is_kv else core.configure_weighted_recurrent_node
    classes = w.aggregate_program.classes
    configure(weight, config, region_config(core, w.read_program.config), classes.edges, classes.inputs, w.aggregate_program.alpha)
    def state(s):
        return None if s is None else core.State(s.value, s.last_time, s.observations, s.slots)
    contents, times = [r.content.value for r in requests], [r.time for r in requests]
    if kind == "read":
        return core.weighted_read_probe(weight, [state(r.state) for r in requests], times, contents)
    result = core.weighted_full_probe(weight, [state(r.comparison) for r in requests], times, contents,
                                      [r.control for r in requests], slots)
    return [FullResult(value, emitted) for value, emitted in result]


def independent(dtype, family, read):
    w, requests = fixture(dtype, family, read, True)
    roots = {name: t for name, t in w.named_parameters() if t.requires_grad}
    rows = []
    for i, r in enumerate(requests):
        state = r.state
        if state is not None:
            def leaf(name, t):
                t = t.detach().requires_grad_(t.is_floating_point())
                if t.requires_grad:
                    roots[f"state.{i}.{name}"] = t
                return t
            state = replace(state, value=leaf("value", state.value), slots={k: leaf(k, t) for k, t in state.slots.items()})
            if family == "kv" and i == 3:
                with torch.no_grad():
                    state.slots["weights"].zero_()
        value = r.content.value.detach().requires_grad_()
        roots[f"input.{i}"] = value
        rows.append(replace(r, state=state, content=replace(r.content, value=value)))
    return w, rows, roots


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("kind", ["read", "full"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_independent_fields_states_inputs_and_zero_cotangents(dtype, family, read, kind, native, monkeypatch):
    w, requests, roots = independent(dtype, family, read)
    if kind == "full":
        rows = []
        for i, r in enumerate(requests):
            control = r.content.value.new_tensor(.2+.01*i, requires_grad=True)
            roots[f"control.{i}"] = control
            rows.append(FullInput(w.initial() if r.state is None else r.state, r.time, r.content, control))
        requests = rows
        p = w.full_program
        expected = [p.step(w, r, 3, "hard", 1.) for r in requests]
    else:
        p = w.read_program
        expected = [p.step(w, r) for r in requests]
    def forbidden(*args, **kw):
        raise AssertionError("query batch VJP called scalar local function")
    monkeypatch.setattr(p, "step", forbidden)
    monkeypatch.setattr(w.kernel, "mem", forbidden)
    query = native_query if native else evaluate
    actual = query(p, w, requests, kind=kind, slots=3)
    equivalent(expected, actual)
    def flatten(values):
        return [[r.value] if kind == "full" else [r[k] for k in sorted(r)] for r in values]
    a, b = flatten(expected), flatten(actual)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            assert x.requires_grad == y.requires_grad
            if x.requires_grad:
                equivalent(vjp(x, roots), vjp(y, roots))
    # Different fields active in different rows, and a numerically zero root.
    def mixed(rows):
        return sum(row[i % len(row)].square().sum() for i, row in enumerate(rows))+0*rows[0][-1].sum()
    equivalent(vjp(mixed(a), roots), vjp(mixed(b), roots))
    equivalent(vjp(b[0][-1], roots, zero=True), vjp(a[0][-1], roots, zero=True))
    if kind == "full":
        equivalent(vjp(actual[2].emitted[1], roots), vjp(expected[2].emitted[1], roots))
    assert query(p, w, [], kind=kind, slots=3) == []


@pytest.mark.parametrize("kind", ["read", "full"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_query_parameter_versions_and_first_order(kind, native):
    w, requests, roots = independent(torch.float64, "gated_delta", "state")
    if kind == "full":
        requests = [FullInput(w.initial() if r.state is None else r.state, r.time, r.content, r.content.value.new_tensor(.3)) for r in requests]
    p = w.read_program if kind == "read" else w.full_program
    query = native_query if native else evaluate
    def root():
        row = query(p, w, requests, kind=kind, slots=2)[2]
        return row["score"] if kind == "read" else row.value
    with pytest.raises((ValueError, RuntimeError), match="first-order"):
        torch.autograd.grad(root().sum(), tuple(roots.values()), allow_unused=True, create_graph=True)
    y = root()
    with torch.no_grad():
        w.kernel.conv.weight.add_(.01)
    with pytest.raises(RuntimeError, match="modified by an inplace"):
        y.sum().backward()


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("kind", ["read", "full"])
def test_query_parameter_aliases_and_repeated_inputs(dtype, native, kind):
    w, requests, original = independent(dtype, "kv", "qry")
    read = w.read_program
    read.key = read.evidence.encoders["mem"] = read.score.encoders["mem"]
    w.full_program.content.query = w.kernel.read_q
    roots = {name: p for name, p in w.named_parameters() if p.requires_grad}
    roots.update({k: t for k, t in original.items() if k.startswith(("state.", "input."))})
    # Distinct private snapshots can use one public content root. It must get
    # every contribution; unrelated rows remain disconnected.
    requests = [requests[2], replace(requests[3], content=requests[2].content), requests[2]]
    query = native_query if native else evaluate
    if kind == "full":
        control = requests[0].content.value.new_tensor(.3, requires_grad=True)
        roots["control"] = control
        requests = [FullInput(r.state, r.time, r.content, control) for r in requests]
        p = w.full_program
        expected = [p.step(w, r, 2, "hard", 1.).value for r in requests]
        actual = [r.value for r in query(p, w, requests, kind=kind, slots=2)]
    else:
        p = w.read_program
        def root(r):
            return r["score"]+r["evidence"].sum()+r["key"].sum()
        expected = [root(p.step(w, r)) for r in requests]
        actual = [root(r) for r in query(p, w, requests)]
    equivalent(vjp(sum(expected), roots), vjp(sum(actual), roots))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("kind", ["read", "full"])
@pytest.mark.parametrize("family,read", [("kv", "qry"), ("gated_delta", "state"), ("ssd", "state")])
def test_query_directional_difference_of_parameters_and_snapshot(native, kind, family, read):
    w, requests, roots = independent(torch.float64, family, read)
    r = requests[2]
    p = w.read_program
    if kind == "full":
        control = r.content.value.new_tensor(.37, requires_grad=True)
        roots["control"] = control
        r = FullInput(r.state, r.time, r.content, control); p = w.full_program
    def scalar():
        return p.step(w, r) if kind == "read" else p.step(w, r, 2, "hard", 1.)
    def objective(row):
        if kind == "full":
            return (row.value*row.value.new_tensor([.3, -.4, .7])).sum()
        return row["score"]+.3*row["evidence"].sum()-.4*row["key"].sum()+.2*row["x"].sum()
    query = native_query if native else evaluate
    value = objective(query(p, w, [r], kind=kind, slots=2)[0])
    tensors = tuple(roots.values())
    grads = torch.autograd.grad(value, tensors, allow_unused=True)
    directions = [torch.linspace(-.13, .21, t.numel(), dtype=t.dtype).reshape_as(t) for t in tensors]
    derivative = sum((g*d).sum() for g, d in zip(grads, directions) if g is not None)
    originals = [t.detach().clone() for t in tensors]
    def move(amount):
        with torch.no_grad():
            for t, v, d in zip(tensors, originals, directions):
                t.copy_(v+amount*d)
    try:
        move(1e-5); plus = objective(scalar()).detach()
        move(-1e-5); minus = objective(scalar()).detach()
    finally:
        move(0.)
    torch.testing.assert_close((plus-minus)/2e-5, derivative, atol=2e-8, rtol=2e-6)
