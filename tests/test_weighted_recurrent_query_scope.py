"""Recurrent §2/3: consumed query fields, complete guards and private roots.

Scalar functions remain independent. Int64/empty/no-match fixtures also anchor
None versus connected-zero gradients when unused numerical branches are removed.
"""
from collections import Counter
from dataclasses import replace
import pytest
import torch
from tidegraph.compare import equivalent
from tidegraph.full import FullInput
from tidegraph.weighted.query_vjp import evaluate
from test_weighted_describe_capability import prepared, linear_vjp
from test_weighted_query_vjp import native_query

CASES = [(f, read) for f in ("affine", "delta", "gated_delta", "ssd") for read in ("sum", "state")]


def material(dtype, family, read):
    _, weights, _, rows, roots = prepared(dtype, family, read)
    # Same parameter owns Read and Full; each use must contribute once.
    if read == "state":
        weights.full_program.content.out = weights.kernel.read_out
    roots = {**{n: p for n, p in weights.named_parameters() if p.requires_grad},
             **{n: p for n, p in roots.items() if n.startswith("row.")}}
    full = []
    for i, row in enumerate(rows):
        control = row.content.value.new_tensor(.2+.01*i, requires_grad=True)
        roots[f"control.{i}"] = control
        full.append(FullInput(row.state or weights.initial(), row.time, row.content, control))
    return weights, rows, full, roots


def execute(weights, rows, full, native):
    query = native_query if native else evaluate
    return (query(weights.read_program, weights, rows),
            query(weights.full_program, weights, full, kind="full", slots=2))


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_recurrent_read_full_complete_and_partial_roots(dtype, family, read, native):
    w, rows, full, roots = material(dtype, family, read)
    expected = ([w.read_program.step(w, row) for row in rows],
                [w.full_program.step(w, row, 2, "hard", 1.) for row in full])
    actual = execute(w, rows, full, native)
    equivalent(actual, expected)
    for i in (0, 1, 2, 4, len(rows)-1):
        for field in expected[0][i]:
            equivalent(linear_vjp(actual[0][i][field], roots), linear_vjp(expected[0][i][field], roots))
        equivalent(linear_vjp(actual[1][i].value, roots), linear_vjp(expected[1][i].value, roots))
    def partial(result):
        desc, outputs = result
        return 2*desc[1]["score"]+desc[2]["evidence"][0]+outputs[4].emitted[1][-1]+outputs[1].value[1]+0*desc[0]["mu"]
    equivalent(linear_vjp(partial(actual), roots), linear_vjp(partial(expected), roots))
    for side, field in ((0, "evidence"), (1, "value")):
        left = actual[side][2][field] if side == 0 else actual[side][2].value
        right = expected[side][2][field] if side == 0 else expected[side][2].value
        equivalent(linear_vjp(left*0, roots), linear_vjp(right*0, roots))
    # Complete independent states include unused fields; dropping computations
    # must leave them disconnected instead of manufacturing connected zeros.
    grad = linear_vjp(actual[0][1]["score"], roots)
    assert grad["row.1.value"] is None
    if read == "sum":
        assert grad["row.1.H"] is None
    elif family in {"affine", "delta"}:
        assert grad["row.1.Z"] is None
    if family == "ssd":
        assert grad["kernel.delta_bias"] is None
    grad = linear_vjp(actual[1][1].value, roots)
    assert grad["row.1.Z"] is grad["row.1.W"] is None
    if family == "ssd":
        assert grad["kernel.delta_bias"] is None
    if family in {"gated_delta", "ssd"}:
        empty = linear_vjp(actual[1][0].value, roots)["row.0.conv_values"]
        missing = linear_vjp(actual[1][4].value, roots)["row.4.conv_values"]
        assert empty is None
        assert missing is not None and torch.equal(missing, torch.zeros_like(missing))
        assert rows[4].time > 2**54 and rows[4].state.slots["conv_times"].dtype == torch.int64


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("kind", ["read", "full"])
def test_only_consumed_state_fields_are_materialized_once(family, read, kind):
    from tidegraph.weighted.read_batch import evaluate as read_numeric
    from tidegraph.weighted.full_batch import evaluate as full_numeric
    w, rows, full, _ = material(torch.float64, family, read)
    hits = Counter()
    class Slots(dict):
        def __getitem__(self, key):
            hits[key] += 1
            return super().__getitem__(key)
    # Use concrete nonempty snapshots, so every access is observable.
    rows = [replace(r, state=replace(r.state, slots=Slots(r.state.slots))) for r in rows[1:4]]
    full = [replace(r, comparison=rows[i].state) for i, r in enumerate(full[1:4])]
    with torch.no_grad():
        if kind == "read":
            read_numeric(w.read_program, w, rows)
        else:
            full_numeric(w.full_program, w, full, 2, "hard", 1.)
    wanted = {"H"} if kind == "full" else {"Z", "W"} if read == "sum" else {
        "H", "W"} if family in {"affine", "delta"} else {"H", "Z", "W"}
    assert {k: hits[k] for k in ("H", "Z", "W")} == {k: 3 if k in wanted else 0 for k in ("H", "Z", "W")}
    assert hits["time"] == 3


@pytest.mark.parametrize("family", ["gated_delta", "ssd"])
@pytest.mark.parametrize("full", [False, True], ids=["read", "full"])
def test_query_features_keep_projection_but_skip_unused_work(family, full, monkeypatch):
    from tidegraph.weighted import query_batch
    w, rows, _, _ = material(torch.float64, family, "state")
    rows = rows[:3]
    projections, norms, expansions = [], [], []
    project, norm, expand = query_batch.project, query_batch.epsilon_l2, torch.Tensor.repeat_interleave
    def projection(x, weight):
        projections.append(tuple(weight.shape))
        return project(x, weight)
    def normalize(x, epsilon):
        norms.append(tuple(x.shape))
        return norm(x, epsilon)
    def expansion(self, *args, **kwargs):
        expansions.append(tuple(self.shape))
        return expand(self, *args, **kwargs)
    def unused(*args, **kwargs):
        raise AssertionError("query computed unused discretization")
    monkeypatch.setattr(query_batch, "project", projection)
    monkeypatch.setattr(query_batch, "epsilon_l2", normalize)
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", expansion)
    monkeypatch.setattr(torch.nn.functional, "softplus", unused)
    with torch.no_grad():
        query, value, gate = query_batch.features(w.kernel, [r.state for r in rows], [r.time for r in rows],
                                                torch.stack([r.content.value[1:] for r in rows]), full=full)
    assert projections == [tuple(w.kernel.projection.shape)]
    assert query.shape == (3, w.kernel.config.heads, w.kernel.config.key_width)
    assert len(norms) == (family == "gated_delta")
    assert len(expansions) == (family == "ssd")
    assert (value is not None) == (gate is not None) == (family == "ssd" and full)


@pytest.mark.parametrize("kind", ["read", "full"])
@pytest.mark.parametrize("corrupt", ["projection-short", "projection-long", "bias-width", "bias-rank", "bias-missing"])
def test_ssd_query_metadata_keeps_complete_projection_and_bias_contract(kind, corrupt):
    w, rows, full, _ = material(torch.float64, "ssd", "state")
    c = w.kernel.config
    if corrupt == "projection-short":
        w.kernel.projection = torch.nn.Parameter(w.kernel.projection[:-1].detach().clone())
    elif corrupt == "projection-long":
        w.kernel.projection = torch.nn.Parameter(torch.cat((w.kernel.projection.detach(),
                                                           w.kernel.projection.new_zeros(1, c.width))))
    elif corrupt == "bias-width":
        w.kernel.delta_bias = torch.nn.Parameter(w.kernel.delta_bias.new_zeros(c.heads+1))
    elif corrupt == "bias-rank":
        w.kernel.delta_bias = torch.nn.Parameter(w.kernel.delta_bias.detach().reshape(1, c.heads))
    else:
        del w.kernel.delta_bias
    program = w.read_program if kind == "read" else w.full_program
    requests = rows if kind == "read" else full
    match = "projection shape" if corrupt.startswith("projection") else "delta_bias shape"
    with pytest.raises(ValueError, match=match):
        evaluate(program, w, [requests[1], requests[1]], kind=kind, slots=2)


@pytest.mark.parametrize("family,read", CASES)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_negative_recurrent_query_time_rejected_before_decay(family, read, native):
    w, rows, full, _ = material(torch.float64, family, read)
    query = native_query if native else evaluate
    for kind, program, requests in (("read", w.read_program, rows), ("full", w.full_program, full)):
        requests = [replace(requests[1], time=-1), replace(requests[2], time=-1)]
        with pytest.raises((ValueError, RuntimeError), match="time"):
            query(program, w, requests, kind=kind, slots=2)


@pytest.mark.parametrize("family", ["gated_delta", "ssd"])
@pytest.mark.parametrize("corrupt", ["projection-key", "cache-key", "cache-time"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_unused_channels_and_unmatched_cache_still_validate(family, corrupt, native):
    w, rows, full, _ = material(torch.float64, family, "state")
    c = w.kernel.config
    key = c.key_width if family == "gated_delta" else c.heads*c.value_width
    with torch.no_grad():
        if corrupt == "projection-key":
            offset = 0 if family == "gated_delta" else c.heads*c.value_width
            w.kernel.projection[offset+key].fill_(float("nan"))
        elif corrupt == "cache-key":
            rows[4].state.slots["conv_values"][0, key] = float("nan")
        else:
            rows[4].state.slots["conv_times"][0] = -1
    query = native_query if native else evaluate
    for kind, program, requests in (("read", w.read_program, rows), ("full", w.full_program, full)):
        with pytest.raises((ValueError, RuntimeError), match="convolution"):
            query(program, w, [requests[4], requests[4]], kind=kind, slots=2)


@pytest.mark.parametrize("kind", ["read", "full"])
@pytest.mark.parametrize("changed", ["unused-parameter", "unused-state"])
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_pruned_query_branches_preserve_version_guards(kind, changed, native):
    w, rows, full, _ = material(torch.float64, "ssd", "state")
    program = w.read_program if kind == "read" else w.full_program
    requests = rows if kind == "read" else full
    query = native_query if native else evaluate
    actual = query(program, w, [requests[1], requests[1]], kind=kind, slots=2)
    target = (w.kernel.delta_bias if changed == "unused-parameter" else
              rows[1].state.slots["Z"] if kind == "full" else rows[1].state.value)
    with torch.no_grad():
        target.add_(0)
    root = actual[0]["score"] if kind == "read" else actual[0].value.sum()
    with pytest.raises(RuntimeError, match="modified|version"):
        root.backward()
