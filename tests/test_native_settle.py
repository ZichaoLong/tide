"""Independent C++ construction is the subject; Python encoding is an oracle."""
from pathlib import Path
import os
import subprocess
import pytest
import torch
import _tide_native as core
from tidegraph import Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.native import Native
from tidegraph.native_records import from_continuation, to_continuation, window_records
from tidegraph.ops import Model
from tidegraph.records import Result
from tidegraph.settle import SettleGraph, run
from cases import gradients
from isolated_cases import vjp
from test_settle import fixture


def native(spec, model, q, *, mode="hst", packed=True, workers=3):
    body = Native(spec.graph, model)
    compiled = core.SettleGraph(body.compiled, spec.ranks)
    initial = compiled.embed_initial(to_continuation(core, spec.graph, body.compiled, q))
    options = core.Options()
    options.mode, options.packed, options.workers = mode, packed, workers
    return compiled, core.SettleExecutor(compiled, body.weights, options), initial


def projected(spec, compiled, result):
    value = compiled.project(result)
    assert value.continuation.identity == compiled.graph.identity
    return Result(from_continuation(spec.graph, value.continuation), *window_records(value))


@pytest.mark.parametrize("clear", [False, True])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
@pytest.mark.parametrize("packed,workers", [(False, 1), (False, 3), (True, 3)])
def test_native_construction_full_trace_initial_state_and_vjp(dtype, clear, mode, packed, workers):
    spec, model, q, x, initial = fixture(dtype, clear)
    expected = run(spec, model, q, x, mode=mode)
    expected_grad = gradients(objective(expected), model, x, initial)
    spec, model, q, x, initial = fixture(dtype, clear)
    compiled, engine, eq = native(spec, model, q, mode=mode, packed=packed, workers=workers)
    encoded = engine.run(eq, x)
    actual = projected(spec, compiled, encoded)
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), model, x, initial))
    assert encoded.stats["max_state_sequence"] == 5
    # A C++ constructed encoding must equal the separately built Python encoding.
    eg, em = spec.embed(model)
    oracle = Native(eg, em).compiled
    assert compiled.encoded_graph.identity == oracle.identity


def test_native_encoding_ports_parallel_edges_source_domains_and_aliases(dtype):
    from test_port_layout import graph
    def setup():
        g = graph(); spec = SettleGraph(g, (1, 2, 3)); m = Model(g, dtype=dtype)
        m.nodes[1] = m.nodes[0]
        m.output_scale[0] = m.input_scale[0]
        x = (torch.arange(18, dtype=dtype).reshape(2, 3, 3) / 30).requires_grad_()
        return spec, m, Continuation(g.identity, 2), x
    spec, m, q, x = setup()
    expected = run(spec, m, q, x, mode="hst")
    expected_grad = vjp(objective(expected), dict(m.named_parameters()) | {"input": x})
    spec, m, q, x = setup()
    body = Native(spec.graph, m)
    compiled, engine, eq = native(spec, m, q)
    em = compiled.embed_model(body.weights)
    assert em.nodes[0].weight is em.nodes[1].weight is m.nodes[0].weight
    e, p = len(spec.graph.edges), len(spec.graph.inputs)
    assert em.agg_scale[e] is em.agg_scale[e+p] is m.input_scale[0]
    assert len(em.parameters().owners()) == len(body.weights.parameters().owners())
    eg, reference_model = spec.embed(m)
    assert compiled.encoded_graph.identity == Native(eg, reference_model).compiled.identity
    actual = projected(spec, compiled, engine.run(eq, x))
    equivalent(expected, actual)
    equivalent(expected_grad, vjp(objective(actual), dict(m.named_parameters()) | {"input": x}))
    mixed = next(e for e in actual.trace if e["node"] == 2)
    assert {a.kind for a in mixed["fiber"]} == {0, 1}
    assert {a.source for a in mixed["fiber"] if a.kind == 1} == {0, 1, 2}


@pytest.mark.parametrize("kind", ["ema", "ssm", "attention", "linear", "delta", "delta-rule-v1"])
def test_native_settle_isolated_roots_and_all_slots(dtype, kind):
    from test_isolated_schedules import fixture as memory_fixture
    g, m, q, x, variables = memory_fixture(dtype, kind)
    spec = SettleGraph(g, (1, 2))
    expected = run(spec, m, q, x, mode="hst")
    g, m, q, x, actual_variables = memory_fixture(dtype, kind)
    compiled, engine, eq = native(SettleGraph(g, (1, 2)), m, q)
    actual = projected(spec, compiled, engine.run(eq, x))
    equivalent(expected, actual)
    roots = lambda r: [r.outputs[0][3], r.continuation.states[0, 0].value,
                       *r.continuation.states[0, 0].slots.values()]
    for expected_root, actual_root in zip(roots(expected), roots(actual)):
        for zero in (False, True):
            equivalent(vjp(expected_root, variables, zero), vjp(actual_root, actual_variables, zero))


@pytest.mark.parametrize("clear", [False, True])
def test_native_settle_complete_cuts_empty_chunk_and_streaming_projection(dtype, clear):
    spec, m, q, x, initial = fixture(dtype, clear)
    expected = run(spec, m, q, x, mode="hst")
    expected_grad = gradients(objective(expected), m, x, initial)
    spec, m, q, x, initial = fixture(dtype, clear)
    compiled, engine, eq = native(spec, m, q)
    events, outputs, messages, begin = [], [], [], 0
    for end in (1, 3, 3, 5):
        result = engine.run(eq, x[:, begin:end]); eq = result.continuation
        part = projected(spec, compiled, result)
        events += part.trace; outputs += part.outputs; messages += part.messages
        begin = end
    actual = Result(part.continuation, events, outputs, messages, {})
    equivalent(expected, actual)
    equivalent(expected_grad, gradients(objective(actual), m, x, initial))
    spec, m, q, x, _ = fixture(dtype, clear)
    body = Native(spec.graph, m); compiled, _, eq = native(spec, m, q)
    options = core.Options(); options.mode = "hst"
    streaming = core.Streaming(compiled.encoded_graph, compiled.embed_model(body.weights), options)
    result = streaming.run(eq, compiled.external(x), 5 * spec.stride, 5 * spec.stride)
    equivalent(expected, projected(spec, compiled, result))


def test_native_settle_rejects_malformed_contracts_without_mutating_inputs(dtype):
    spec, m, q, x, _ = fixture(dtype, False)
    body = Native(spec.graph, m); compiled, engine, eq = native(spec, m, q)
    for ranks in ((1, 1, 4), (0, 2, 4), (1, 3, 4), (1, 2), (1, 2, 2**63-1)):
        with pytest.raises(ValueError, match="rank"):
            core.SettleGraph(body.compiled, ranks)
    with pytest.raises(ValueError, match="clock overflow"):
        compiled.external(x, 2**63-1)
    with pytest.raises(ValueError, match="batch"):
        engine.run(eq, x[:1])
    with pytest.raises(ValueError, match="width"):
        engine.run(eq, x[0])
    result = engine.run(eq, x[:, :1])
    with pytest.raises(ValueError, match="initial cut"):
        compiled.embed_initial(result.continuation)
    bad = result.continuation; bad.identity = "wrong"; result.continuation = bad
    with pytest.raises(ValueError, match="identity"):
        compiled.project(result)
    bad.identity = compiled.encoded_graph.identity; bad.cut = 1; result.continuation = bad
    with pytest.raises(ValueError, match="position cut"):
        compiled.project(result)
    options = core.Options()
    streaming = core.Streaming(compiled.encoded_graph, compiled.embed_model(body.weights), options)
    partial = streaming.run(eq, compiled.external(x[:, :1]), 1, 1)
    assert partial.continuation.pending
    with pytest.raises(ValueError, match="position cut"):
        compiled.project(partial)
    assert eq.cut == 0 and not eq.ledger and not eq.pending
    equivalent(run(spec, m, q, x, mode="hst"), projected(spec, compiled, engine.run(eq, x)))


def test_standalone_settle_entry(dtype, tmp_path):
    build = Path(os.environ.get("TIDE_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"))
    result = subprocess.run([str(build / "tidegraph-settle-check"), "--device=cpu",
                             f"--dtype={str(dtype).split('.')[-1]}", f"--output-dir={tmp_path / 'report'}"],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "standalone-settle: passed" in result.stdout
