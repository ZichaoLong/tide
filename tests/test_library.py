"""Consumer contracts: input ownership, independent oracle and resumable sessions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph import GraphConfig, GraphRuntime, ExecutionOptions, External, Continuation
from tidegraph.compare import equivalent, objective
from tidegraph.records import Result
from tidegraph.blocks import canonicalize


def config(family, dtype, implementation="python"):
    return GraphConfig.from_dict(dict(schema_version=1, family=family,
        topology=dict(kind="layered", layers=[2, 3, 2], budget=1,
            module=dict(full="swiglu"), node_overrides={"0":dict(memory="linear"),
                "2":dict(memory="ssm"), "3":dict(memory="delta")}),
        model=dict(width=4, dtype=str(dtype).split(".")[-1]),
        execution=dict(implementation=implementation, trace=True, mode="hst")))


def advance(session, values, start, stop):
    if session.runtime.spec:
        return session.advance(values[:, start:stop])
    records = [External(b, p, t, t, values[b, t]) for b in range(len(values))
               for p in range(len(session.runtime.graph.inputs)) for t in range(start, min(stop, values.shape[1]))]
    return session.advance(records, stop=stop, sealed_until=stop)


def gradients(result, runtime, values):
    roots = dict(runtime.model.named_parameters()) | {"input":values}
    return dict(zip(roots, torch.autograd.grad(objective(result), tuple(roots.values()), allow_unused=True)))


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_full_observables_vjp_and_chunking(dtype, family, implementation):
    cfg = config(family, dtype, implementation)
    assert GraphConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).identity == cfg.identity
    reference = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, trace=True, mode="hst"))
    runtime = GraphRuntime(cfg, device="cpu")
    x = torch.randn(2, 4, 4, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    stop = 4 if family == "settle" else 8
    expected = advance(reference.session(2), x, 0, stop)
    session = runtime.session(2)
    first = advance(session, y, 0, 2)
    second = advance(session, y, 2, stop)
    actual = canonicalize(runtime.graph, Result(second.continuation, first.trace + second.trace,
                         first.outputs + second.outputs, first.messages + second.messages, {}))
    equivalent(expected, actual)
    equivalent(gradients(expected, reference, x), gradients(actual, runtime, y))
    if family == "settle":
        from tidegraph.settle import run
        direct = run(runtime.spec, reference.model, Continuation(runtime.graph.identity, 2), x,
                     packed=False, prefill=False, mode="hst")
        equivalent(expected, direct)


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
def test_checkpoint_policy_change_and_weight_only(dtype, family, tmp_path):
    cfg = config(family, dtype)
    runtime = GraphRuntime(cfg, device="cpu")
    session = runtime.session(2)
    opt = torch.optim.AdamW(runtime.model.parameters(), lr=.001)
    x = torch.randn(2, 4, 4, dtype=dtype)
    first = advance(session, x, 0, 2)
    # The consumer owns this task and its differentiable objective.
    objective(first).backward()
    session.detach()
    opt.step()
    file = tmp_path / "state.pt"
    session.save(file, opt)
    with pytest.raises(FileExistsError):
        session.save(file, opt)
    other = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(implementation="native", schedule="streaming", trace=True, mode="hst"))
    restored = other.session(2)
    other_opt = torch.optim.AdamW(other.model.parameters(), lr=.001)
    restored.load(file, other_opt)
    equivalent(opt.state_dict(), other_opt.state_dict())
    equivalent(session.continuation, restored.continuation)
    stop = 4 if family == "settle" else 8
    expected = advance(session, x, 2, stop)
    actual = advance(restored, x, 2, stop)
    equivalent(expected, actual)
    fresh = GraphRuntime(replace(cfg, seed=99), device="cpu")
    fresh.load_weights(file)
    equivalent(runtime.model.state_dict(), fresh.model.state_dict())
    assert fresh.session(2).continuation.cut == 0
    assert runtime.session(2).continuation.cut == 0
    restored.reset()
    assert restored.continuation.cut == 0


def test_reject_invalid_configuration_and_ignored_options():
    raw = config("pdg", torch.float32).to_dict()
    for invalid in [raw | {"schema_version":True}, raw | {"surprise":1},
                    raw | {"topology":{"kind":"chain"}}, raw | {"model":{"width":True}}]:
        with pytest.raises(ValueError):
            GraphConfig.from_dict(invalid)
    cfg = config("pdg", torch.float32)
    for options in [dict(prefill=True), dict(max_events=3), dict(workers=2),
                    dict(batch_next=True), dict(fiber_pooling="csr")]:
        with pytest.raises(ValueError):
            GraphRuntime(cfg, device="cpu", options=options)
    with pytest.raises(ValueError, match="budget"):
        GraphConfig.from_dict(dict(schema_version=1, family="pdg", topology=dict(kind="chain", budget=2)))
    with pytest.raises(ValueError):
        GraphConfig.from_dict(dict(schema_version=1, family="timed-dag", topology=dict(kind="ring")))


def test_clock_override_and_malformed_checkpoint(tmp_path):
    cfg = GraphConfig.from_dict(dict(schema_version=1, family="pdg",
        topology=dict(kind="chain", module=dict(state_clock=dict(period=2, first=0, count=1)))))
    assert cfg.graph.nodes[0].state_clock.period == 2
    runtime = GraphRuntime(cfg, device="cpu")
    file = tmp_path / "bad.pt"
    torch.save({"schema":"tide-continuation-v5"}, file)
    before = {k:v.clone() for k,v in runtime.model.state_dict().items()}
    with pytest.raises(ValueError, match="malformed"):
        runtime.session(1).load(file)
    equivalent(before, runtime.model.state_dict())


def test_python_import_is_lazy_and_explicit_native_loader(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build = Path(os.environ.get("TIDE_BUILD_DIR", root / "build")).resolve()
    code = """
import sys
from tidegraph import GraphConfig, GraphRuntime
assert '_tide_native' not in sys.modules and 'torch_npu' not in sys.modules
cfg=GraphConfig.from_dict(dict(schema_version=1, family='pdg', topology=dict(kind='chain')))
GraphRuntime(cfg, device='cpu')
assert '_tide_native' not in sys.modules and 'torch_npu' not in sys.modules
r=GraphRuntime(cfg, device='cpu', options=dict(implementation='native'), native_library=sys.argv[1])
assert r.manifest()['native_sha256']
"""
    env = dict(os.environ, PYTHONPATH=str(root / "python"))
    subprocess.run([sys.executable, "-c", code, str(build)], cwd=tmp_path, env=env, check=True)


def test_shared_owners_and_custom_module_boundary(dtype):
    from tidegraph.ops import Model
    from tidegraph.full import ProjectionEmit
    class CustomFull(ProjectionEmit):
        pass
    cfg = config("timed-dag", dtype)
    model = Model(cfg.graph, cfg.width, dtype=dtype,
                  full_programs={0:CustomFull(cfg.graph.nodes[0])})
    model.nodes[1].weight = model.nodes[0].weight
    runtime = GraphRuntime(cfg, device="cpu", model=model)
    result = advance(runtime.session(1), torch.ones(1, 4, 4, dtype=dtype), 0, 8)
    objective(result).backward()
    assert model.nodes[0].weight is model.nodes[1].weight
    with pytest.raises(ValueError, match="custom Full"):
        GraphRuntime(cfg, device="cpu", model=model, options=dict(implementation="native"))
