"""Public resident ownership, online inputs, export/restore and explicit refusal."""
from dataclasses import replace
import os
import pytest
import torch
from tidegraph import (Edge, Graph, Node, Region, GraphConfig, GraphRuntime,
                       ExecutionOptions, ExecutionPlacement, ResidentLimits, External)
from tidegraph.compare import equivalent


@pytest.fixture
def target():
    device = os.environ.get("TIDE_RESIDENT_DEVICE")
    if device is None:
        pytest.skip("optional NPU resident target not requested")
    import torch_npu
    assert device.startswith("npu"), "resident target must explicitly select NPU"
    assert os.environ.get("TIDE_RESIDENT_LIBRARY"), "explicit target requires its backend build"
    return device


def config(family, memory="ema"):
    nodes = tuple(Node(i//2, memory=memory, full="swiglu", readout="norm-fp32-v1",
                       query_heads=2, kv_heads=2 if memory.startswith("lh-fiber") else 1) for i in range(4))
    edges = (Edge(0,2,1), Edge(0,2,1), Edge(1,2,1), Edge(1,3,1))
    if family != "settle":
        edges += (Edge(0,3,2),)
    if family == "pdg":
        edges += (Edge(3,0,3),)
    graph = Graph(nodes, edges, (Region(1), Region(1)), (0,1), (2,3))
    return GraphConfig(family, graph, width=4, ranks=(1,2) if family == "settle" else ())


def runtime(cfg, target, schedule="greedy", mode="hard", resident_workspace_bytes=64*1024*1024, **kwargs):
    options = ExecutionOptions(implementation="native", schedule=schedule, mode=mode, trace=True,
        placement=ExecutionPlacement(preset="resident"), resident_limits=ResidentLimits(
            queue=96, arrivals=128, outputs=128, trace=2048, kv_rows=64, kv_trace_rows=2048,
            workspace_bytes=resident_workspace_bytes))
    return GraphRuntime(cfg, device=target, options=options,
                        resident_library=os.environ["TIDE_RESIDENT_LIBRARY"], **kwargs)


def inputs(session, x, start, stop):
    if session.runtime.spec:
        return (x[:,start:stop],), {}
    records = [External(b,p,t,t,x[b,t]) for b in range(len(x))
               for p in range(len(session.runtime.graph.inputs)) for t in range(start,min(stop,x.shape[1]))]
    return (records,), dict(stop=stop, sealed_until=stop)


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("memory", ["ema", "attention", "lh-fiber-attention-all-softmax-repeat-v1"])
def test_resident_public_online_windows(target, family, schedule, memory, tmp_path):
    online_windows(target, family, schedule, memory, tmp_path)


@pytest.mark.parametrize("mode", ["hst", "softp"])
@pytest.mark.parametrize("memory", ["attention", "lh-fiber-attention-all-softmax-repeat-v1"])
def test_resident_control_attention_windows(target, mode, memory, tmp_path):
    online_windows(target, "pdg", "greedy", memory, tmp_path, mode)


def online_windows(target, family, schedule, memory, tmp_path, mode="hard"):
    cfg = config(family, memory)
    r = runtime(cfg,target,schedule,mode)
    oracle = GraphRuntime(cfg,device="cpu",options=ExecutionOptions(schedule="reference",packed=False,trace=True,mode=mode))
    x = torch.sin(torch.arange(64,dtype=torch.float32).reshape(2,8,4)*.37)*.2
    x[0,1].zero_()  # A present-zero input is still an event.
    with torch.no_grad():
        baseline = oracle.session(2)
        with r.session(2) as session:
            for start,stop in ((0,2),(2,4),(4,8)):
                args,kw = inputs(baseline,x,start,stop)
                expected = baseline.advance(*args,**kw)
                # Alternate boundary locations, including non-default streams.
                stream = torch.npu.Stream(device=target)
                with torch.npu.stream(stream):
                    y = x.to(target) if start != 2 else x
                    args,kw = inputs(session,y,start,stop)
                    view = session.advance_device(*args,**kw)
                    assert view.values.device == torch.device(target)
                    assert view.coordinates.dtype == torch.int64 and view.valid.dtype == torch.bool
                equivalent(expected,session.result())
                # Mutating an exported checkpoint cannot mutate live state.
                exported = session.snapshot()
                for s in exported.states.values():
                    s.value.fill_(float("nan"))
            checkpoint = tmp_path / "resident.pt"
            session.save(checkpoint)
            final = session.snapshot()
        other = runtime(cfg,target,"streaming" if schedule=="greedy" else "greedy",mode)
        with other.session(2) as restored:
            restored.load(checkpoint)
            equivalent(final,restored.snapshot())
            args,kw = inputs(restored,x,8,8 if family=="settle" else 11)
            actual = restored.advance(*args,**kw)
            args,kw = inputs(baseline,x,8,8 if family=="settle" else 11)
            expected = baseline.advance(*args,**kw)
            equivalent(expected,actual)
        assert r.manifest()["resident"]["autograd"] is False


def test_resident_changed_input_and_parameters(target):
    cfg = config("pdg")
    r = runtime(cfg,target)
    with pytest.raises(ValueError,match="no_grad"):
        r.session(1)
    x = torch.arange(16,dtype=torch.float32).reshape(1,4,4)*.01
    with torch.no_grad():
        with r.session(1) as first, r.session(1) as second:
            a,kw = inputs(first,x,0,4);b,_ = inputs(second,-x,0,4)
            first.advance_device(*a,**kw);second.advance_device(*b,**kw)
            assert any(not torch.equal(v.value,second.snapshot().states[k].value)
                       for k,v in first.snapshot().states.items())
            r.model.nodes[0].bias.add_(.1)
            with pytest.raises(RuntimeError,match="parameters changed"):
                first.advance_device([],stop=5,sealed_until=5)
            with pytest.raises(RuntimeError,match="changed parameters"):
                first.save("unused.pt")
            q = first.snapshot()
        with r.session(1,continuation=q) as refreshed:
            oracle = GraphRuntime(cfg,device="cpu",options=ExecutionOptions(schedule="reference",packed=False,trace=True))
            oracle.model.load_state_dict(r.model.state_dict())
            expected = oracle.session(1,continuation=q).advance([],stop=9,sealed_until=9)
            equivalent(expected,refreshed.advance([],stop=9,sealed_until=9))
            refreshed.close()
            with pytest.raises(RuntimeError,match="closed"):
                refreshed.snapshot()


def test_resident_refuses_partial_capacity_and_unsupported_module(target):
    cfg = config("pdg")
    r = runtime(cfg,target)
    r.options = replace(r.options,resident_limits=replace(r.options.resident_limits,stages=1))
    with torch.no_grad(), r.session(1) as session:
        x = torch.ones(1,4,4)*.01
        args,kw = inputs(session,x,0,4)
        with pytest.raises(RuntimeError,match="refusal"):
            session.advance_device(*args,**kw)
        with pytest.raises(RuntimeError,match="failed"):
            session.snapshot()
    bad = runtime(config("pdg","ssm"),target)
    with torch.no_grad(), pytest.raises(ValueError,match="module contract"):
        bad.session(1)


def test_resident_limits_and_host_configuration():
    with pytest.raises(ValueError,match="capacity"):
        ResidentLimits(queue=0)
    cfg = config("pdg")
    with pytest.raises(ValueError,match="require device-resident"):
        GraphRuntime(cfg,device="cpu",options=ExecutionOptions(resident_limits=ResidentLimits()))
    options = ExecutionOptions(resident_limits=dict(queue=31,chunk_policy="aggressive"))
    assert options.resident_limits.queue == 31


def test_resident_loader_rejects_standalone_and_mismatched_core(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    from tidegraph.resident_loader import load_resident
    native = tmp_path / "_tide_native.so"
    native.write_bytes(b"core identity only; never imported")
    core = SimpleNamespace(__file__=str(native))
    record = dict(npu_runtime="standalone")
    manifest = tmp_path / "control-build.json"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Python-owned"):
        load_resident(tmp_path, core)
    record.update(npu_runtime="python", core=dict(binary_sha256={native.name:"wrong"}))
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exact native core"):
        load_resident(tmp_path, core)
    record["core"]["binary_sha256"][native.name] = hashlib.sha256(native.read_bytes()).hexdigest()
    record["binary_sha256"] = {"_tide_resident.so":"wrong"}
    (tmp_path / "_tide_resident.so").write_bytes(b"untrusted plugin; never imported")
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="binary differs"):
        load_resident(tmp_path, core)


def test_resident_lean_windows_do_not_export_continuation(target, monkeypatch):
    cfg = config("pdg")
    r = runtime(cfg,target)
    r.options = replace(r.options,trace=False)
    baseline = GraphRuntime(cfg,device="cpu",options=ExecutionOptions(schedule="reference",packed=False,trace=False))
    x = torch.cos(torch.arange(32,dtype=torch.float32).reshape(1,8,4)*.13)*.1
    with torch.no_grad(), r.session(1) as session:
        oracle = baseline.session(1)
        def forbidden(*args, **kwargs):
            raise AssertionError("advance_device must not export continuation")
        # Patching this conversion rejects all implicit Python snapshot/result
        # materialization while allowing device outputs to be consumed.
        import tidegraph.resident as adapter
        with monkeypatch.context() as patch:
            patch.setattr(adapter,"from_continuation",forbidden)
            for start,stop in ((0,2),(2,5),(5,8)):
                args,kw = inputs(oracle,x,start,stop);expected=oracle.advance(*args,**kw)
                args,kw = inputs(session,x.to(target),start,stop)
                view=session.advance_device(*args,**kw)
                assert view.values.device.type=="npu"
        equivalent(expected,session.result())


def test_resident_invalid_input_preserves_complete_cut(target):
    r = runtime(config("pdg"),target)
    with torch.no_grad(), r.session(1) as session:
        before = session.snapshot()
        for value in (torch.full((4,),float("nan"),device=target),
                      torch.ones(4,dtype=torch.float16,device=target),
                      torch.ones(3,device=target)):
            with pytest.raises(ValueError):
                session.advance_device([External(0,0,0,0,value)],stop=1,sealed_until=1)
            equivalent(before,session.snapshot())
        with pytest.raises(ValueError,match="port history"):
            session.advance_device([External(0,0,1,0,torch.ones(4,device=target))],stop=1,sealed_until=1)
        equivalent(before,session.snapshot())
        session.advance_device([External(0,0,0,0,torch.ones(4,device=target))],stop=1,sealed_until=1)
        assert session.cut == 1


def test_resident_checkpoint_preflight_and_reset(target, tmp_path):
    r = runtime(config("settle"),target)
    with torch.no_grad(), r.session(1) as session:
        x = torch.ones(1,2,4,device=target)*.1
        session.advance_device(x)
        before = session.snapshot()
        weights = {k:v.cpu().clone() for k,v in r.model.state_dict().items()}
        path = tmp_path / "complete.pt"
        session.save(path)
        record = torch.load(path,weights_only=True)
        record["cut"] -= 1  # Incomplete Settle position must not change weights/state.
        invalid = tmp_path / "invalid.pt"
        torch.save(record,invalid)
        with pytest.raises(ValueError,match="complete position"):
            session.load(invalid)
        equivalent(before,session.snapshot())
        equivalent(weights,{k:v.cpu() for k,v in r.model.state_dict().items()})
        session.reset()
        assert session.position == 0 and not session.snapshot().states
        session.load(path)
        equivalent(before,session.snapshot())
