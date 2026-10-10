"""Public inference with separate Full/state owners and portable complete cuts."""
from dataclasses import replace
import pytest
import torch
from resident_test_target import owner_devices
from tidegraph import GraphRuntime, ExecutionOptions, ResidentPlacement
from tidegraph.compare import equivalent
from test_resident_library import target, config, runtime, inputs


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("family,memory,mode", [
    ("pdg", "lh-fiber-attention-all-softmax-repeat-v1", "hard"),
    ("timed-dag", "attention", "hst"), ("settle", "ema", "softp")])
def test_sharded_inference_windows_and_repartition(target, family, memory, mode, dtype_name, schedule, tmp_path):
    cfg = replace(config(family, memory), dtype=dtype_name)
    r = runtime(cfg, target, schedule, mode)
    nodes = len(r.execution_graph.nodes)
    owners = ResidentPlacement(devices=owner_devices(target, 2),
        full_owners=tuple(i % 2 for i in range(nodes)),
        state_owners=tuple((i+1) % 2 for i in range(nodes)))
    oracle = GraphRuntime(cfg, device="cpu",
        options=ExecutionOptions(schedule="reference", packed=False, trace=True, mode=mode))
    values = (torch.sin(torch.arange(64, dtype=torch.float32).reshape(2, 8, 4)*.37)*.2).to(getattr(torch, dtype_name))
    values[0, 1].zero_()
    tolerance = dict(atol=2e-3, rtol=2e-2) if dtype_name=="float16" else dict(atol=1e-6, rtol=1e-5)
    checkpoint = tmp_path / "inference.pt"
    with torch.no_grad():
        baseline = oracle.session(2)
        with r.session(2, placement=owners) as session:
            assert session.placement == owners.to_dict()
            manifest = session.manifest()["resident"]
            assert manifest["requested_inference_placement"] == owners.to_dict()
            assert manifest["resolved_inference_placement"] == session.placement
            assert not manifest["training"]
            for start, stop in ((0, 2), (2, 4)):
                args, kw = inputs(baseline, values, start, stop)
                expected = baseline.advance(*args, **kw)
                args, kw = inputs(session, values.to(target) if start==0 else values, start, stop)
                view = session.advance_device(*args, **kw)
                assert view.values.dtype == values.dtype and str(view.values.device) == target
                assert view.coordinates.dtype == torch.int64
                equivalent(expected, session.result(), **tolerance)
            session.save(checkpoint)
            complete = session.snapshot()
            session.reset()
            assert session.cut == 0 and session.placement == owners.to_dict()
            session.load(checkpoint)
            equivalent(complete, session.snapshot(), atol=0, rtol=0)
            assert session.placement == owners.to_dict()
        # Checkpoints contain global state/KV, so the next session may repartition.
        next_owners = (ResidentPlacement(devices=owner_devices(target, 3), policy="memory")
                       if schedule=="greedy" else ResidentPlacement())
        other = runtime(cfg, target, "streaming" if schedule=="greedy" else "greedy", mode)
        with other.session(2, placement=next_owners) as restored:
            restored.load(checkpoint)
            assert len(restored.placement["devices"]) == (3 if schedule=="greedy" else 1)
            equivalent(complete, restored.snapshot(), atol=0, rtol=0)
            for start, stop in ((4, 8), (8, 8 if family=="settle" else 11)):
                args, kw = inputs(baseline, values, start, stop)
                expected = baseline.advance(*args, **kw)
                args, kw = inputs(restored, values, start, stop)
                restored.advance_device(*args, **kw)
                equivalent(expected, restored.result(), **tolerance)


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_sharded_lean_inference_never_creates_training_owner(target, dtype_name, monkeypatch):
    cfg = replace(config("pdg"), dtype=dtype_name)
    r = runtime(cfg, target)
    r.options = replace(r.options, trace=False)
    reference = GraphRuntime(cfg, device="cpu",
        options=ExecutionOptions(schedule="reference", packed=False, trace=False))
    values = torch.full((1, 4, 4), .125, dtype=getattr(torch, dtype_name))
    owners = ResidentPlacement(devices=owner_devices(target, 2))
    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not create training owners or implicitly export continuation")
    monkeypatch.setattr(r.engine.module, "TrainingSession", forbidden)
    with torch.no_grad(), r.session(1, placement=owners) as session:
        oracle = reference.session(1)
        import tidegraph.resident as adapter
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "from_continuation", forbidden)
            for start, stop in ((0, 2), (2, 4)):
                args, kw = inputs(oracle, values, start, stop)
                expected = oracle.advance(*args, **kw)
                args, kw = inputs(session, values.to(target), start, stop)
                view = session.advance_device(*args, **kw)
                assert view.values.device.type == torch.device(target).type
        equivalent(expected, session.result(), atol=2e-3 if dtype_name=="float16" else 1e-6,
                   rtol=2e-2 if dtype_name=="float16" else 1e-5)


@pytest.mark.parametrize("placement,match", [
    (dict(devices=(1, 0)), "coordinator"),
    (dict(devices=(0, 1), full_owners=(0,)), "every execution node"),
    (dict(devices=(0, 1), state_owners=(0,0,0,0)), "empty Full shard")])
def test_inference_placement_refusals(target, placement, match):
    devices = owner_devices(target, 2)
    placement = dict(placement, devices=tuple(devices[i] for i in placement["devices"]))
    r = runtime(config("pdg"), target)
    with torch.no_grad(), pytest.raises(ValueError, match=match):
        r.session(1, placement=placement)


def test_host_session_refuses_resident_owner_mapping():
    r = GraphRuntime(config("pdg"), device="cpu")
    with pytest.raises(ValueError, match="requires resident"):
        r.session(1, placement=ResidentPlacement())
