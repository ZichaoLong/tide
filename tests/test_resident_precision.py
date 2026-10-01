"""Resident FP16 inference and explicit FP32 training-root boundaries."""
from dataclasses import replace
import pytest
import torch
from tidegraph import GraphRuntime, ExecutionOptions, ExecutionPlacement
from tidegraph.compare import equivalent
from test_resident_library import target, config, runtime, inputs


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("schedule", ["streaming", "greedy"])
@pytest.mark.parametrize("memory", ["ema", "attention", "lh-fiber-attention-all-softmax-repeat-v1"])
@pytest.mark.parametrize("mode", ["hard", "hst", "softp"])
def test_resident_fp16_continuation(target, family, schedule, memory, mode, tmp_path):
    cfg = replace(config(family, memory), dtype="float16")
    r = runtime(cfg, target, schedule, mode)
    oracle = GraphRuntime(cfg, device="cpu", options=ExecutionOptions(schedule="reference", packed=False, trace=True, mode=mode))
    values = (torch.sin(torch.arange(64, dtype=torch.float32).reshape(2, 8, 4)*.37)*.2).half()
    values[0, 1].zero_()
    assert r.manifest()["resident"]["dtype"] == "float16"
    assert r.manifest()["resident"]["scoring_dtype"] == "float32"
    assert r.manifest()["resident"]["exported_control_dtype"] == "float16"
    with torch.no_grad(), r.session(2) as session:
        baseline = oracle.session(2)
        for start, stop in ((0, 2), (2, 4), (4, 8)):
            args, kw = inputs(baseline, values, start, stop)
            expected = baseline.advance(*args, **kw)
            args, kw = inputs(session, values.to(target) if start != 2 else values, start, stop)
            view = session.advance_device(*args, **kw)
            assert view.values.dtype == torch.float16 and view.values.device == torch.device(target)
            equivalent(expected, session.result(), atol=2e-3, rtol=2e-2)
        checkpoint = tmp_path / "resident-half.pt"
        session.save(checkpoint)
        final = session.snapshot()
        other = runtime(cfg, target, "streaming" if schedule == "greedy" else "greedy", mode)
        with other.session(2) as restored:
            restored.load(checkpoint)
            equivalent(final, restored.snapshot(), atol=0, rtol=0)
            args, kw = inputs(restored, values, 8, 8 if family == "settle" else 11)
            restored.advance_device(*args, **kw)
            args, kw = inputs(baseline, values, 8, 8 if family == "settle" else 11)
            equivalent(baseline.advance(*args, **kw), restored.result(), atol=2e-3, rtol=2e-2)


def test_resident_fp16_scope_refusals(target):
    cfg = replace(config("pdg"), dtype="float16")
    r = runtime(cfg, target)
    options = replace(r.options, placement=ExecutionPlacement(preset="resident", scoring_dtype="payload"))
    with pytest.raises(ValueError, match="FP32"):
        GraphRuntime(cfg, device=target, options=options)
    with torch.no_grad(), r.training_session(1) as session:
        window = session.advance_device([], stop=0, sealed_until=0)
        with pytest.raises(ValueError, match="cotangent layout"):
            session.backward([session.cotangents(window, outputs=torch.zeros_like(window.outputs.values))])
        gradient = session.backward([session.cotangents(window)])
        assert gradient.values.dtype == torch.float32
        assert not gradient.connected.any()
        session.detach()
