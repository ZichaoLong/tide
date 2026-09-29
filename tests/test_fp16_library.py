"""Low precision keeps discrete and ownership contracts; numerical policy is explicit."""
import copy
from pathlib import Path
import pytest
import torch
from tidegraph import GraphConfig, GraphRuntime, FP32MasterOptimizer
from tidegraph.qualification import qualify


@pytest.mark.parametrize("family", ["pdg", "timed-dag", "settle"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_fp16_complete_gate(family, implementation, tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "python"))
    topology = dict(kind="ring", size=4) if family == "pdg" else dict(kind="diamond")
    topology.update(module=dict(full="swiglu"), node_overrides={"0":dict(memory="linear"),
                    "1":dict(memory="ssm"), "2":dict(memory="delta"), "3":dict(memory="attention")})
    cfg = GraphConfig.from_dict(dict(schema_version=1, family=family, topology=topology,
        model=dict(width=4, dtype="float16"), execution=dict(implementation=implementation, mode="hst")))
    report = qualify(cfg, device="cpu", output_dir=tmp_path / "gate", positions=3, steps=2,
                     native_library=os.environ.get("TIDE_BUILD_DIR") if implementation == "native" else None)
    assert report["state"] == "passed"
    assert report["tolerances"] == dict(atol=1e-3, rtol=2e-2)
    assert {x["name"] for x in report["checks"]} >= {"complete-observables", "independent-vjps",
        "chunk-observables-and-vjps", "optimizer-trajectory", "fresh-process-checkpoint"}


def test_fp16_init_is_quantized_fp32():
    base = dict(schema_version=1, family="pdg", topology=dict(kind="ring", size=2), model=dict(width=4))
    full = GraphRuntime(base, device="cpu")
    base["model"]["dtype"] = "float16"
    half = GraphRuntime(base, device="cpu")
    for name, value in full.model.state_dict().items():
        assert torch.equal(value.half(), half.model.state_dict()[name])


def test_master_update_none_zero_and_overflow():
    p, unused, zero = [torch.nn.Parameter(torch.tensor([x], dtype=torch.float16)) for x in (1.,2.,3.)]
    opt = FP32MasterOptimizer([p, unused, zero], optimizer="sgd", lr=.1, weight_decay=.01, momentum=.9)
    opt.backward(p.float().sum()*2 + zero.float().sum()*0)
    opt.step()
    torch.testing.assert_close(opt.masters[0], torch.tensor([.799]))
    torch.testing.assert_close(opt.masters[2], torch.tensor([2.997]))
    assert opt.masters[1].item()==2. and unused.grad is None
    assert len(opt.state)==2
    assert all(t.dtype==torch.float32 for t in opt.masters)
    assert all(v.dtype==torch.float32 for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    opt.zero_grad()
    opt.backward(p.float().sum()*1e10)
    before = opt.masters[0].clone()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        opt.step()
    assert torch.equal(before,opt.masters[0])


def test_master_checkpoint_preflight_preserves_live_weights(tmp_path):
    cfg = dict(schema_version=1, family="pdg", topology=dict(kind="self_loop", size=1),
               model=dict(width=2, dtype="float16"))
    runtime = GraphRuntime(cfg, device="cpu")
    session = runtime.session(1)
    opt = FP32MasterOptimizer(runtime.model.parameters(), lr=.01)
    opt.backward(sum(p.float().square().sum() for p in runtime.model.parameters()))
    opt.step()
    opt.param_groups[0]["lr"] = .007
    checkpoint = tmp_path/"state.pt"
    session.save(checkpoint,opt)
    record = torch.load(checkpoint,weights_only=True)
    bad = copy.deepcopy(record)
    bad["optimizer"]["precision"]["masters"][0].add_(1.)
    corrupt = tmp_path/"corrupt.pt";torch.save(bad,corrupt)
    before = {k:v.clone() for k,v in runtime.model.state_dict().items()}
    with pytest.raises(ValueError,match="master and payload"):
        session.load(corrupt,opt)
    assert all(torch.equal(v,runtime.model.state_dict()[k]) for k,v in before.items())
    opt.param_groups[0]["lr"] = .5
    session.load(checkpoint,opt)
    assert opt.loss_scale==128. and opt.param_groups[0]["lr"]==.007
    with pytest.raises(ValueError,match="unique"):
        FP32MasterOptimizer([opt.payload[0],opt.payload[0]])
