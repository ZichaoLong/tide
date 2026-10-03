"""Actual multi-owner consumers: independent model construction and full updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from tools.online_bench.eager_placement import owner_indices
from tools.online_bench.fixture import build_model
from tools.online_bench.host import run, runtime_for
from flow_protocol import native_text
from online_consumer_support import observer, same
from test_online_consumer import packet
from test_online_consumer_npu import CASES


def target():
    spec=os.environ.get("TIDE_ONLINE_DEVICE")
    if not spec:pytest.skip("two-device eager consumer target not explicitly selected")
    from tidegraph.runtime import resolve_device
    device,_=resolve_device(spec)
    assert device.type in {"cuda","npu"},"explicit accelerator target required"
    assert device.index is not None and getattr(torch,device.type).device_count()>=device.index+2
    return device


@pytest.mark.parametrize("memory", ["add", "attention"])
@pytest.mark.parametrize("policy", ["memory", "locality"])
def test_static_parameter_accounting(memory, policy):
    p = packet(memory, delayed=True)
    owners, elements = owner_indices(p, 3, policy)
    _, model, embedding, head = build_model(p)
    actual = [embedding.numel()+head.numel(), 0, 0]
    for i, node in enumerate(model.nodes):
        actual[owners[i]] += sum(v.numel() for v in node.parameters() if v.requires_grad)
    assert elements == actual and sum(elements) == p["counts"]["parameters"]
    assert owners[0] == 0 and owners[-2:] == [0, 0] and set(owners) == {0, 1, 2}
    assert owner_indices(p, 3, policy, tuple(owners)) == (owners, elements)


@pytest.mark.parametrize("bad", [(0,1), (True,1,1,1,1,1,0,0), (0,1,1,1,1,1,0,1), (0,)*8])
def test_invalid_owner_map_precedes_construction(bad, monkeypatch):
    import tools.online_bench.host as host
    def forbidden(*args, **kwargs):
        raise AssertionError("model allocation entered")
    monkeypatch.setattr(host, "build_model", forbidden)
    with pytest.raises(ValueError, match="owner map"):
        host.run(packet(), family="timed-dag", implementation="python", device="cpu", devices=2, owner_map=bad)


def test_cpu_multiple_devices_rejected_before_construction(monkeypatch):
    import tools.online_bench.host as host
    def forbidden(*args, **kwargs):
        raise AssertionError("model allocation entered")
    monkeypatch.setattr(host, "build_model", forbidden)
    with pytest.raises(ValueError, match="accelerator"):
        host.run(packet(), family="timed-dag", implementation="python", device="cpu", devices=2)


def test_per_owner_constants_and_actual_parameters():
    device = target();p = packet("attention")
    runtime, embedding, head = runtime_for(p, family="timed-dag", implementation="python",
        device=device, dtype="float32", schedule="prefill", preset="mixed-c", devices=2)
    owners, elements = owner_indices(p, 2)
    groups = {}
    for node, owner in zip(runtime.model.nodes[:-2], owners[:-2]):
        assert node.bias.device.index == device.index+owner
        if owner in groups:
            assert node.weight is groups[owner].weight
        groups[owner] = node
    assert len(groups) == 2 and groups[0].weight is not groups[1].weight
    actual = [embedding.numel()+head.numel(), 0]
    for p in runtime.model.parameters():
        if p.requires_grad:
            actual[p.device.index-device.index] += p.numel()
    assert actual == elements


@pytest.mark.parametrize("case", CASES, ids=['-'.join(c) for c in CASES])
@pytest.mark.parametrize("implementation", ["python", "native", "libtorch"])
@pytest.mark.parametrize("training", [True, False], ids=["train", "infer"])
def test_actual_two_device_run(case, implementation, training, tmp_path):
    device = target();family, memory, schedule, preset = case
    p = packet(memory, delayed=family=="timed-dag");expected=[];actual=[]
    policy = "memory" if memory=="attention" else "locality"
    # Alternate explicit and automatic layouts; both independently build leaves.
    explicit = (0,1,0,1,0,1,0,0) if schedule=="streaming" else ()
    common = dict(family=family, training=training, optimizer="adamw", steps=2, warmup=0,
                  windows_per_step=2, sample_chunk_rows=1, diagnostics=True)
    reference = run(p, implementation="python", device="cpu", schedule="streaming",
                    observer=observer(expected), **common)
    if implementation=="libtorch":
        path=tmp_path/"packet.txt";path.write_text(native_text(p));out=tmp_path/"standalone"
        cmd=[os.environ["TIDE_ONLINE_BINARY"],"--device="+str(device),"--dtype=float32",
             "--packet="+str(path),"--output-dir="+str(out),"--family="+family,"--preset="+preset,
             "--schedule="+schedule,"--optimizer=adamw","--steps=2","--warmup=0",
             "--windows-per-step=2","--sample-chunk-rows=1","--diagnostics","--devices=2",
             "--owner-policy="+policy,"--workers=2"]
        if explicit:cmd.append("--owner-map="+','.join(map(str, explicit)))
        if training:cmd.append("--training")
        if schedule=="prefill":cmd += ["--packed-sources", "--batch-next"]
        done=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        assert done.returncode==0,done.stdout+done.stderr
        actual=[json.loads(line) for line in (out/"diagnostics.jsonl").read_text().splitlines()]
        measured=json.loads((out/"result.json").read_text())
    else:
        measured=run(p,implementation=implementation,device=device,schedule=schedule,preset=preset,
            devices=2,owner_policy=policy,owner_map=explicit,observer=observer(actual),
            workers=2 if implementation=="native" else 1,
            packed_sources=implementation=="native" and schedule=="prefill",
            batch_next=implementation=="native" and schedule=="prefill",
            native_library=os.environ["TIDE_BUILD_DIR"] if implementation=="native" else None, **common)
    same(actual,expected)
    torch.testing.assert_close(torch.tensor(measured["losses"]),torch.tensor(reference["losses"]),atol=1e-6,rtol=1e-5)
    for field in ("parameters","outputs","final_cut","batch_execution"):
        assert measured[field]==reference[field]
    owners,elements=owner_indices(p,2,policy,explicit)
    assert measured["payload_placement"]["node_owners"]==owners
    assert measured["payload_placement"]["parameter_elements"]==elements
    for phase in measured["memory"]["phases"]:
        assert [r["device"] for r in phase["devices"]]==[str(device),f"{device.type}:{device.index+1}"]
    assert all(r["allocated_bytes"]>0 for r in measured["memory"]["phases"][1]["devices"])


@pytest.mark.parametrize("implementation", ["python", "native", "libtorch"])
def test_unified_cli_owner_replay(implementation, tmp_path):
    device=target();p=packet("attention");path=tmp_path/"packet.json";path.write_text(json.dumps(p))
    out=tmp_path/"cli";owners=(0,1,0,1,0,1,0,0)
    cmd=[sys.executable,str(ROOT/"scripts/run_execution_flow.py"),"--packet",str(path),"--output-dir",str(out),
         "--device",str(device),"--dtype","float32","--family","timed-dag","--preset","mixed-c",
         "--schedule","prefill","--implementation",implementation,"--devices","2",
         "--owner-map",','.join(map(str,owners)),"--steps","1","--warmup","0","--training",
         "--sample-chunk-rows","1","--phase-timing"]
    if implementation=="libtorch":cmd += ["--native-binary",os.environ["TIDE_ONLINE_BINARY"]]
    elif implementation=="native":cmd += ["--native-library",os.environ["TIDE_BUILD_DIR"]]
    done=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    assert done.returncode==0,done.stdout+done.stderr
    record=json.loads((out/"result.json").read_text())
    assert record["payload_placement"]["node_owners"]==list(owners)
    assert record["payload_placement"]["owner_selection"]=="explicit"
    assert len(record["memory"]["phases"][1]["devices"])==2
    assert record["phase_timing"]["enabled"] and len(record["phase_timing"]["measured"])==1
