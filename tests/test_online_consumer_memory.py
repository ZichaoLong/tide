"""Real phase peaks across CPU/eager/resident consumers; no timing thresholds."""
import json
import os
import subprocess
from pathlib import Path
import pytest
import torch
from test_online_consumer import packet
from test_online_consumer_npu import target
from flow_protocol import native_text
from tools.online_bench.host import run
from tools.online_bench.memory import MemoryRecord
from tidegraph import ResidentPlacement


def observed(record, devices):
    memory = record["memory"]
    assert memory["schema"] == "tide-consumer-memory-v1"
    assert [p["phase"] for p in memory["phases"]] == ["initial", "construction", "warmup", "measured"]
    previous_rss = 0
    for phase in memory["phases"]:
        assert phase["cpu_peak_rss_bytes"] >= previous_rss > -1
        previous_rss = phase["cpu_peak_rss_bytes"]
        assert [d["device"] for d in phase["devices"]] == devices
        for d in phase["devices"]:
            assert 0 <= d["allocated_bytes"] <= d["peak_allocated_bytes"] <= d["peak_reserved_bytes"]
            assert d["allocated_bytes"] <= d["reserved_bytes"] <= d["peak_reserved_bytes"]
    if devices:
        for d in memory["phases"][-1]["devices"]:
            assert d["peak_allocated_bytes"] > 0


def standalone(p, device, preset, dtype, training, devices, tmp_path):
    binary = os.environ.get("TIDE_ONLINE_BINARY")
    if not binary:
        pytest.skip("standalone memory consumer not explicitly selected")
    path = tmp_path/"topology.txt";path.write_text(native_text(p));out = tmp_path/"consumer"
    command = [binary, "--device="+str(device), "--dtype="+dtype, "--packet="+str(path), "--output-dir="+str(out),
               "--family=timed-dag", "--schedule=prefill", "--preset="+preset, "--steps=1", "--warmup=1",
               "--windows-per-step=2", "--optimizer=adamw"]
    if training:
        command.append("--training")
    if preset == "resident":
        command.append("--devices="+str(devices))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert result.returncode == 0, result.stdout
    return json.loads((out/"result.json").read_text())


@pytest.mark.parametrize("implementation", ["python", "libtorch"])
@pytest.mark.parametrize("training", [False, True])
def test_cpu_consumer_memory(implementation, training, tmp_path):
    p = packet("attention")
    reference = run(p, family="timed-dag", implementation="python", device="cpu", schedule="streaming",
                    steps=1, warmup=1, windows_per_step=2, training=training, optimizer="adamw")
    candidate = (standalone(p, "cpu", "cpu", "float32", training, 1, tmp_path) if implementation == "libtorch" else
        run(p, family="timed-dag", implementation="python", device="cpu", schedule="prefill",
            steps=1, warmup=1, windows_per_step=2, training=training, optimizer="adamw"))
    observed(candidate, [])
    assert candidate["outputs"] == reference["outputs"] and candidate["final_cut"] == reference["final_cut"]
    torch.testing.assert_close(torch.tensor(candidate["losses"]), torch.tensor(reference["losses"]))
    (tmp_path/"memory.json").write_text(json.dumps(candidate))


@pytest.mark.parametrize("implementation,preset,devices,payload_dtype,training", [
    ("python", "mixed-a", 1, "float32", False), ("native", "mixed-c", 1, "float32", True),
    ("libtorch", "mixed-b", 1, "float32", True), ("native", "resident", 2, "float32", True),
    ("libtorch", "resident", 2, "float16", True), ("native", "resident", 1, "float16", False)])
def test_npu_consumer_memory(implementation, preset, devices, payload_dtype, training, tmp_path):
    device = target();p = packet("attention")
    reference = run(p, family="timed-dag", implementation="python", device="cpu", schedule="streaming",
                    steps=1, warmup=1, windows_per_step=2, training=training, optimizer="adamw")
    logical = [f"npu:{device.index+i}" for i in range(devices)]
    if implementation == "libtorch":
        candidate = standalone(p, device, preset, payload_dtype, training, devices, tmp_path)
    else:
        options = dict(native_library=os.environ["TIDE_BUILD_DIR"]) if implementation == "native" else {}
        if preset == "resident":
            options.update(resident_library=os.environ["TIDE_RESIDENT_LIBRARY"],
                           resident_placement=ResidentPlacement(devices=tuple(logical)) if devices>1 else None)
        candidate = run(p, family="timed-dag", implementation=implementation, device=device, dtype=payload_dtype, preset=preset,
                        schedule="prefill", steps=1, warmup=1, windows_per_step=2, training=training, optimizer="adamw", **options)
    observed(candidate, logical)
    assert candidate["outputs"] == reference["outputs"] and candidate["final_cut"] == reference["final_cut"]
    torch.testing.assert_close(torch.tensor(candidate["losses"]), torch.tensor(reference["losses"]),
                              atol=2e-3 if payload_dtype=="float16" else 1e-6, rtol=2e-2 if payload_dtype=="float16" else 1e-5)
    (tmp_path/"memory.json").write_text(json.dumps(candidate))


def test_npu_peak_survives_free_then_resets():
    device = target();memory = MemoryRecord([device]);size = 4*1024*1024
    temporary = torch.empty(size, dtype=torch.float32, device=device)
    del temporary
    torch.npu.synchronize(device);memory.capture("construction")
    memory.capture("warmup");memory.capture("measured", reset_peak=False)
    phases = memory.record()["phases"]
    peak = phases[1]["devices"][0]
    assert peak["peak_allocated_bytes"]-peak["allocated_bytes"] >= size*4
    assert phases[2]["devices"][0]["peak_allocated_bytes"] == phases[2]["devices"][0]["allocated_bytes"]
