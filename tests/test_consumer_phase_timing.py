"""Optional timing must preserve complete updates and continued sample slices."""
import json
import math
import os
import subprocess
import sys
import pytest
import torch
from test_online_consumer import ROOT, packet
from test_online_consumer_npu import target
from test_consumer_sample_chunks import compare_records
from online_consumer_support import observer, same
from tools.online_bench.host import run


def timing(record, enabled, training):
    phases = record["phase_timing"]
    assert phases["enabled"] is enabled
    for name, total_name, count in (("measured", "seconds", 2), ("warmup", "warmup_seconds", 1)):
        assert len(record[total_name]) == count
        assert len(phases[name]) == (count if enabled else 0)
        for row, total in zip(phases[name], record[total_name]):
            sample, update = row["sample_work_seconds"], row["optimizer_seconds"]
            assert math.isfinite(sample) and sample > 0
            assert math.isfinite(update) and (update > 0 if training else update == 0)
            assert math.isclose(sample+update, total, rel_tol=1e-12, abs_tol=1e-12)


def cli(p, implementation, device, family, schedule, preset, dtype, training, devices, enabled, path):
    path.mkdir()
    source = path/"packet.json"
    source.write_text(json.dumps(p))
    output = path/"run"
    command = [sys.executable, str(ROOT/"scripts/run_execution_flow.py"), "--packet", str(source),
               "--output-dir", str(output), "--device", str(device), "--dtype", dtype,
               "--family", family, "--implementation", implementation, "--preset", preset,
               "--schedule", schedule, "--optimizer", "adamw", "--steps", "2", "--warmup", "1",
               "--windows-per-step", "2", "--sample-chunk-rows", "1", "--diagnostics"]
    if training:
        command.append("--training")
    if enabled:
        command.append("--phase-timing")
    if implementation == "libtorch":
        command.extend(("--native-binary", os.environ["TIDE_ONLINE_BINARY"]))
    elif implementation == "native":
        command.extend(("--native-library", os.environ["TIDE_BUILD_DIR"]))
    if preset == "resident":
        command.extend(("--devices", str(devices), "--chunk-policy", "aggressive",
                        "--resident-context-bytes", str(64*1024**2)))
        if implementation == "native":
            command.extend(("--resident-library", os.environ["TIDE_RESIDENT_LIBRARY"]))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        log = output/"consumer.log"
        pytest.fail(completed.stdout+completed.stderr+(log.read_text() if log.exists() else ""))
    record = json.loads((output/"result.json").read_text())
    assert record["state"] == "passed"
    diagnostics = output/("consumer/diagnostics.jsonl" if implementation == "libtorch" else "diagnostics.jsonl")
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    return record, rows


def compare(implementation, device, family, schedule, preset, dtype, training, devices, tmp_path):
    if implementation == "libtorch" and not os.environ.get("TIDE_ONLINE_BINARY"):
        pytest.skip("standalone consumer not explicitly selected")
    if implementation == "native" and not os.environ.get("TIDE_BUILD_DIR"):
        pytest.skip("native consumer not explicitly selected")
    p = packet("add" if family == "pdg" else "attention")
    expected = []
    reference = run(p, family=family, implementation="python", device="cpu", schedule="streaming",
                    dtype="float64" if dtype == "float64" else "float32",
                    training=training, optimizer="adamw", steps=2, warmup=1, windows_per_step=2,
                    diagnostics=True, observer=observer(expected))
    timing(reference, False, training)
    previous = previous_rows = None
    tolerance = dict(atol=2e-3, rtol=2e-2) if dtype == "float16" else dict(atol=1e-6, rtol=1e-5)
    for enabled in (False, True):
        record, rows = cli(p, implementation, device, family, schedule, preset, dtype, training,
                           devices, enabled, tmp_path/("enabled" if enabled else "default"))
        timing(record, enabled, training)
        # Compare all states, messages, history, pending routes and None/zero
        # gradients, and every parameter generation, with independent CPU runs.
        compare_records(rows, expected, p["workload"]["batch"], tensor_norm=dtype=="float16", **tolerance)
        for key in ("outputs", "parameters", "final_cut", "input_tokens_per_step"):
            assert record[key] == reference[key], key
        torch.testing.assert_close(torch.tensor(record["losses"]), torch.tensor(reference["losses"]), **tolerance)
        assert record["batch_execution"]["physical_chunks"] == 2
        if previous is not None:
            same(rows, previous_rows)
            for key in ("losses", "outputs", "statistics", "final_cut", "batch_execution"):
                assert record[key] == previous[key], key
        previous, previous_rows = record, rows


@pytest.mark.parametrize("implementation", ["python", "native", "libtorch"])
@pytest.mark.parametrize("training", [False, True])
def test_cpu_phase_timing(implementation, training, dtype, tmp_path):
    compare(implementation, "cpu", "settle" if training else "timed-dag",
            "prefill" if training else "streaming", "cpu", str(dtype).split(".")[-1], training, 1, tmp_path)


@pytest.mark.parametrize("implementation,preset,family,schedule,payload_dtype,training,devices", [
    ("python", "mixed-a", "timed-dag", "prefill", "float32", True, 1),
    ("native", "mixed-c", "settle", "streaming", "float32", True, 1),
    ("libtorch", "mixed-b", "pdg", "prefill", "float32", True, 1),
    ("python", "mixed-b", "settle", "streaming", "float32", False, 1),
    ("native", "resident", "timed-dag", "prefill", "float32", True, 2),
    ("libtorch", "resident", "pdg", "streaming", "float16", True, 2),
    ("native", "resident", "settle", "streaming", "float16", False, 2),
    ("libtorch", "resident", "timed-dag", "prefill", "float32", False, 1),
])
def test_npu_phase_timing(implementation, preset, family, schedule, payload_dtype, training, devices, tmp_path):
    compare(implementation, target(), family, schedule, preset, payload_dtype, training, devices, tmp_path)


def test_phase_timing_requires_boolean():
    for value in (1, "false", None):
        with pytest.raises(ValueError, match="phase-timing must be boolean"):
            run(packet(), family="settle", implementation="python", device="cpu", phase_timing=value)
