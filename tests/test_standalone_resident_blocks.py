"""Fixed-build standalone block qualification; CPU collection never starts NPU.

Supply TIDE_WEIGHTED_BENCHMARK, TIDE_BUILD_DIR and an explicitly admitted
TIDE_STANDALONE_BLOCK_DEVICE. Complete event/root qualification is separate.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from tidegraph.planning.topology import topology
from tidegraph.planning.workload import make_plan

ROOT = Path(__file__).resolve().parents[1]


def execute(*args):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def archive_fields(run):
    root = run / "checkpoint"
    generation = json.loads((root / "latest.json").read_text())["generation"]
    module = torch.jit.load(str(root / generation / "state.pt"), map_location="cpu")
    return dict(module.named_parameters()) | dict(module.named_buffers())


@pytest.mark.parametrize("read", ["sum", "qry"])
def test_standalone_block_task_and_fresh_process_resume(tmp_path, read):
    binary = os.environ.get("TIDE_WEIGHTED_BENCHMARK")
    device = os.environ.get("TIDE_STANDALONE_BLOCK_DEVICE")
    if not binary or not device:
        pytest.skip("standalone block qualification requires an admitted device and matched build")
    from tidegraph.experiment.records import write_json
    plan, _ = make_plan(topology("fc", depth=2, region_width=2), family="kv", memory_read=read,
                        width=4, samples=2, positions=3)
    write_json(tmp_path / "plan.json", plan)
    execute(sys.executable, ROOT / "scripts/export_weighted_benchmark.py", "--native-library",
            os.environ["TIDE_BUILD_DIR"], "--workload", tmp_path / "plan.json", "--binary",
            "--output", tmp_path / "fixture.json")
    common = [binary, "--fixture", tmp_path / "fixture.json", "--device", device,
              "--implementation", "resident", "--resident-executor", "auto", "--dtype", "float32",
              "--warmup", "0", "--threads", "1", "--return-mode", "outputs", "--validation", "none"]

    def run(name, *args):
        execute(*common, "--output-dir", tmp_path / name, *args)
        result = json.loads((tmp_path / name / "result.json").read_text())
        assert result["effective_execution"]["effective"] == "block-v2"
        assert result["scheduler"] == "settle-block-v2"
        assert result["task_loss_computed"] and not result["independent_complete_records_checked"]
        return result

    whole = run("whole", "--steps", "3", "--mode", "training", "--checkpoint-check", "true")
    assert whole["exact_resume"]["state"] == "passed"
    run("prefix", "--steps", "1", "--mode", "training")
    resumed = run("resumed", "--steps", "2", "--mode", "training", "--resume", tmp_path / "prefix/checkpoint",
                  "--checkpoint-check", "true")
    assert resumed["start_window"] == 1 and resumed["exact_resume"]["state"] == "passed"
    expected, actual = archive_fields(tmp_path / "whole"), archive_fields(tmp_path / "resumed")
    assert expected and expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    outputs = run("outputs", "--steps", "1", "--mode", "inference")
    metrics = run("metrics", "--steps", "1", "--mode", "inference", "--return-mode", "metrics")
    assert outputs["metrics"][0]["loss"] == metrics["metrics"][0]["loss"]
    for key in ("candidate_updates", "full_calls", "messages", "outputs", "pending"):
        assert outputs["metrics"][0][key] == metrics["metrics"][0][key]
