"""V2 process boundary, phase denominators and policy manifests."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_v2_smoke_records_actual_schedules_and_modes(tmp_path):
    build = Path(os.environ.get("TIDE_BUILD_DIR", ROOT/"build")).resolve()
    out = tmp_path/"suite"
    command = [sys.executable, str(ROOT/"scripts/benchmark_foundation.py"), "--device", "cpu",
               "--suite", str(ROOT/"benchmarks/foundation-v2.json"), "--tier", "smoke",
               "--ids", "T01", "S01", "TR01", "TR02", "--variants", "native-frontier-optimized", "native-settle-optimized",
               "--build-dir", str(build), "--output-dir", str(out), "--workers", "2", "--warmup", "0",
               "--tracking", "off", "--allow-dirty-smoke"]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=100)
    assert process.returncode == 0, process.stdout+process.stderr
    suite = json.loads((out/"suite.json").read_text())
    assert suite["state"] == "evaluated" and suite["completed_runs"] == 4 and suite["failed_runs"] == 0
    for run in suite["runs"]:
        directory = out/run["directory"]
        record = json.loads((directory/"run.json").read_text())
        worker = json.loads((directory/"worker/worker.json").read_text())
        assert record["project"] == "tide-foundation-v2"
        assert "attention_packing" not in worker["options"]
        assert worker["execution_policy"]["resolved"]["schedule"]["aggregate_autograd"] == "batched"
        measured = json.loads((directory/"worker/measured.json").read_text())
        if run["config"]["execution_pattern"] == "prefill-stream":
            segments = measured["segments"]
            assert [s["effective_input_positions"] for s in segments] == [16, 4, 4]
            assert segments[0]["execution_paths"]["max_state_sequence"] == 4
            assert not segments[-1]["execution_paths"]["effective_schedule"]["prefill"]
            metrics = run["metrics"]
            assert metrics["perf/nograd-forward-prefill/seconds"] > 0
            assert metrics["perf/nograd-forward-streaming/seconds"] > 0
        else:
            assert measured["optimizer_updates"] == 2
            assert not measured["observations"]["stats"].get("semantic_full_replays", 0)
            import hashlib
            import torch
            from tidegraph.compare import equivalent
            snapshots = []
            for phase in ("measured", "profile"):
                saved = worker["training_observations"][phase]
                path = directory/"worker"/saved["file"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"]
                snapshots.append(torch.load(path, weights_only=True))
            equivalent(*snapshots)
