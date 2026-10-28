"""Complete workload diagnostics include optimizer slots and exact next resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from weighted_topology import topology
from weighted_workload_plan import make_plan
from durable_records import write_json


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("implementation", ["python", "native"])
def test_complete_workload_optimizer_and_checkpoint(tmp_path, precision, implementation):
    if implementation == "native" and not os.environ.get("TIDE_BUILD_DIR"):
        pytest.skip("matching native library required")
    plan, _ = make_plan(topology("fc", depth=2, region_width=2), width=4, samples=1, positions=2)
    write_json(tmp_path/"plan.json", plan)
    out = tmp_path/"diagnostic"
    command = [sys.executable, str(ROOT/"scripts/diagnose_weighted_workload.py"), "--workload", str(tmp_path/"plan.json"),
               "--dtype", precision, "--implementation", implementation, "--steps", "2", "--checkpoint-check", "--output-dir", str(out)]
    if implementation == "native":
        command += ["--native-library", os.environ["TIDE_BUILD_DIR"]]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    record = json.loads((out/"result.json").read_text())
    assert record["state"] == "passed" and record["steps"][0]["checkpoint_exact_decode"]
    assert all(step["optimizer"] is None for step in record["steps"])
    assert record["steps"][1]["exact_resume"] == [dict(records=None, gradients=None, parameters=None, optimizer=None)]*2
