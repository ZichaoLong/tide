"""The independent C++ accelerator gate must also preserve both CPU baselines."""
import json
import os
from pathlib import Path
import subprocess


def test_standalone_graph_training_checkpoint(dtype, tmp_path):
    build = Path(os.environ.get("TIDE_BUILD_DIR", "build")).resolve()
    out = tmp_path / "gate"
    subprocess.run([str(build / "tidegraph-accelerator-check"), "--device", "cpu",
                    "--dtype", str(dtype).split(".")[-1], "--output-dir", str(out)], check=True)
    report = json.loads((out / "result.json").read_text())
    assert report["state"] == "passed" and report["checkpoint_handoff"]
