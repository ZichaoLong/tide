"""Standalone public C++ consumer, separate from Python/native binding tests."""
import os
from pathlib import Path
import subprocess


def test_standalone_greedy_entry(dtype):
    build = Path(os.environ.get("TIDE_BUILD_DIR", Path(__file__).resolve().parents[1] / "build"))
    result = subprocess.run([str(build / "tidegraph-greedy-check"), "--device=cpu",
                             f"--dtype={str(dtype).split('.')[-1]}"],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "standalone-greedy: passed" in result.stdout
