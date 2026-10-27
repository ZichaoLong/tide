"""Real child lifecycle checks for docs/weighted-performance.md startup records."""
import importlib.util
from pathlib import Path
import sys
import pytest


def observer():
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("weighted_process_observation", scripts/"measure_weighted_process.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.observe
    finally:
        sys.path.remove(str(scripts))


@pytest.mark.parametrize("markers,exit_code", [(1, 0), (0, 0), (2, 0), (1, 3)])
def test_readiness_does_not_replace_exit_status(tmp_path, markers, exit_code):
    program = f"print('TIDE_WEIGHTED_RUNTIME_READY_V1\\n'*{markers}, end='', flush=True); raise SystemExit({exit_code})"
    with (tmp_path/"child.log").open("wb") as log:
        result = observer()([sys.executable, "-c", program], log, 10)
    assert result["state"] == ("passed" if markers == 1 and exit_code == 0 else "failed")
    assert result["exit_code"] == exit_code
    assert result["ready_markers"] == markers
    if markers == 1:
        assert 0 < result["launch_to_runtime_ready_seconds"] <= result["process_seconds"]


def test_readiness_timeout_keeps_failure_and_log(tmp_path):
    program = "import time; print('TIDE_WEIGHTED_RUNTIME_READY_V1', flush=True); time.sleep(30)"
    path = tmp_path/"child.log"
    with path.open("wb") as log:
        result = observer()([sys.executable, "-c", program], log, 1)
    assert result["state"] == "failed" and result["timed_out"]
    assert result["exit_code"] != 0
    assert result["ready_markers"] == 1
    assert "TIDE_WEIGHTED_RUNTIME_READY_V1" in path.read_text()
