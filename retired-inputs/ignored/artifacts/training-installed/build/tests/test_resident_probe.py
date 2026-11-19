import json
import os
import signal
import subprocess
import sys
import time

import pytest

from scripts.probe_resident import run_bounded


def test_resident_probe_timeout_reaps_only_its_child(tmp_path):
    result = run_bounded([sys.executable, "-c", "import time; time.sleep(30)"],
                         stdout=tmp_path/"stdout", stderr=tmp_path/"stderr", seconds=.1)
    assert result["timed_out"] and result["return_code"] != 0
    assert result["cleanup_complete"]
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)


@pytest.mark.parametrize("interrupt", [False, True])
def test_bounded_probe_reaps_orphan_in_nested_session_and_records_terminal_state(tmp_path, interrupt):
    marker, lifecycle = tmp_path/"nested.pid", tmp_path/"lifecycle.json"
    nested = ("import os,signal,time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)")
    root = ("import os,subprocess,sys; "
            f"subprocess.Popen([sys.executable,'-c',{nested!r}],start_new_session=True); os._exit(0)")
    supervisor = f"""
import os,signal,sys
from scripts.job import enable_subreaper
from scripts.probe_resident import run_bounded
enable_subreaper()
def interrupted(signum, frame):
    raise KeyboardInterrupt('probe interrupted')
signal.signal(signal.SIGTERM, interrupted)
run_bounded([sys.executable, '-c', {root!r}], stdout={str(tmp_path/'stdout')!r},
            stderr={str(tmp_path/'stderr')!r}, seconds={10 if interrupt else .5},
            supervisor_pid=os.getpid(), terminate_grace=.1, record_path={str(lifecycle)!r})
"""
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    process = subprocess.Popen([sys.executable, "-c", supervisor], start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and process.poll() is None and time.monotonic()<deadline:
            time.sleep(.01)
        assert marker.exists()
        if interrupt:
            process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        record = json.loads(lifecycle.read_text())
        assert record["state"] == ("interrupted" if interrupt else "timed_out")
        assert record["timed_out"] is not interrupt
        assert record["cleanup_complete"] and record["sent_sigkill"]
        nested_pid = int(marker.read_text())
        assert any(row["pid"] == nested_pid and row["start_ticks"] > 0 for row in record["descendant_identities"])
        with pytest.raises(ProcessLookupError):
            os.kill(nested_pid, 0)
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_actual_model_fullgraph_attempt_records_scope_and_no_route_substitution(tmp_path):
    path = tmp_path/"result.json"
    command = [sys.executable, "-m", "scripts.probe_resident", "--device", "cpu",
               "--worker-config", "configs/fc-k.json", "--setting", "strict", "--width", "64",
               "--batch", "1", "--length", "2", "--output", str(path)]
    result = run_bounded(command, stdout=tmp_path/"stdout", stderr=tmp_path/"stderr", seconds=30)
    assert not result["timed_out"] and result["return_code"] == 0
    assert result["cleanup_complete"]
    observation = json.loads(path.read_text())
    assert observation["backend"] == "eager"
    assert observation["evidence_scope"].startswith("frontend_only")
    assert not observation["resident_qualified"] and not observation["precomputed_route_supplied"]
    assert not observation["dense_full_substitution"]
    assert len(observation["eager_inputs"]) == 2
    assert "construction-only" in observation["host_decision_scan_scope"]
    assert observation["state"] in {"capture_blocked", "captured", "captured_with_deviation"}
