"""Failure-injected record persistence and real launcher/re-entry behavior."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("durable_records", ROOT/"scripts/durable_records.py")
records = importlib.util.module_from_spec(spec); spec.loader.exec_module(records)


@pytest.mark.parametrize("failure", ["serialize", "write", "file_fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_record_failure_preserves_previous_value_and_allows_retry(tmp_path, monkeypatch, failure, existing):
    path = tmp_path/"status.json"; old = {"state": "running"}; new = {"state": "passed"}
    if existing: records.write_json(path, old)
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs): raise OSError("injected persistence failure")
        if failure == "serialize": patch.setattr(records.json, "dumps", fail)
        elif failure == "file_fsync": patch.setattr(records.os, "fsync", fail)
        elif failure == "replace": patch.setattr(records.os, "replace", fail)
        else:
            original = records.os.fdopen
            class Partial:
                def __init__(self, *args, **kwargs): self.stream = original(*args, **kwargs)
                def __enter__(self): return self
                def __exit__(self, *args): self.stream.close()
                def write(self, value): self.stream.write(value[:4]); self.stream.flush(); fail()
            patch.setattr(records.os, "fdopen", Partial)
        with pytest.raises(OSError, match="injected"): records.write_json(path, new)
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    if existing: assert json.loads(path.read_text()) == old
    records.write_json(path, new); assert json.loads(path.read_text()) == new


def test_record_is_complete_at_replacement_and_directory_fsync_error_is_reported(tmp_path, monkeypatch):
    path = tmp_path/"status.json"; records.write_json(path, {"state": "running"})
    original = records.os.replace; count = 0
    def replace(source, target):
        assert json.loads(path.read_text())["state"] == "running"
        assert json.loads(Path(source).read_text())["state"] == "passed"
        original(source, target)
    def fsync(fd):
        nonlocal count
        count += 1
        if count == 2: raise OSError("injected directory fsync")
    with monkeypatch.context() as patch:
        patch.setattr(records.os, "replace", replace); patch.setattr(records.os, "fsync", fsync)
        with pytest.raises(OSError, match="directory fsync"): records.write_json(path, {"state": "passed"})
    assert count == 2 and json.loads(path.read_text())["state"] == "passed"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("code", [0, 7])
def test_job_launcher_persists_real_exit_and_preserves_existing_output(tmp_path, code):
    out = tmp_path/"job"
    command = [sys.executable, str(ROOT/"scripts/job.py"), "--output-dir", str(out), "--",
               sys.executable, "-c", f"print('workload executed', flush=True); raise SystemExit({code})"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == code, result.stderr
    status = json.loads((out/"status.json").read_text())
    assert status["state"] == ("passed" if code == 0 else "failed") and status["exit_code"] == code
    assert status["finished"] >= status["started"]
    assert (out/"task.log").read_text() == "workload executed\n"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode != 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("bad", ["{", "[]", '{"state":"passed"}',
                                  '{"state":"passed","exit_code":2,"started":"x","source":"y"}'])
def test_reentry_reports_broken_record_without_hiding_healthy_jobs(tmp_path, bad):
    root = tmp_path/"checkout"; (root/"scripts").mkdir(parents=True); (root/"docs").mkdir()
    for name in ("status.py", "durable_records.py"): shutil.copyfile(ROOT/"scripts"/name, root/"scripts"/name)
    (root/"docs/STATUS.md").write_text("Current handoff\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-c", "user.name=Record Test", "-c", "user.email=record@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "fixture"], cwd=root, check=True)
    for name in ("broken", "healthy"): (root/"artifacts"/name).mkdir(parents=True)
    (root/"artifacts/broken/status.json").write_text(bad)
    records.write_json(root/"artifacts/healthy/status.json", {"state": "passed", "source": "test",
                       "started": "2026-09-22T00:00:00Z", "exit_code": 0})
    result = subprocess.run([sys.executable, str(root/"scripts/status.py")], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert "unknown state" in result.stdout and "broken/status.json" in result.stdout
    assert 'healthy/status.json: {"state": "passed"' in result.stdout
    assert "Current handoff" in result.stdout


def test_explicit_terminal_postmortem_has_no_invented_start_time(tmp_path):
    folder = tmp_path/"failed-launcher"; folder.mkdir()
    record = {"state": "failed", "exit_code": 1, "source": "observed-source",
              "record_kind": "postmortem from observed terminal systemd status",
              "observed": "2026-09-21T20:57:39Z", "error": "disk full before workload start"}
    records.write_json(folder/"status.json", record)
    actual, errors = records.read_jobs(tmp_path)
    assert not errors and actual == [(folder/"status.json", record)]
    assert "started" not in actual[0][1]
