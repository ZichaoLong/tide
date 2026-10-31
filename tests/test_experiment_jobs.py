import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest
from tidegraph.experiment import container_job, jobs
from tidegraph.experiment.records import read_json


def environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]/"python")
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    return env


def terminal(out):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        record = container_job.inspect(out)
        if record["state"] in container_job.TERMINAL:
            return record
        time.sleep(.05)
    raise AssertionError("short job failed to reach a terminal state")


def test_container_stop_drains_descendant_in_a_new_session(tmp_path):
    out = tmp_path/"job"
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); time.sleep(30)"
    container_job.start_command(out, command=[sys.executable, "-c", code], cwd=tmp_path,
                                timeout=10, grace=.2, environment=environment())
    try:
        deadline = time.monotonic()+5
        while not (out/"task.log").read_text().strip() and time.monotonic() < deadline:
            time.sleep(.05)
        grandchild = int((out/"task.log").read_text().strip())
        assert container_job.process(grandchild)["pgid"] == grandchild
        record = container_job.stop(out)
        assert record["state"] == "cancelled"
        assert record["remaining_live_pids"] == []
        info = container_job.process(grandchild)
        assert info is None or info["state"] in {"Z", "X"}
    finally:
        if container_job.inspect(out)["state"] not in container_job.TERMINAL:
            container_job.stop(out)


def test_timeout_is_failure_and_no_orphan_survives(tmp_path):
    out = tmp_path/"timeout"
    container_job.start_command(out, command=[sys.executable, "-c", "import time; time.sleep(30)"],
                                cwd=tmp_path, timeout=.2, grace=.1, environment=environment())
    record = terminal(out)
    assert record["state"] == "failed"
    assert record["exit_code"] == 124
    assert record["remaining_live_pids"] == []


def test_detached_job_survives_submission_process_exit(tmp_path):
    out = tmp_path/"detached"
    code = ("from tidegraph.experiment.container_job import start_command; import sys; "
            "start_command(sys.argv[1],command=[sys.executable,'-c','import time; time.sleep(30)'],"
            "cwd=sys.argv[2],timeout=10,grace=.1)")
    subprocess.run([sys.executable, "-c", code, str(out), str(tmp_path)], env=environment(),
                   capture_output=True, text=True, check=True, timeout=5)
    try:
        record = container_job.inspect(out)
        assert record["observed_supervisor_alive"]
        assert record["observed_live_workload_pids"]
    finally:
        assert container_job.stop(out)["remaining_live_pids"] == []


def test_systemd_argv_has_separate_background_scope_and_explicit_environment(tmp_path):
    argv = jobs.systemd_command(tmp_path, "tidegraph-1234567890abcdef.service", {"OMP_NUM_THREADS": "4"}, 30, 5)
    assert "--slice=background.slice" in argv
    assert "--property=KillMode=control-group" in argv
    assert "--service-type=exec" in argv
    assert "--setenv=OMP_NUM_THREADS=4" in argv
    assert not any("focus.service" in value or value == "--scope" for value in argv)


def test_missing_supervisor_is_unconfirmed_not_passed(tmp_path):
    out = tmp_path/"missing"
    container_job.prepare(out, command=[sys.executable, "-V"], cwd=tmp_path, timeout=5)
    record = jobs.status(out)
    assert record["state"] == "starting"
    assert record["observed_state"] == "unconfirmed"
    with pytest.raises(RuntimeError, match="supervisor identity missing"):
        container_job.stop(out)


def test_matrix_logs_show_current_child_without_hiding_its_failure(tmp_path):
    from tidegraph.experiment.records import write_json
    out = tmp_path/"job"
    current = out/"run"/"case-b"
    current.mkdir(parents=True)
    (out/"task.log").write_text("")
    (current/"task.log").write_text("started\nNumericMismatch: original threshold\n")
    write_json(out/"run"/"result.json", dict(schema="tide-experiment-matrix-result-v1",
        state="running", current_case="case-b", cases=[dict(name="case-a", state="passed")]))
    text = jobs.logs(out, lines=1)
    assert str(current/"task.log") in text
    assert "NumericMismatch" in text and "started" not in text
    write_json(out/"run"/"result.json", dict(schema="tide-experiment-matrix-result-v1", current_case="../other"))
    with pytest.raises(ValueError, match="current_case"):
        jobs.logs(out)


def test_status_retains_strict_failure_and_first_difference_without_dumping_every_report(tmp_path):
    from tidegraph.experiment.records import write_json
    out = tmp_path/"job"
    container_job.prepare(out, command=[sys.executable, "-V"], cwd=tmp_path, timeout=5)
    (out/"run").mkdir()
    first = dict(kind="numeric", path="state.keys", atol=1e-6)
    validation = dict(numeric="failed", discrete="passed", strict_passed=False,
        reports=[dict(mismatch_counts=dict(numeric=2, discrete=0), differences=[first]),
                 dict(mismatch_counts=dict(numeric=3, discrete=0), differences=[])])
    write_json(out/"run"/"result.json", dict(state="completed", execution_status="completed", validation=validation))
    summary = jobs.status(out)["experiment"]["validation"]
    assert summary["numeric"] == "failed" and summary["strict_passed"] is False
    assert summary["mismatch_counts"] == dict(numeric=5, discrete=0)
    assert summary["first_difference"] == first and "reports" not in summary
    assert read_json(out/"run"/"result.json")["validation"] == validation
