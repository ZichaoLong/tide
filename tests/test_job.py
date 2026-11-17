"""Small real subprocess checks for the durable workload runner."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


RUNNER=Path(__file__).resolve().parents[1]/"scripts/job.py"


def launch(tmp_path,code,limit=None):
    status=tmp_path/"status.json"
    command=[sys.executable,str(RUNNER),"--status",str(status),"--source","test-source",
             "--poll-interval","0.02","--terminate-grace","0.1"]
    if limit is not None:
        command += ["--max-rss-gib",str(limit)]
    process=subprocess.Popen(command+["--",sys.executable,"-c",code],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    return process,status


def cleanup(process,status):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        if status.exists():
            child=json.loads(status.read_text()).get("child_pid")
            if child:
                try:
                    os.killpg(child,signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.kill()
        process.communicate(timeout=2)


def completed(process,status):
    try:
        stdout,stderr=process.communicate(timeout=8)
    finally:
        cleanup(process,status)
    record=json.loads(status.read_text())
    assert record["source"]=="test-source"
    assert record["elapsed"]>=0 and "finished" in record
    assert record["rss_sample_count"]>0
    assert not list(status.parent.glob(".status.*"))
    return record,stderr.decode()


def wait_file(path,process):
    deadline=time.monotonic()+5
    while not path.exists() and time.monotonic()<deadline and process.poll() is None:
        time.sleep(.02)
    assert path.exists(),"child did not publish its readiness marker"


def live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[0]!="Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("code",[0,7])
def test_runner_preserves_normal_exit_and_default_limit(tmp_path,code):
    process,status=launch(tmp_path,f"raise SystemExit({code})")
    record,stderr=completed(process,status)
    assert process.returncode==code,stderr
    assert record["state"]==("passed" if code==0 else "failed")
    assert record["exit_code"]==code
    assert record["max_rss_bytes"]==64*2**30
    assert record["process_group"]==record["child_pid"]!=process.pid


def test_runner_stops_child_above_small_rss_bound(tmp_path):
    process,status=launch(tmp_path,"import time; time.sleep(20)",limit=1/1024)
    record,stderr=completed(process,status)
    assert process.returncode==1,stderr
    assert record["state"]=="failed" and record["reason"]=="max_rss_exceeded"
    assert record["peak_rss_bytes"]>record["max_rss_bytes"]
    assert record["cleanup_complete"] and not live(record["child_pid"])


def test_runner_counts_nested_child_rss_and_stops_the_group(tmp_path):
    marker=tmp_path/"nested.pid"
    nested=("import os,time; from pathlib import Path; "
            f"Path({str(marker)!r}).write_text(str(os.getpid())); "
            "memory=bytearray(8*1024*1024); time.sleep(20)")
    code=("import subprocess,sys,time; "
          f"child=subprocess.Popen([sys.executable,'-c',{nested!r}]); time.sleep(20)")
    process,status=launch(tmp_path,code,limit=24/1024)
    record,stderr=completed(process,status)
    assert process.returncode==1,stderr
    assert record["reason"]=="max_rss_exceeded" and record["max_processes"]>=2
    assert record["peak_rss_bytes"]>24*2**20 and record["cleanup_complete"]
    assert marker.exists() and not live(int(marker.read_text()))
    assert not live(record["child_pid"])


def test_sigterm_escalates_only_own_group_and_records_termination(tmp_path):
    nested_ready=tmp_path/"nested.pid"
    nested=("import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(nested_ready)!r}).write_text(str(os.getpid())); time.sleep(20)")
    code=("import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
          f"child=subprocess.Popen([sys.executable,'-c',{nested!r}]); time.sleep(20)")
    other=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"],start_new_session=True)
    process,status=launch(tmp_path,code)
    try:
        wait_file(nested_ready,process)
        process.send_signal(signal.SIGTERM)
        record,stderr=completed(process,status)
        assert process.returncode==128+signal.SIGTERM,stderr
        assert record["state"]=="terminated" and record["signal"]==signal.SIGTERM
        assert record["sent_sigkill"] and record["cleanup_complete"]
        assert not live(record["child_pid"]) and not live(int(nested_ready.read_text()))
        assert other.poll() is None
    finally:
        cleanup(process,status)
        other.terminate()
        other.wait(timeout=3)
