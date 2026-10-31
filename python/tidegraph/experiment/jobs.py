"""Human job lifecycle over the installed CLI and the shared subreaper."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from . import container_job
from .config import resolve
from .identity import capture, verify
from .records import now, read_json, write_json
from .site import launch_environment


def _systemd(unit, *options):
    if not re.fullmatch(r"tidegraph-[a-f0-9]{16}\.service", unit):
        raise ValueError("invalid tidegraph transient unit identity")
    return subprocess.run(["systemctl", "--user", *options, unit], capture_output=True, text=True, timeout=30)


def systemd_state(unit):
    result = _systemd(unit, "show", "-p", "Id", "-p", "ActiveState", "-p", "SubState", "-p", "Result",
                      "-p", "ExecMainStatus", "-p", "MainPID", "-p", "Slice", "-p", "ControlGroup", "-p", "Transient")
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    values["query_exit_code"] = result.returncode
    return values


def systemd_command(out, unit, environment, timeout, grace):
    """Publicly reviewable argv, never a shell string or inherited focus scope."""
    out = Path(out).resolve()
    command = ["systemd-run", "--user", "--unit="+unit, "--service-type=exec", "--slice=background.slice",
               "--description=Bounded tidegraph experiment", "--working-directory="+str(out),
               "--property=KillMode=control-group", "--property=Nice=10",
               f"--property=RuntimeMaxSec={timeout+grace+30}", f"--property=TimeoutStopSec={grace+20}",
               "--property=StandardOutput=append:"+str(out/"supervisor.log"),
               "--property=StandardError=append:"+str(out/"supervisor.log")]
    command.extend("--setenv="+key+"="+value for key, value in sorted(environment.items()))
    return [*command, sys.executable, "-m", "tidegraph.experiment.container_job", "_supervise", str(out)]


def start(config, output_dir, *, site=None, resume=None, matrix=False):
    from .matrix import resolve_matrix
    resolved = resolve_matrix(config, site=site) if matrix else resolve(config, site=site)
    launcher = resolved["site"]["launcher"]
    if launcher not in {"systemd", "container"}:
        raise ValueError("site.launcher: job start requires systemd or container; use run for foreground execution")
    if matrix and resume:
        raise ValueError("matrix resume: select the case and its checkpoint as a new explicit run")
    resume = str(Path(resume).resolve()) if resume else None
    if resume and not Path(resume).exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")
    identity = capture(resolved["site"]["paths"])
    from .task_identity import identity as task_identity
    task_identities = ([task_identity(row["config"]["task"]) for row in resolved["cases"]]
                       if matrix else [task_identity(resolved["task"])])
    execution = resolved["cases"][0]["config"]["execution"] if matrix else resolved["execution"]
    environment = launch_environment(resolved["site"], execution)
    out = Path(output_dir).resolve()
    command = [sys.executable, "-m", "tidegraph.experiment.jobs", "_worker", str(out)]
    timeout = resolved["timeout_seconds"] if matrix else resolved["measurement"]["timeout_seconds"]
    metadata = dict(launcher=launcher, config_id=resolved.get("config_id", resolved.get("matrix_id")),
                    task_kind="matrix" if matrix else "experiment", install_identity=identity,
                    environment=environment, result_dir=str(out/"run"), resume=resume,
                    task_identities=task_identities)
    if launcher == "systemd":
        metadata["unit"] = "tidegraph-"+uuid.uuid4().hex[:16]+".service"
    container_job.prepare(out, command=command, cwd=out, timeout=timeout, metadata=metadata)
    launching = False
    try:
        write_json(out/"resolved.json", resolved, exclusive=True)
        write_json(out/"identity.json", identity, exclusive=True)
        for path in (out/"resolved.json", out/"identity.json"):
            path.chmod(0o444)
        launching = True
        if launcher == "container":
            return container_job.detach(out, environment=environment)
        result = subprocess.run(systemd_command(out, metadata["unit"], environment, timeout, 5),
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            launching = False
            raise RuntimeError("systemd submission failed: "+result.stderr.strip())
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            record = status(out)
            if record["state"] != "starting":
                manager = record["systemd"]
                if manager.get("ActiveState") == "active" and (
                        manager.get("Slice") != "background.slice" or manager.get("Transient") != "yes"
                        or "focus.service" in manager.get("ControlGroup", "")):
                    raise RuntimeError("systemd job is not in its independent background.slice scope")
                return record
            time.sleep(.1)
        raise RuntimeError("systemd startup unconfirmed; inspect job status before retrying")
    except BaseException as error:
        record = read_json(out/"status.json")
        # Never relabel a live/uncertain submission as a terminal failed job.
        if record["state"] == "starting" and not launching:
            record.update(state="failed", finished=now(), exit_code=1, error=repr(error))
            write_json(out/"status.json", record)
        raise


def worker(out):
    """Recheck frozen inputs before importing Torch or invoking an independent C++ client."""
    out = Path(out)
    from .matrix import resolve_matrix, run_matrix
    requested = read_json(out/"resolved.json")
    matrix = requested.get("schema") == "tide-experiment-matrix-resolved-v1"
    resolved = resolve_matrix(requested) if matrix else resolve(requested)
    verify(read_json(out/"identity.json"), resolved["site"]["paths"])
    record = read_json(out/"status.json")
    from .task_identity import verify as verify_task
    tasks = [row["config"]["task"] for row in resolved["cases"]] if matrix else [resolved["task"]]
    expected_tasks = record["metadata"]["task_identities"]
    if len(tasks) != len(expected_tasks):
        raise ValueError("job task identity inventory changed")
    for task, expected in zip(tasks, expected_tasks):
        verify_task(task, expected)
    if matrix:
        result = run_matrix(resolved, out/"run")
    else:
        from .engine import run
        result = run(resolved, out/"run", resume=record["metadata"].get("resume"))
    print(json.dumps(dict(state=result.get("state"), execution_status=result.get("execution_status"),
                          validation=result.get("validation"), checkpoint=result.get("checkpoint"))))
    return 0 if result.get("state") in {"passed", "completed"} else 1


def status(output_dir):
    out = Path(output_dir).resolve()
    record = container_job.inspect(out)
    metadata = record.get("metadata", {})
    if metadata.get("launcher") == "systemd":
        record["systemd"] = systemd_state(metadata["unit"])
    record["observed_state"] = record["state"]
    if record["state"] not in container_job.TERMINAL and not record["observed_supervisor_alive"]:
        manager = record.get("systemd", {})
        if manager.get("ActiveState") != "activating":
            record["observed_state"] = "unconfirmed"
    result_file = out/"run"/"result.json"
    if result_file.is_file():
        result = read_json(result_file)
        record["experiment"] = {key: result.get(key, "unchecked") for key in (
            "state", "execution_status", "validation", "checkpoint", "config_id", "phase", "window")}
        if result.get("schema") == "tide-experiment-matrix-result-v1":
            record["experiment"].update(completed_cases=len(result["cases"]), current_case=result.get("current_case"))
            current = _case_path(out, result)
            if current is not None and (current/"run"/"result.json").is_file():
                child = read_json(current/"run"/"result.json")
                record["experiment"]["current_result"] = {key: child.get(key, "unchecked") for key in (
                    "state", "execution_status", "phase", "window", "validation", "checkpoint", "error")}
    return record


def stop(output_dir):
    out = Path(output_dir).resolve()
    record = status(out)
    if record.get("metadata", {}).get("launcher") == "systemd":
        unit = record["metadata"]["unit"]
        result = _systemd(unit, "stop")
        if result.returncode:
            raise RuntimeError("systemd stop failed: "+result.stderr.strip())
        record = status(out)
        if record["systemd"].get("ActiveState") in {"active", "activating", "deactivating"}:
            raise RuntimeError("systemd stop has not reached an inactive unit")
    else:
        record = container_job.stop(out)
    if record["state"] not in container_job.TERMINAL or record["observed_live_workload_pids"] or record.get("remaining_live_pids"):
        raise RuntimeError("stop did not establish terminal status and complete descendant cleanup")
    return record


def logs(output_dir, *, lines=80):
    if type(lines) is not int or lines < 1:
        raise ValueError("log line count must be positive")
    from collections import deque
    out = Path(output_dir)
    path = out/"task.log"
    result_file = out/"run"/"result.json"
    if result_file.is_file():
        current = _case_path(out, read_json(result_file))
        if current is not None and (current/"task.log").is_file():
            path = current/"task.log"
    with path.open(errors="replace") as stream:
        return f"Log: {path}\n"+"".join(deque(stream, maxlen=lines))


def _case_path(out, result):
    """Matrix logs stay in their original per-case files, without a relay."""
    if result.get("schema") != "tide-experiment-matrix-result-v1":
        return None
    current = result.get("current_case")
    if current is None and result.get("cases"):
        current = result["cases"][-1]["name"]
    if current is None:
        return None
    if not isinstance(current, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", current):
        raise ValueError("matrix.current_case: invalid case identity")
    return Path(out)/"run"/current


def resume(output_dir, new_output_dir, *, checkpoint=None):
    old = Path(output_dir).resolve()
    record = status(old)
    if record.get("metadata", {}).get("task_kind") == "matrix":
        raise ValueError("matrix resume: choose a case's resolved.json and checkpoint for job start --resume")
    if record["state"] not in container_job.TERMINAL or record["observed_live_workload_pids"]:
        raise ValueError("resume: previous job must be terminal with no live descendants")
    if checkpoint is None:
        # A stop may occur after publishing the latest generation and before
        # updating the human result summary. Use the durable pointer first.
        if (old/"run"/"checkpoints"/"latest.json").is_file():
            return start(read_json(old/"resolved.json"), new_output_dir, resume=old/"run"/"checkpoints")
        result = read_json(old/"run"/"result.json")
        checkpoint = result.get("checkpoint", {}).get("path")
        if not checkpoint:
            raise ValueError("resume: no checkpoint was recorded; supply --checkpoint PATH")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = old/"run"/checkpoint
    return start(read_json(old/"resolved.json"), new_output_dir, resume=checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("_worker",))
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    return worker(args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
