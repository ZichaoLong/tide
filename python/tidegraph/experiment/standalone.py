"""Launch one real independent C++ client from the installed public CLI."""
from pathlib import Path
import os
import subprocess
import signal
import statistics
import time
from .records import now, read_json, write_json


def run(resolved, output, *, resume=None, validation_override=None, check_roots=False):
    from .identity import capture, verify
    from .site import launch_environment
    from .export import material
    e, m, paths = resolved["execution"]["effective"], resolved["measurement"], resolved["site"]["paths"]
    if "standalone_binary" not in paths:
        raise ValueError("site.paths.standalone_binary: specify the matching standalone LibTorch executable")
    if resolved["task"].get("plugin"):
        raise ValueError("task.plugin: Python callbacks cannot run inside standalone LibTorch; use the public C++ consumer task extension")
    if e.get("return_mode", "complete") != "complete" and m["checkpoint_check"]:
        raise ValueError("measurement.checkpoint_check: standalone minimal-return run needs false; perform separate complete-return check first")
    if m.get("observation") == "profile":
        raise ValueError("measurement.observation: use a separate native/PyTorch profile or an external C++ profiler; packing observation is available for standalone")
    identity = capture(paths)
    # Validate the immutable dependency/config identity before a new run path.
    resume_path = None
    if resume is not None:
        resume_path = Path(resume).resolve()
        parent = resume_path if (resume_path/"identity.json").is_file() else resume_path.parent.parent
        previous = read_json(parent/"resolved.json")
        for key in ("model_id", "workload_id", "task_id", "execution_id"):
            if previous[key] != resolved[key]:
                raise ValueError("resume: standalone "+key+" differs")
        verify(read_json(parent/"identity.json"), paths)
        if (resume_path/"client"/"checkpoint").is_dir():
            resume_path = resume_path/"client"/"checkpoint"
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"resolved.json", resolved)
    write_json(out/"identity.json", identity)
    policy = m["validation"] if validation_override is None else validation_override
    record = dict(schema="tide-experiment-result-v1", state="running", started=now(), execution_status="not_started",
        resolved_config=resolved, **{k: resolved[k] for k in ("model_id", "workload_id", "task_id", "execution_id", "config_id")},
        client="libtorch", mode=m["mode"], identity=identity, validation=dict(policy=policy, outcome="not_checked",
        numeric="not_checked", discrete="not_checked", complete="not_checked", strict_passed=False),
        checkpoint=dict(state="not_checked"), metrics=[], exploratory_policy=policy == "explore",
        timing_contract=dict(return_mode=e.get("return_mode", "complete"), client="independent C++ process",
                             included="C++ window preparation through synchronization; see client/result.json",
                             excluded="Python static material export, process launch, construction, validation and checkpoint"))
    write_json(out/"result.json", record)
    start = time.perf_counter()
    try:
        # Only the controller's matched stack is imported here. The child gets
        # the independently declared SDK environment after material publication.
        from .engine import _configure
        from .admission import preflight
        record["controller_runtime"] = _configure(resolved)
        record["admission"] = preflight(resolved, out)
        if record["admission"]["refused"]:
            raise ValueError("capacity preflight: "+"; ".join(record["admission"]["refused"]))
        began = time.perf_counter()
        fixture = material(resolved, out/"material.json")
        record["material_export_seconds"] = time.perf_counter()-began
        record["fixture_sha256"] = fixture["fixture_sha256"]
        command = [paths["standalone_binary"], "--fixture", str(out/"material.json"), "--device", e["device"],
            "--output-dir", str(out/"client"), "--mode", m["mode"], "--schedule", e["schedule"],
            "--steps", str(m["steps"]), "--warmup", str(m["warmup"]), "--dtype", e["dtype"], "--scoring-dtype", e["scoring_dtype"],
            "--implementation", "resident" if e["placement"] == "resident" else "native",
            "--placement", "native" if e["placement"] in {"resident", "cpu"} else e["placement"],
            "--packing", e["packing"], "--return-packing", e["return_packing"], "--return-mode", e.get("return_mode", "complete"),
            "--validation", policy, "--threads", str(e["intraop_threads"]), "--interop-threads", str(e["interop_threads"]),
            "--workers", str(e["workers"]), "--communication", e.get("communication", "ordinary"),
            "--checkpoint-check", str(m["checkpoint_check"]).lower(),
            "--checkpoint-max-bytes", str(resolved["resources"]["checkpoint_max_bytes"]),
            "--loss-scale", str(resolved["task"]["objective"]["scale"])]
        for key in ("lr", "momentum", "weight_decay"):
            command += ["--"+key.replace("_", "-"), str(resolved["task"]["optimizer"][key])]
        if e["peer_devices"]:
            command += ["--peer-devices", ",".join(e["peer_devices"])]
        if e.get("node_devices"):
            command += ["--node-devices", ",".join(e["node_devices"])]
        if e["placement"] == "resident":
            write_json(out/"resident-limits.json", resolved["resources"]["resident_limits"])
            command += ["--resident-limits", str(out/"resident-limits.json")]
        if resume_path is not None:
            command += ["--resume", str(resume_path)]
        environment = launch_environment(resolved["site"], resolved["execution"])
        environment.update(resolved["site"].get("standalone_environment", {}))
        if m.get("observation") == "packing":
            command += ["--observe-work", "true"]
        if check_roots:
            command += ["--check-roots", "true"]
        record["command"] = command
        record["child_environment"] = environment
        write_json(out/"result.json", record)
        with (out/"client.log").open("w") as log:
            child_process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            try:
                code = child_process.wait(timeout=m["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(child_process.pid, signal.SIGTERM)
                try:
                    child_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child_process.pid, signal.SIGKILL)
                    child_process.wait()
                raise
        record["client_exit_code"] = code
        child = read_json(out/"client"/"result.json")
        record["client_result"] = "client/result.json"
        if "error" in child:
            record["error"] = dict(type="StandaloneError", message=child["error"], details=child.get("comparison_failure"))
        for key in ("validation", "checkpoint", "metrics", "warmup", "memory", "effective", "timing_summary",
                    "construction_seconds", "runtime", "execution_status", "strict_passed", "independent_roots",
                    "independent_roots_seconds", "throughput_sample", "throughput_qualified", "kind"):
            if key in child:
                record[key] = child[key]
        record["state"] = "failed" if code else child.get("state", "completed")
        if record["state"] in {"exploratory", "unchecked"}:
            record["state"] = "completed"
        record["execution_status"] = "completed" if child.get("execution_completed") else "failed"
        if child.get("independent_complete_records_checked"):
            from .validation import outcome
            comparison = child["independent_comparison"]
            record["validation"] = dict(comparison, complete="checked", policy=policy, outcome=outcome(comparison, policy))
        record["strict_passed"] = record["validation"]["outcome"] == "passed"
        if child.get("exact_resume", {}).get("state") == "passed":
            record["checkpoint"].update(state="passed", exact_resume=True, resume_check=child["exact_resume"])
        elif child.get("phase") == "exact-resume" and code:
            record["checkpoint"].update(state="failed", exact_resume=False)
        record["warmup"] = child.get("warmup_metrics", [])
        times = [row["seconds"] for row in record["metrics"]]
        if times:
            record["timing_summary"] = dict(median_seconds=statistics.median(times), total_seconds=sum(times), measured_windows=len(times))
        verify(identity, paths)
    except BaseException as error:
        record.update(state="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record.update(finished=now(), total_wall_seconds=time.perf_counter()-start)
        write_json(out/"result.json", record)
    return record
